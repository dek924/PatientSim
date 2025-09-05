import os
import re
import sys
import time
import json
import random
import logging
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
from models import get_response_method, vllm_model_setup, get_answer
from utils import file_to_string, save_to_dialogue, save_to_json, log_and_print, set_seed, prompt_valid_check, detect_termination


class ScenarioLoaderMIMICIV:
    def __init__(self, data_dir, data_name="sample_info") -> None:
        with open(os.path.join(data_dir, f"{data_name}.json"), "r") as f:
            self.scenario_dict = json.load(f)
        self.num_scenarios = len(self.scenario_dict)
        log_and_print(f"Load {self.num_scenarios} scenarios from {data_dir}")

    def sample_scenario(self):
        return self.scenario_dict[random.randint(0, self.num_scenarios - 1)]

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenario_dict[id]


class PatientAgent:
    def __init__(
        self,
        patient_profile,
        backend_str="gpt4",
        backend_api_type="gpt_azure",
        temperature=0,
        random_seed=42,
        prompt_dir=None,
        prompt_file=None,
        num_word_sample=3,
    ):
        self.prompt_dir = prompt_dir
        self.prompt_file = prompt_file
        self.backend = backend_str  # language model backend for patient agent
        self.backend_api_type = backend_api_type  # language model backend for patient agent
        self.temperature = temperature
        self.random_seed = random_seed
        self.num_word_sample = num_word_sample

        # Load patient profile & setting bias
        self.patient_profile = patient_profile
        self.bias_prompt_dict = {
            "personality": json.load(open(os.path.join(self.prompt_dir, "personality_type.json"), "r")),
            "cefr_level": json.load(open(os.path.join(self.prompt_dir, "cefr_type.json"), "r")),
            "recall_level": json.load(open(os.path.join(self.prompt_dir, "recall_level_type.json"), "r")),
            "dazed_level": json.load(open(os.path.join(self.prompt_dir, "dazed_level_type.json"), "r")),
        }
        self.sentence_limit = json.load(open(os.path.join(self.prompt_dir, "sentence_length_limit.json"), "r"))

        patient_profile['cefr_option'] = patient_profile.pop('cefr')
        patient_profile['personality_option'] = patient_profile.pop('personality')
        patient_profile['recall_level_option'] = patient_profile.pop('recall_level')
        patient_profile['dazed_level_option'] = patient_profile.pop('dazed_level')
        
        self.cefr_type = patient_profile["cefr_option"]
        self.personality_type = patient_profile["personality_option"]
        self.recall_level_type = patient_profile["recall_level_option"]
        self.dazed_level_type = patient_profile["dazed_level_option"]

        # Set CEFR bias
        cefr_levels = ["A", "B", "C"]
        current_index = cefr_levels.index(self.cefr_type)
        higher_level = cefr_levels[current_index + 1] if self.cefr_type != "C" else None

        self.patient_profile["understand_words"] = ", ".join(self.patient_profile[f"cefr_{self.cefr_type}1"].split(", ")[: self.num_word_sample])
        self.patient_profile["misunderstand_words"] = ", ".join(self.patient_profile[f"cefr_{self.cefr_type}2"].split(", ")[: self.num_word_sample])
        self.patient_profile["understand_med_words"] = ", ".join(self.patient_profile[f"med_{self.cefr_type}"].split(", ")[: self.num_word_sample])
        self.patient_profile["misunderstand_med_words"] = (
            ", ".join(self.patient_profile[f"med_{higher_level}"].split(", ")[: self.num_word_sample]) if higher_level is not None else ""
        )
        self.patient_profile["cefr"] = "\n\t\t" + "\n\t\t\t".join(self.bias_prompt_dict["cefr_level"][self.cefr_type].split("\n\t")[1:]).format(**self.patient_profile)
        self.patient_profile["personality"] = "\n\t\t"  + "\n\t\t".join(self.bias_prompt_dict["personality"][self.personality_type].split("\n\t")[1:])
        self.patient_profile["personality"] += "\n\t\tIMPORTANT: Ensure that your personality is clearly represented throughout the conversation, while allowing your emotional tone and style to vary naturally across turns." if self.personality_type != "plain" else ""
        
        self.patient_profile["memory_recall_level"] = f"{self.recall_level_type.capitalize()}\n\t\t" + "\n\t\t".join(
            self.bias_prompt_dict["recall_level"][self.recall_level_type].split("\n\t")[1:]
        )

        dazed_levels = ["high", "moderate", "normal"]
        dazed_states = ["initial", "intermediate", "later"]

        dazed_index = dazed_levels.index(self.dazed_level_type)
        dazed_description = ""

        if self.dazed_level_type != "normal":
            dazed_description += (
                f"\n\tThe patient's initial dazed level is {self.dazed_level_type}. "
                "The dazedness should gradually fade throughout the conversation as the doctor continues to reassure them. "
                "Transitions should feel smooth and natural, rather than abrupt. "
                "While the change should be subtle and progressive, the overall dazed level is expected to decrease noticeably every 4-5 turns, following the instructions for each level below."
            )

            for _dazed_index in range(dazed_index, len(dazed_levels)):
                dazed_description += f"\n\t{dazed_levels[_dazed_index].capitalize()} Dazedness ({dazed_states[_dazed_index].capitalize()} Phase)\n\t\t" + "\n\t\t".join(
                    self.bias_prompt_dict["dazed_level"][dazed_levels[_dazed_index]].split("\n\t")[1:]
                )

            dazed_description += "\n\tNote: Dazedness reflects the patient's state of confusion and inability in following the conversation, independent of their language proficiency."
        else:
            dazed_description = f"{self.dazed_level_type.capitalize()}\n\t\t" + "\n\t\t".join(self.bias_prompt_dict["dazed_level"][self.dazed_level_type].split("\n\t")[1:])

        self.patient_profile["dazed_level"] = dazed_description
        self.patient_profile["reminder"] = (
            "You should act like "
            + self.bias_prompt_dict["cefr_level"][self.cefr_type].split("\n\t")[0]
            + " You are "
            + self.bias_prompt_dict["personality"][self.personality_type].split("\n\t")[0]
            + ". Also, you "
            + self.bias_prompt_dict["recall_level"][self.recall_level_type].split("\n\t")[0].lower()
        )
        self.patient_profile["reminder"] += " " + self.bias_prompt_dict["dazed_level"][self.dazed_level_type].split("\n\t")[0]
        self.patient_profile["sent_limit"] = self.sentence_limit[self.personality_type] if self.personality_type is not None else "3"

        # Load prompt text file
        prompt_file = self.prompt_file
        if self.patient_profile["diagnosis"] == "Urinary tract infection":
            prompt_file += "_uti"
        self.system_prompt_text = file_to_string(os.path.join(self.prompt_dir, prompt_file + ".txt"))

        # Set gt diagnosis labels
        self.diagnosis = patient_profile["diagnosis"]
        self.reset()

    def inference(self, question) -> str:
        answer = str()
        self.messages.append({"role": "user", "content": f"{question}"})
        client = get_response_method(self.backend_api_type)
        model = vllm_model_setup(self.backend) if self.backend_api_type == "vllm" else self.backend

        response = client(self.messages, model=model, temperature=self.temperature, seed=self.random_seed)
        answer = get_answer(response)

        self.messages.append({"role": "assistant", "content": f"{answer}"})
        return answer

    def set_system_prompt(self) -> None:
        self.system_prompt = self.system_prompt_text.format(**self.patient_profile)
        prompt_valid_check(self.system_prompt, self.patient_profile)

    def reset(self) -> None:
        self.set_system_prompt()
        system_message = {"role": "system", "content": self.system_prompt}
        self.messages = [system_message]


class DoctorAgent:
    def __init__(self, max_infs=15, top_k_diagnosis=5, backend_str="gpt4", backend_api_type="gpt_azure", temperature=0, random_seed=42, prompt_dir=None, prompt_file=None, patient_info=None) -> None:
        self.prompt_dir = prompt_dir
        self.prompt_file = prompt_file
        self.infs = 0  # number of inference calls to the doctor
        self.max_infs = max_infs  # maximum number of inference calls to the doctor
        self.top_k_diagnosis = top_k_diagnosis
        self.backend = backend_str  # language model backend for doctor agents
        self.backend_api_type = backend_api_type
        self.temperature = temperature
        self.random_seed = random_seed
        self.patient_info = patient_info if patient_info is not None else {}

        # Load prompt text file
        self.system_prompt_text = file_to_string(os.path.join(self.prompt_dir, self.prompt_file + ".txt"))

        # prepare initial conditions for LLM
        self.doctor_greet = "Hello, how can I help you?"
        self.reset()

    def inference(self, question) -> str:
        answer = str()
        if self.infs >= self.max_infs:
            return "Maximum inferences reached"
        self.infs += 1
        self.messages[0]["content"] = self.system_prompt()  # update current turns
        self.messages.append({"role": "user", "content": f"{question}"})
        client = get_response_method(self.backend_api_type)
        model = vllm_model_setup(self.backend) if self.backend_api_type == "vllm" else self.backend

        response = client(self.messages, model=model, temperature=self.temperature, seed=self.random_seed)
        answer = get_answer(response)

        self.messages.append({"role": "assistant", "content": f"{answer}"})
        return answer

    def system_prompt(self) -> str:
        system_prompt = self.system_prompt_text.format(
            total_idx=self.max_infs, curr_idx=self.infs, remain_idx=self.max_infs - self.infs, top_k_diagnosis=self.top_k_diagnosis, **self.patient_info
        )
        prompt_valid_check(system_prompt, self.patient_info)
        return system_prompt

    def reset(self) -> None:
        system_message = {"role": "system", "content": self.system_prompt()}
        self.messages = [system_message]


def main(args):
    scenario_loader = ScenarioLoaderMIMICIV(args.data_dir, args.data_file_name)
    log_and_print(f"Load Datasets from {args.data_dir}, size: {scenario_loader.num_scenarios}")
    log_and_print(f"""Patient prompt template:\n\t{file_to_string(os.path.join(args.prompt_dir, args.patient_prompt_file + ".txt"))}""")
    log_and_print(f"""Doctor prompt template:\n\t{file_to_string(os.path.join(args.prompt_dir, args.doctor_prompt_file + ".txt"))}""")

    # Pipeline for huggingface models
    num_scenarios = min(args.num_scenarios, scenario_loader.num_scenarios) if args.num_scenarios is not None else scenario_loader.num_scenarios
    for _scenario_id in range(0, num_scenarios):
        # Initialize scenarios
        scenario = scenario_loader.get_scenario(id=_scenario_id)

        # Initialize agents
        patient_agent = PatientAgent(
            patient_profile=scenario,
            backend_str=args.patient_llm,
            backend_api_type=args.patient_api_type,
            temperature=args.patient_temperature,
            random_seed=args.random_seed,
            prompt_dir=args.prompt_dir,
            prompt_file=args.patient_prompt_file,
            num_word_sample=args.num_word_sample,
        )
        doctor_agent = DoctorAgent(
            max_infs=args.total_inferences,
            top_k_diagnosis=args.top_k_diagnosis,
            backend_str=args.doctor_llm,
            backend_api_type=args.doctor_api_type,
            temperature=args.doc_temperature,
            random_seed=args.random_seed,
            prompt_dir=args.prompt_dir,
            prompt_file=args.doctor_prompt_file,
            patient_info=scenario,
        )

        log_and_print(f"""Patient prompt:\n\t{patient_agent.system_prompt}""")
        log_and_print(f"""Doctor prompt:\n\t{doctor_agent.system_prompt()}""")

        # Start dialogue
        dialog_history = [{"role": "Doctor", "content": doctor_agent.doctor_greet}]
        doctor_agent.messages.append({"role": "assistant", "content": f"{doctor_agent.doctor_greet}"})
        log_and_print(f"Doctor: {doctor_agent.doctor_greet}")

        for inf_idx in range(args.total_inferences):
            # # Obtain response from patient
            patient_response = patient_agent.inference(dialog_history[-1]["content"])

            dialog_history.append({"role": "Patient", "content": patient_response})
            log_and_print("Patient [{}%]: {}".format(int(((inf_idx + 1) / args.total_inferences) * 100), patient_response))

            # Obtain doctor dialogue
            if inf_idx == args.total_inferences - 1:
                doctor_response = doctor_agent.inference(dialog_history[-1]["content"] + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis.")
            else:
                doctor_response = doctor_agent.inference(dialog_history[-1]["content"])
            dialog_history.append({"role": "Doctor", "content": doctor_response})
            log_and_print("Doctor [{}%]: {}".format(int(((inf_idx + 1) / args.total_inferences) * 100), doctor_response))

            end_flag = detect_termination(doctor_response)
            if end_flag:
                break

            # Prevent API timeouts
            time.sleep(1.0)

        dialog_info = {
            "hadm_id": scenario["hadm_id"],
            "doctor_engine_name": doctor_agent.backend,
            "patient_engine_name": patient_agent.backend,
            "cefr_type": patient_agent.patient_profile["cefr_option"],
            "personality_type": patient_agent.patient_profile["personality_option"],
            "recall_level_type": patient_agent.patient_profile["recall_level_option"],
            "dazed_level_type":patient_agent.patient_profile["dazed_level_option"],
            "diagnosis": patient_agent.diagnosis,
            "dialog_history": dialog_history,
        }
        save_to_dialogue(dialog_info, os.path.join(args.save_dir, "dialogue.jsonl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Agent setting
    parser.add_argument("--doctor_api_type", type=str, default="azure", choices=["gpt_azure", "vllm", "genai"])
    parser.add_argument("--patient_api_type", type=str, default="azure", choices=["gpt_azure", "vllm", "genai"])
    parser.add_argument(
        "--doctor_llm",
        type=str,
        default="gpt-4o-mini",
        choices=[
            "gpt-4o-mini",
            "gpt-5-nano",
            "gemini-2.5-flash",
            "vllm-llama3.1-8b-instruct",
            "vllm-llama3.1-70b-instruct",
            "vllm-llama3.3-70b-instruct",
            "vllm-qwen2.5-72b-instruct",
            "vllm-qwen2.5-7b-instruct",
            "vllm-deepseek-llama-70b",
        ],
    )
    parser.add_argument(
        "--patient_llm",
        type=str,
        default="gpt-4o-mini",
        choices=[
            "gpt-4o-mini",
            "gpt-5-nano",
            "gemini-2.5-flash",
            "vllm-llama3.1-8b-instruct",
            "vllm-llama3.1-70b-instruct",
            "vllm-llama3.3-70b-instruct",
            "vllm-qwen2.5-72b-instruct",
            "vllm-qwen2.5-7b-instruct",
            "vllm-deepseek-llama-70b",
        ],
    )
    parser.add_argument("--patient_temperature", type=float, default=0.7)
    parser.add_argument("--doc_temperature", type=float, default=0.7)
    parser.add_argument("--random_seed", type=int, default=42)

    # Path define
    parser.add_argument("--agent_dataset", type=str, default="mimiciv")
    parser.add_argument("--data_dir", type=str, default="./data/final_data")
    parser.add_argument("--prompt_dir", type=str, default="./prompts/simulation")
    parser.add_argument("--save_dir", type=str, default="./results", help="save dir")
    parser.add_argument("--exp_name", type=str, default="", help="exp_name")
    parser.add_argument("--num_scenarios", type=int, default=None, help="Number of scenarios to simulate")

    # Data file names
    parser.add_argument("--data_file_name", type=str, default="patient_profile")
    parser.add_argument("--patient_prompt_file", type=str, default="initial_system_patient_w_persona")
    parser.add_argument("--doctor_prompt_file", type=str, default="initial_system_doctor")

    # Agent options
    parser.add_argument("--total_inferences", type=int, default=30, required=False, help="Number of inferences between patient and doctor")
    parser.add_argument("--top_k_diagnosis", type=int, default=5, required=False, help="Number of diagnosis of doctor")
    parser.add_argument("--num_word_sample", type=int, default=10, required=False, help="Number of scenarios to simulate")

    args = parser.parse_args()
    set_seed(args.random_seed)
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, now if len(args.exp_name) == 0 else args.exp_name + "_" + now)
    os.makedirs(args.save_dir, exist_ok=False)
    log_file_path = os.path.join(args.save_dir, "dialog_history.log")
    logging.basicConfig(filename=log_file_path, filemode="a", format="%(message)s", level=logging.INFO)  # Append mode

    main(args)
