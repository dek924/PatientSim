import os
import sys
import time
import json
import hydra
import random
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hydra.core.hydra_config import HydraConfig
from agent.doctor_agent import DoctorAgent
from agent.patient_agent import PatientAgent
from utils import file_to_string, save_to_dialogue, set_seed, detect_termination


class ScenarioLoaderMIMICIV:
    def __init__(self, data_dir, data_name="sample_info") -> None:
        with open(os.path.join(data_dir, f"{data_name}.json"), "r") as f:
            self.scenario_dict = json.load(f)
        self.num_scenarios = len(self.scenario_dict)
        logging.info(f"Load {self.num_scenarios} scenarios from {data_dir}")

    def sample_scenario(self):
        return self.scenario_dict[random.randint(0, self.num_scenarios - 1)]

    def get_scenario(self, id):
        if id is None:
            return self.sample_scenario()
        return self.scenario_dict[id]


@hydra.main(config_path="./config", config_name="base", version_base="1.3")
def main(cfg):
    # Set random seed & create save directory
    set_seed(cfg.experiment.random_seed)
    cfg.save_dir = os.path.join(HydraConfig.get().run.dir, cfg.save_dir)
    os.makedirs(cfg.save_dir, exist_ok=True)
    if cfg.experiment.verbose:
        print(f"Save directory: {cfg.save_dir}")

    # Load scenarios
    scenario_loader = ScenarioLoaderMIMICIV(cfg.data_dir, cfg.data.data_file_name)
    logging.info(f"Load Datasets from {cfg.data_dir}, size: {scenario_loader.num_scenarios}")
    logging.info(f"""Patient prompt template:\n\t{file_to_string(os.path.join(cfg.prompt_dir, cfg.data.patient_prompt_file + ".txt"))}""")
    logging.info(f"""Doctor prompt template:\n\t{file_to_string(os.path.join(cfg.prompt_dir, cfg.data.doctor_prompt_file + ".txt"))}""")

    # Pipeline for huggingface models
    num_scenarios = min(cfg.data.num_scenarios, scenario_loader.num_scenarios) if cfg.data.num_scenarios is not None else scenario_loader.num_scenarios
    for _scenario_id in range(0, num_scenarios):
        # Initialize scenarios
        scenario = scenario_loader.get_scenario(id=_scenario_id)

        # Initialize agents
        patient_agent = PatientAgent(
            patient_profile=scenario,
            backend_str=cfg.patient_agent.backend,
            backend_api_type=cfg.patient_agent.api_type,
            prompt_dir=cfg.prompt_dir,
            prompt_file=cfg.data.patient_prompt_file,
            num_word_sample=cfg.data.num_word_sample,
            client_params=cfg.patient_agent.params,
            verbose=cfg.experiment.verbose,
        )
        doctor_agent = DoctorAgent(
            max_infs=cfg.doctor_agent.max_infs,
            top_k_diagnosis=cfg.doctor_agent.top_k_diagnosis,
            backend_str=cfg.doctor_agent.backend,
            backend_api_type=cfg.doctor_agent.api_type,
            prompt_dir=cfg.prompt_dir,
            prompt_file=cfg.data.doctor_prompt_file,
            patient_info=scenario,
            client_params=cfg.doctor_agent.params,
            verbose=cfg.experiment.verbose,
        )

        # Start dialogue
        start_time = time.time()
        dialog_history = [{"role": "Doctor", "content": doctor_agent.doctor_greet}]
        doctor_agent.messages.append({"role": "assistant", "content": f"{doctor_agent.doctor_greet}"})
        logging.info(f"Doctor: {doctor_agent.doctor_greet}")

        for inf_idx in range(cfg.experiment.total_inferences):
            # # Obtain response from patient
            patient_response = patient_agent.inference(dialog_history[-1]["content"])

            dialog_history.append({"role": "Patient", "content": patient_response})
            logging.info("Patient [{}%]: {}".format(int(((inf_idx + 1) / cfg.experiment.total_inferences) * 100), patient_response))

            # Obtain doctor dialogue
            if inf_idx == cfg.experiment.total_inferences - 1:
                doctor_response = doctor_agent.inference(dialog_history[-1]["content"] + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis.")
            else:
                doctor_response = doctor_agent.inference(dialog_history[-1]["content"])
            dialog_history.append({"role": "Doctor", "content": doctor_response})
            logging.info("Doctor [{}%]: {}".format(int(((inf_idx + 1) / cfg.experiment.total_inferences) * 100), doctor_response))

            end_flag = detect_termination(doctor_response)
            if end_flag:
                break

            # Prevent API timeouts
            time.sleep(1.0)

        end_time = time.time()
        dialog_info = {
            "hadm_id": scenario["hadm_id"],
            "doctor_engine_name": doctor_agent.backend,
            "patient_engine_name": patient_agent.backend,
            "doctor_api_type": doctor_agent.backend_api_type,
            "patient_api_type": patient_agent.backend_api_type,
            "cefr_type": patient_agent.patient_profile["cefr_option"],
            "personality_type": patient_agent.patient_profile["personality_option"],
            "recall_level_type": patient_agent.patient_profile["recall_level_option"],
            "dazed_level_type":patient_agent.patient_profile["dazed_level_option"],
            "diagnosis": patient_agent.diagnosis,
            "dialog_history": dialog_history,
            "patient_token_log": patient_agent.token_log,
            "doctor_token_log": doctor_agent.token_log,
            "elapsed_time": end_time - start_time,
        }
        save_to_dialogue(dialog_info, os.path.join(cfg.save_dir, "dialogue.jsonl"))


if __name__ == "__main__":
    main()
