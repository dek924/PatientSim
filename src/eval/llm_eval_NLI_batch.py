import os
import re
import ast
import sys
import nltk
import json
import argparse
import numpy as np

nltk.download("punkt_tab")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize
from models import get_response_method, vllm_model_setup
from utils import load_json, load_jsonl, save_to_json, get_profile, set_seed, process_string
from prompts.eval.prompts import PATIENT_PROFILE_TEMPLATE, PATIENT_PROFILE_TEMPLATE_UTI


KEY_DESCRIPTION = {
    "age": "Age: {age}",
    "gender": "Gender: {gender}",
    "race": "Race: {race}",
    "tobacco": "Tobacco: {tobacco}",
    "alcohol": "Alcohol: {alcohol}",
    "illicit_drug": "Illicit drug use: {illicit_drug}",
    "sexual_history": "Sexual History: {sexual_history}",
    "exercise": "Exercise: {exercise}",
    "marital_status": "Marital status: {marital_status}",
    "children": "Children: {children}",
    "living_situation": "Living Situation: {living_situation}",
    "occupation": "Occupation: {occupation}",
    "insurance": "Insurance: {insurance}",
    "allergies": "Allergies: {allergies}",
    "family_medical_history": "Family medical history: {family_medical_history}",
    "medical_device": "Medical devices previously used or currently in use before this ED admission: {medical_device}",
    "medical_history": "Medical history prior to this ED admission: {medical_history}",
    "present_illness": "Present illness:\n\tpositive: {present_illness_positive}\n\tnegative (denied): {present_illness_negative}",
    "chief_complaint": "ED chief complaint: {chiefcomplaint}",
    "pain": "Pain level at ED Admission (0 = no pain, 10 = worst pain imaginable): {pain}",
    "medication": "Current medications they are taking: {medication}",
    "arrival_transport": "ED Arrival Transport: {arrival_transport}",
    "diagnosis": "ED Diagnosis: {diagnosis}",
}


def process_answer(response, expected_type="dict"):
    if hasattr(response, "choices"):
        output = response.choices[0].message.content.strip()
    elif hasattr(response, "text"):
        output = response.text.strip()
    else:
        raise NotImplementedError(f"Fail to extract answer: {output}")
        
    output = re.sub(r'```json\s*([\s\S]*?)\s*```', r'\1', output)
    output = re.sub(r'```\s*([\s\S]*?)\s*```', r'\1', output)

    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict) and expected_type == "dict":
            return parsed
        elif isinstance(parsed, list) and expected_type == "list":
            return parsed
        else:
            raise ValueError(f"Expected {expected_type}, got {type(parsed)}")
    except json.JSONDecodeError:
        pass

    if expected_type == "list":
        match = re.search(r'(\[[\s\S]*\])', output)
    else:  
        match = re.search(r'({[\s\S]*})', output)

    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict) and expected_type == "dict":
                return parsed
            elif isinstance(parsed, list) and expected_type == "list":
                return parsed
            else:
                raise ValueError(f"Expected {expected_type}, got {type(parsed)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse {expected_type}: {match.group(1)[:100]}... (Error: {e})")

    if expected_type == "list":
        pattern = r'\[[\s\S]*?\]'
    else:
        pattern = r'{[\s\S]*?}'
    
    matches = sorted(re.findall(pattern, output, re.DOTALL), key=len, reverse=True)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and expected_type == "dict":
                return parsed
            elif isinstance(parsed, list) and expected_type == "list":
                return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(match)
                if isinstance(parsed, dict) and expected_type == "dict":
                    return parsed
                elif isinstance(parsed, list) and expected_type == "list":
                    return parsed
            except (SyntaxError, ValueError):
                continue

    raise ValueError(f"No valid {expected_type} found in: {output[:100]}...")


def get_valid_answer_with_retries(client, messages, model, temperature, max_retries=10, random_seed=None, expected_type="dict"):
    try:
        response = client(messages, model=model, temperature=temperature, seed=random_seed)
        answer = process_answer(response, expected_type)
        return answer, response
    except Exception as e:
        print(f"Initial attempt failed: {e}")
        
        for attempt in range(max_retries):
            try:
                response = client(messages, model=model, temperature=temperature, seed=None)
                answer = process_answer(response, expected_type)
                return answer, response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
                
        print(f"All {max_retries} attempts failed.")
        return None, None


def process_batch(batch_data, args, scenario_dict, batch_idx, temp_dir):
    batch_results = {} 
    client = get_response_method(args.moderator_api_type)
    model = vllm_model_setup(args.moderator) if "vllm" in args.moderator else args.moderator
    batch_save_path = os.path.join(temp_dir, f"batch_{batch_idx}.json")
    if os.path.exists(batch_save_path):
        batch_results = load_json(batch_save_path)
        
    for idx, data in enumerate(batch_data):
        scenario = data["hadm_id"]
        dialogue = data["dialog_history"]

        if scenario not in batch_results:
            # Get patient profile
            profile = get_profile(scenario_dict, str(scenario))
            if profile:
                profile["medical_history"] = "\n\t" + profile["medical_history"].replace("; ", "\n\t")
                if profile["diagnosis"] == "Urinary tract infection":
                    profile_information = PATIENT_PROFILE_TEMPLATE_UTI.format(**profile)
                else:
                    profile_information = PATIENT_PROFILE_TEMPLATE.format(**profile)

                nli_phase0_prompt = load_json(os.path.join(args.prompt_dir, "eval_nli_step0.json"))
                nli_phase1_prompt = load_json(os.path.join(args.prompt_dir, "eval_nli_step1.json"))
                nli_phase1_hallucination_prompt = load_json(os.path.join(args.prompt_dir, "eval_nli_step1_hallucination.json"))
                nli_phase2_case_cls = load_json(os.path.join(args.prompt_dir, "eval_nli_step2_cls.json"))
                nli_phase2_case_rate = load_json(os.path.join(args.prompt_dir, "eval_nli_step2_rate.json"))

                conversation = ""
                utterance_results = {}

                for utter in tqdm(dialogue, desc=f"Processing scenario {scenario}"):
                    if utter["role"] == "Patient":
                        utterance_results[utter["content"]] = {}
                        sentences = sent_tokenize(process_string(utter["content"]))
                        for i, sent in enumerate(sentences):
                            utterance_results[utter["content"]][sent] = {}
                            conversation += f"""\t{utter["role"]}: """ if i == 0 else ""

                            # Step 0: Information state or not
                            step0_messages = deepcopy(nli_phase0_prompt)
                            step0_messages[-1]["content"] = json.dumps({"dialogue_history": conversation, "current_utterance": sent})
                            step0_answer, step0_response = get_valid_answer_with_retries(
                                client, step0_messages, model=model, temperature=args.temperature, random_seed=args.random_seed, expected_type="dict"
                            )
                            utterance_results[utter["content"]][sent]["step0"] = step0_answer
                            conversation += sent + " "

                            if step0_answer["prediction"].lower() == "information":
                                # Step 1: Exclusively mentioned in the profile or not
                                step1_messages = deepcopy(nli_phase1_prompt)
                                step1_messages[-1]["content"] = json.dumps(
                                    {
                                        "profile": profile_information,
                                        "dialogue_history": conversation,
                                        "current_utterance": sent,
                                    }
                                )

                                step1_answer, step1_response = get_valid_answer_with_retries(
                                    client, step1_messages, model=model, temperature=args.temperature, random_seed=args.random_seed, expected_type="list"
                                )
                                utterance_results[utter["content"]][sent]["step1-1"] = step1_answer

                                # Step 1-1: Any information which not explicitly mentioned in profile
                                step1_hallucination_messages = deepcopy(nli_phase1_hallucination_prompt)
                                step1_hallucination_messages[-1]["content"] = json.dumps({"profile": profile_information, "dialogue_history": conversation, "current_utterance": sent})
                                step1_2_answer, step1_2_response = get_valid_answer_with_retries(
                                    client, step1_hallucination_messages, model=model, temperature=args.temperature, random_seed=args.random_seed, expected_type="dict"
                                )
                                utterance_results[utter["content"]][sent]["step1-2"] = step1_2_answer

                                related_categories = list({result_dict["category"] for result_dict in step1_answer if int(result_dict["prediction"]) == 1})
                                hallucination_flag = int(step1_2_answer["prediction"]) == 1

                                if len(related_categories) > 0:
                                    # Step 2-2: if patient's utter explicitly mentioned in profile, classify entail / contradict
                                    profile_list = [KEY_DESCRIPTION[related_cat].format(**profile) for related_cat in related_categories]
                                    step2_2_messages = deepcopy(nli_phase2_case_cls)
                                    step2_2_messages[-1]["content"] = json.dumps({"profile": profile_list, "dialogue_history": conversation, "current_utterance": sent})
                                    step2_2_answer, step2_2_response = get_valid_answer_with_retries(
                                        client, step2_2_messages, model=model, temperature=args.temperature, random_seed=args.random_seed, expected_type="list"
                                    )
                                    utterance_results[utter["content"]][sent]["step2-2"] = step2_2_answer
                                    related_info = [subdict["profile"] for subdict in step2_2_answer if subdict["entailment_prediction"] != 0]
                                else:
                                    related_info = []

                                if hallucination_flag or (len(related_info) == 0):
                                    # Step 2-1: if patient's utter not explicitly mentioned in profile
                                    step2_1_messages = deepcopy(nli_phase2_case_rate)
                                    step2_1_messages[-1]["content"] = json.dumps({"profile": profile_information, "dialogue_history": conversation, "current_utterance": sent})
                                    step2_1_answer, step2_1_response = get_valid_answer_with_retries(
                                        client, step2_1_messages, model=model, temperature=args.temperature, random_seed=args.random_seed, expected_type="dict"
                                    )
                                    utterance_results[utter["content"]][sent]["step2-1"] = step2_1_answer

                        conversation += "\n"
                    else:
                        conversation += f"""\t{utter["role"]}: {utter["content"]}\n"""

                batch_results[scenario] = utterance_results
                save_to_json(batch_results, batch_save_path)
            else:
                scenario_path = os.path.join(args.data_dir, f"{args.data_file_name}.json")
                print(f"Scenario {scenario} not found in the scenario {scenario_path}.")

    save_to_json(batch_results, batch_save_path)
    return batch_save_path


def merge_batch_results(temp_dir, save_path, existing_results=None):
    total_nli_result = existing_results if existing_results is not None else {}

    for batch_file in sorted(os.listdir(temp_dir)):
        if batch_file.startswith("batch_") and batch_file.endswith(".json"):
            batch_path = os.path.join(temp_dir, batch_file)
            with open(batch_path, 'r') as f:
                batch_results = json.load(f)
            total_nli_result.update(batch_results)

    save_to_json(total_nli_result, save_path)


def process_batch_wrapper(args_tuple):
    return process_batch(*args_tuple)


def main(args):
    # Set evaluate path & setting
    result_path = os.path.join(args.result_dir, args.trg_exp_name)
    temp_dir = os.path.join(result_path, "temp_batches") 
    os.makedirs(temp_dir, exist_ok=True)

    # Setup the moderator
    print(f"{args.moderator_api_type} api call")
    client = get_response_method(args.moderator_api_type)
    model = vllm_model_setup(args.moderator) if "vllm" in args.moderator else args.moderator

    # Load test data
    scenario_dict = load_json(os.path.join(args.data_dir, f"{args.data_file_name}.json"))
    dialogue_hists = load_jsonl(os.path.join(result_path, "dialogue.jsonl"))[:1]

    # Evaluate only the information set
    if args.eval_target == "info":
        scenario_dict = [subdict for subdict in scenario_dict if subdict["split"] == "info"]

    # Eval NLI task
    save_path = os.path.join(result_path, f"{args.moderator}_nli.json")
    if os.path.isfile(save_path):
        total_nli_result = load_json(save_path)
    else:
        total_nli_result = {}

    # Batch setting
    batch_size = args.batch_size
    dialogue_hists_batches = [dialogue_hists[i:i + batch_size] for i in range(0, len(dialogue_hists), batch_size)]
    print(len(dialogue_hists_batches))
    batch_save_paths = []
    with Pool(processes=len(dialogue_hists_batches)) as pool:  
        batch_args = [
            (batch_data, args, scenario_dict, batch_idx, temp_dir)
            for batch_idx, batch_data in enumerate(dialogue_hists_batches)
            if not all(str(data["hadm_id"]) in total_nli_result for data in batch_data)
        ]
        batch_save_paths = pool.map(process_batch_wrapper, batch_args)

    merge_batch_results(temp_dir, save_path, total_nli_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Diagnosis Simulation CLI")
    parser.add_argument(
        "--moderator",
        type=str,
        default="vllm-llama3.1-70b-instruct",
        choices=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-5-nano",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "vllm-deepseek-llama-70b",
            "vllm-llama3.1-70b-instruct",
            "vllm-llama3.3-70b-instruct",
            "vllm-qwen2.5-72b-instruct",
        ],
    )
    parser.add_argument("--moderator_api_type", type=str, default="vllm", choices=["gpt_azure", "vllm", "genai"])
    parser.add_argument("--data_dir", type=str, default="./data/final_data")
    parser.add_argument("--data_file_name", type=str, default="patient_profile")
    parser.add_argument("--eval_target", type=str, default="info", choices=["info", "all"])
    parser.add_argument("--prompt_dir", type=str, default="./prompts/eval/NLI")
    parser.add_argument("--result_dir", type=str, default="./results", help="save dir")
    parser.add_argument("--trg_exp_name", type=str, default=None, help="save dir")
    parser.add_argument("--batch_size", type=int, default=10, required=False, help="batch size for nli")
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.random_seed)
    main(args)
