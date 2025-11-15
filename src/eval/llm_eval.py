import os
import re
import sys
import ast
import json
import copy
import torch
import random
import argparse
import numpy as np
import pandas as pd

from torch import nn
from transformers import AutoTokenizer, AutoModel

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from copy import deepcopy
from models import get_response_method, vllm_model_setup, get_answer
from utils import load_json, load_jsonl, save_to_json, get_profile, file_to_string, set_seed, detect_termination, process_string
from prompts.eval.prompts import ABS_SYSTEM_PROMPT, SCORE_RUBRIC_TEMPLATE, PATIENT_PROFILE_TEMPLATE, PATIENT_PROFILE_TEMPLATE_UTI, PATIENT_PERSONA_TEMPLATE



def process_answer(response, expected_type="dict"):
    if hasattr(response, "choices"):
        output = response.choices[0].message.content
    elif hasattr(response, "text"):
        output = response.text.strip()
    else:
        raise NotImplementedError(f"Fail to extract answer: {output}")

    if expected_type == "dict":
        pattern = r"\{.*\}"
    elif expected_type == "list":
        pattern = r"\[.*\]"

    answer = re.search(pattern, output, re.DOTALL).group()
    try:
        answer = ast.literal_eval(answer)
    except:
        answer = answer

    answer = json.dumps(answer)
    answer = json.loads(answer)

    return answer


def get_valid_answer_with_retries(client, messages, model, temperature, max_retries=10, random_seed=None, expected_type="dict"):
    try:
        response = client(messages, model=model, temperature=temperature, seed=random_seed)
        answer = process_answer(response, expected_type)
        return answer, response
    except:
        for attempt in range(max_retries):
            try:
                response = client(messages, model=model, temperature=temperature, seed=None)
                answer = process_answer(response, expected_type)
                return answer, response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
        print(f"All {max_retries} attempts failed.")
        return None


def get_embedding(tokenizer, model, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def compute_similarity(tokenizer, model, text1, text2, metric):
    emb1 = get_embedding(tokenizer, model, text1)
    emb2 = get_embedding(tokenizer, model, text2)
    return metric(emb1, emb2)[0]


def flatten_dict_simple(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        if parent_key == "present_illness":
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
        else:
            new_key = k

        if isinstance(v, dict):
            items.extend(flatten_dict_simple(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main(args):
    # Set evaluate path & setting
    result_path = os.path.join(args.result_dir, args.trg_exp_name)

    # Setup the moderator
    client = get_response_method(args.moderator_api_type)
    model = vllm_model_setup(args.moderator) if "vllm" in args.moderator else args.moderator

    # Load test data
    scenario_dict = load_json(os.path.join(args.data_dir, f"{args.data_file_name}.json"))
    dialogue_hists = load_jsonl(os.path.join(result_path, "outputs", "dialogue.jsonl"))

    # Eval DDX task
    if args.eval_ddx:
        # Load prompt
        user_prompt_template = file_to_string(os.path.join(args.prompt_dir, "eval_ddx_prompt.txt"))

        # Set save path & save variables
        correct_cnt = 0
        total_ddx_result = {}
        save_path = os.path.join(result_path, f"{args.moderator}_ddx_{args.trg_agent}.json")
        assert not os.path.isfile(save_path)

        # Start evaluation
        for data in tqdm(dialogue_hists):
            # Load data per scenario
            scenario = data["hadm_id"]
            dialogue = data["dialog_history"]
            gt_diagnosis = data["diagnosis"]
            if gt_diagnosis == "Urinary tract infection":
                gt_diagnosis = "Urinary tract infection group (including UTI, pyelonephritis and cystitis)"

            doctor_prediction = None
            for utter in dialogue:
                if detect_termination(utter["content"].lower()):
                    doctor_prediction = utter["content"].lower().split("[ddx]", 1)[-1].strip()

            if doctor_prediction is None:
                doctor_prediction = dialogue[-1]["content"].lower()

            # Set up prompt & get llm response
            user_prompt = copy.deepcopy(user_prompt_template).format(ddx=doctor_prediction, ans=gt_diagnosis)
            # print(user_prompt)
            messages = [{"role": "user", "content": user_prompt}]

            response = client(messages, model=model, temperature=args.temperature, seed=args.random_seed)
            answer = get_answer(response)

            # Save the result
            total_ddx_result[scenario] = {}
            total_ddx_result[scenario]["gt"] = gt_diagnosis
            total_ddx_result[scenario]["pred"] = doctor_prediction
            total_ddx_result[scenario]["answer"] = answer
            if answer.lower() == "y":
                correct_cnt += 1

        # Logging & save
        print(f"Prediction Acc: {(correct_cnt / len(dialogue_hists)) * 100:.2f}%")
        save_to_json(total_ddx_result, save_path)


    if args.eval_persona_quality:
        # Load prompt
        user_prompt_template = file_to_string(os.path.join(args.prompt_dir, "eval_dialogue_user.txt"))
        user_prompt_template_w_persona = file_to_string(os.path.join(args.prompt_dir, "eval_dialogue_user_w_persona.txt"))
        user_prompt_template_w_profile = file_to_string(os.path.join(args.prompt_dir, "eval_dialogue_user_w_profile.txt"))
        eval_criteria_dict = load_json(os.path.join(args.prompt_dir, "llm_eval_metrics_persona.json"))

        # Set save path & save variables
        total_persona_eval_result = {k: {} for k in eval_criteria_dict.keys()}
        save_path = os.path.join(result_path, f"{args.moderator}_persona_quality_{args.trg_agent}.json")
        assert not os.path.isfile(save_path)
        for data in tqdm(dialogue_hists):
            # Load data per scenario
            scenario = data["hadm_id"]
            dialogue = data["dialog_history"]
            dazed_level = data["dazed_level_type"]

            CEFR_DICT = {
                "A": "Beginner. Can make simple sentences.",
                "B": "Intermediate. Can have daily conversations.",
                "C": "Advanced. Can freely use even advanced medical terms.",
            }

            PERSONALITY_DICT = {
                "plain": "Neutral. No strong emotions or noticeable behavior.",
                "verbose": "Talkative Speaks a lot, and provides highly detailed responses.",
                "distrust": "Distrustful. Questions the doctor’s expertise and care.",
                "pleasing": "Pleasing. Overly positive and tend to minimize their problems.",
                "impatient": "Impatient. Easily irritated and lacks patience.",
                "overanxious": "Overanxious. Expresses concern beyond what is typical.",
            }

            RECALL_DICT = {"low": "Low. Often forgetting even major medical history.", "high": "High. Usually recalls their medical history accurately."}
            DAZED_DICT = {
                "normal": "Clear mental status, with naturally reflect their own persona.",
                "high": "Initially highly dazed and confused, struggles with conversation. With the doctor's reassurance, their dazedness subsides to a normal state.",
            }

            persona_prompt = {
                "Personality": PERSONALITY_DICT[data["personality_type"]],
                "CEFR": CEFR_DICT[data["cefr_type"]],
                "Recall_level": RECALL_DICT[data["recall_level_type"]], 
                "Dazed_level": DAZED_DICT[data["dazed_level_type"]]
            }
            profile = get_profile(scenario_dict, scenario)

            conversation = ""
            for utter in dialogue[:-1]:
                conversation += f"""\t{utter["role"]}: {process_string(utter["content"])}\n"""
            conversation += f"""\t{dialogue[-1]["role"]}: {process_string(dialogue[-1]["content"].split(".")[0])}.\n"""

            # Set up prompt & get llm response
            for eval_target, descriptions in eval_criteria_dict.items():
                score_rubric = SCORE_RUBRIC_TEMPLATE.format(**descriptions)
                if dazed_level == "normal":
                    if eval_target in ["Dazed_level"]:
                        continue
                else:
                    if eval_target in ["Personality", "CEFR", "Recall_level"]:
                        continue

                if eval_target in ["Personality", "CEFR", "Recall_level", "Dazed_level"]:
                    user_prompt = user_prompt_template_w_persona.replace("###Patient Persona", f"###Patient's {eval_target}")
                    user_prompt = user_prompt.format(conversation=conversation, rubric=score_rubric, profile=persona_prompt[eval_target])
                elif eval_target in ["Realism_w_Profile"]:
                    persona_info = PATIENT_PERSONA_TEMPLATE.format(personality=persona_prompt["Personality"], cefr=persona_prompt["CEFR"], memory_recall_level=persona_prompt["Recall_level"], dazed_level=persona_prompt["Dazed_level"])
                    user_prompt = user_prompt_template_w_persona.format(conversation=conversation, rubric=score_rubric, profile=persona_info)
                elif eval_target in ["Overall"]:
                    if profile["diagnosis"] == "Urinary tract infection":
                        profile_information = PATIENT_PROFILE_TEMPLATE_UTI.format(**profile)
                    else:
                        profile_information = PATIENT_PROFILE_TEMPLATE.format(**profile)
                    user_prompt = user_prompt_template_w_profile.format(conversation=conversation, rubric=score_rubric, profile=profile_information)
                else:
                    raise NotImplementedError

                user_content = ABS_SYSTEM_PROMPT + "\n\n" + user_prompt
                messages = [{"role": "user", "content": user_content}]
                response = client(messages, model=model, temperature=args.temperature, seed=args.random_seed)
                answer = get_answer(response)
                retry_cnt = 0
                max_retry = 10
                while "[RESULT]:" not in answer:
                    if max_retry < retry_cnt:
                        answer = None
                        break
                    response = client(messages, model=model, temperature=args.temperature, seed=None)
                    answer = get_answer(response)
                total_persona_eval_result[eval_target][scenario] = answer

            # Logging & save
            save_to_json(total_persona_eval_result, save_path)


    if args.eval_doc_quality:
        # Load prompt
        user_prompt_template = file_to_string(os.path.join(args.prompt_dir, "eval_dialogue_user.txt"))
        eval_criteria_dict = load_json(os.path.join(args.prompt_dir, "llm_eval_metrics_doc.json"))

        # Set save path & save variables
        total_doc_eval_result = {k: {} for k in eval_criteria_dict.keys()}
        save_path = os.path.join(result_path, f"{args.moderator}_doc_quality_{args.trg_agent}.json")
        assert not os.path.isfile(save_path)

        # Start evaluation
        for data in tqdm(dialogue_hists):
            # Load data per scenario
            scenario = data["hadm_id"]
            dialogue = data["dialog_history"]
            profile = get_profile(scenario_dict, scenario)

            conversation = ""
            for utter in dialogue[:-1]:
                conversation += f"""\t{utter["role"]}: {process_string(utter["content"])}\n"""
            conversation += f"""\t{dialogue[-1]["role"]}: {process_string(dialogue[-1]["content"].split(".")[0])}.\n"""

            # Set up prompt & get llm response
            for i, (eval_target, descriptions) in enumerate(eval_criteria_dict.items()):
                score_rubric = SCORE_RUBRIC_TEMPLATE.format(**descriptions)
                user_prompt = user_prompt_template.format(conversation=conversation, rubric=score_rubric)

                user_content = ABS_SYSTEM_PROMPT + "\n\n" + user_prompt
                messages = [{"role": "user", "content": user_content}]

                response = client(
                    messages,
                    model=model,
                    temperature=args.temperature,
                    seed=args.random_seed,
                )
                answer = get_answer(response)
                retry_cnt = 0
                max_retry = 10
                while "[RESULT]:" not in answer:
                    print(answer)
                    print("retry")
                    if max_retry < retry_cnt:
                        answer = None
                        break
                    response = client(messages, model=model, temperature=args.temperature, seed=None)
                    answer = get_answer(response)
                    retry_cnt += 1

                total_doc_eval_result[eval_target][scenario] = answer

                # Logging & save
                save_to_json(total_doc_eval_result, save_path)


    if args.eval_profile_consistency:
        # Load prompt
        system_prompt = file_to_string(os.path.join(args.prompt_dir, "eval_profile_consistency_system.txt"))
        user_prompt_template = file_to_string(os.path.join(args.prompt_dir, "eval_profile_consistency_user.txt"))

        # Set save path & save variables
        total_consistency_eval_result = {}
        save_path = os.path.join(result_path, f"{args.moderator}_profile_consistency_{args.trg_agent}.json")

        if not os.path.isfile(save_path):
            # Start evaluation
            for data in tqdm(dialogue_hists):
                # Load data per scenario
                scenario = data["hadm_id"]
                dialogue = data["dialog_history"]
                conversation = ""
                for utter in dialogue:
                    conversation += f"""\t{utter["role"]}: {utter["content"]}\n"""

                # Set up prompt & get llm response
                user_prompt = user_prompt_template.format(conversation=conversation)
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = client(messages, model=model, temperature=args.temperature, seed=args.random_seed)
                answer = get_answer(response)
                answer = re.search(r"\{.*\}", answer, re.DOTALL).group()
                try:
                    answer = json.loads(answer)
                except:
                    answer = answer

                # Save the result
                total_consistency_eval_result[scenario] = answer

            # Logging & save
            save_to_json(total_consistency_eval_result, save_path)

        total_consistency_eval_result = load_json(save_path)
        BERTscore_save_path = os.path.join(result_path, f"{args.moderator}_profile_consistency_BERTscore_{args.trg_agent}.json")
        LLMscore_save_path = os.path.join(result_path, f"{args.moderator}_profile_consistency_LLMscore_{args.trg_agent}.json")
        consistency_prompt = load_json(os.path.join(args.prompt_dir, "eval_profile_consistency.json"))

        BERT_SIM_result = {}
        LLM_SIM_result = {}

        if os.path.isfile(BERTscore_save_path):
            BERT_SIM_result = load_json(BERTscore_save_path)

        if os.path.isfile(LLMscore_save_path):
            LLM_SIM_result = load_json(LLMscore_save_path)

        embedding_model_name = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        embedding_model = AutoModel.from_pretrained(embedding_model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        for scenario, predict_dict in tqdm(total_consistency_eval_result.items()):
            profile_data = get_profile(scenario_dict, scenario)
            predict_dict = flatten_dict_simple(predict_dict)
            profile_data = {k: v for k, v in profile_data.items() if k in predict_dict.keys()}
            assert len(set(predict_dict.keys()).difference(profile_data)) == 0

            if scenario not in BERT_SIM_result:
                # BERT Sim
                bert_result = {}
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                for eval_key in predict_dict.keys():
                    if predict_dict[eval_key] != "Not recorded":
                        bert_result[eval_key] = compute_similarity(tokenizer, embedding_model, str(profile_data[eval_key]), str(predict_dict[eval_key]), cos).item()
                    else:
                        bert_result[eval_key] = None
                BERT_SIM_result[scenario] = bert_result

            if scenario not in LLM_SIM_result:
                # LLM Sim
                messages = deepcopy(consistency_prompt)
                filtered_predict_dict = {k: v for k, v in predict_dict.items() if v != "Not recorded"}
                filtered_profile_data = {k: v for k, v in profile_data.items() if k in filtered_predict_dict}
                messages[-1]["content"] = json.dumps({"GT_profile": filtered_profile_data, "Prediction_profile": filtered_predict_dict})
                llm_result, _ = get_valid_answer_with_retries(client, messages, model=model, temperature=args.temperature, random_seed=args.random_seed, expected_type="dict")
                LLM_SIM_result[scenario] = llm_result

                # Logging & save
                save_to_json(BERT_SIM_result, BERTscore_save_path)
                save_to_json(LLM_SIM_result, LLMscore_save_path)


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
            "vllm-llama4-scout",
            "vllm-qwen2.5-72b-instruct",
        ],
    )
    parser.add_argument("--moderator_api_type", type=str, default="vllm", choices=["gpt_azure", "vllm", "genai"])
    parser.add_argument("--trg_agent", type=str, default="Patient")
    parser.add_argument("--data_dir", type=str, default="./data/final_data")
    parser.add_argument("--data_file_name", type=str, default="patient_profile")
    parser.add_argument("--prompt_dir", type=str, default="./prompts/eval")
    parser.add_argument("--result_dir", type=str, default="./results", help="save dir")
    parser.add_argument("--trg_exp_name", type=str, default="", help="save dir")
    parser.add_argument("--eval_ddx", action="store_true", help="eval ddx performance")
    parser.add_argument("--eval_profile_consistency", action="store_true", help="eval response quality performance")
    parser.add_argument("--eval_persona_quality", action="store_true", help="eval response quality performance")
    parser.add_argument("--eval_doc_quality", action="store_true", help="eval response quality performance")

    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.random_seed)
    main(args)
