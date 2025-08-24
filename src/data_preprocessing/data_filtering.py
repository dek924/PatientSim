import os
import re
import sys
import ast
import json
import time
import logging
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_to_string, find_missing_keys, save_to_json
from models import get_response_method, vllm_model_setup, get_answer


USED_COLUMNS = [
    "hadm_id",
    "age",
    "gender",
    "race",
    "marital_status",
    "insurance",
    "occupation",
    "living_situation",
    "children",
    "exercise",
    "tobacco",
    "alcohol",
    "illicit_drug",
    "sexual_history",
    "allergies",
    "family_medical_history",
    "medical_device",
    "medical_history",
    "present_illness_positive",
    "present_illness_negative",
    "chiefcomplaint",
    "pain",
    "medication",
    "arrival_transport",
    "disposition",
    "diagnosis",
]


def main(args):
    model = args.model
    logging.info(f"Using LLM: {model}")

    # Load text prompts
    system_prompt = file_to_string(os.path.join(args.prompt_dir, "initial_system.txt"))
    user_prompt_template = file_to_string(os.path.join(args.prompt_dir, "initial_user.txt"))

    # Load dataset
    df = pd.read_csv(os.path.join(args.data_dir, "sample_df.csv"), dtype={"hadm_id": str})
    print(df.shape)

    print(f"{args.model_api_type} api call")
    client = get_response_method(args.model_api_type)
    model = vllm_model_setup(args.model) if "vllm" in args.model else args.model

    # Load JSON key data and merge with the dataset
    results_path = os.path.join(args.key_dir, f"{args.model}_results.json")
    with open(results_path, "r") as f:
        raw_results = json.load(f)

    results_df = pd.DataFrame.from_dict(raw_results, orient="index").reset_index().rename(columns={"index": "hadm_id"})
    demog_df = pd.json_normalize(results_df["demographics"])
    social_df = pd.json_normalize(results_df["social_history"])
    present_illness_df = pd.json_normalize(results_df["present_illness"]).add_prefix("present_illness_")
    results_df = pd.concat([results_df.drop(["demographics", "social_history", "present_illness"], axis=1), demog_df, social_df, present_illness_df], axis=1)
    print(results_df.columns)

    merged = df.merge(results_df, on="hadm_id", how="inner")
    merged = merged.rename(columns={"mapped_icd_title": "diagnosis"})
    print(merged.columns)
    merged = merged[USED_COLUMNS]

    if args.debug:
        merged = merged[:1]

    # Extract dataset using chatgpt model
    final_results = {}
    for i, row in merged.iterrows():
        hadm_id = str(int(row["hadm_id"]))
        user_prompt = user_prompt_template.format(**row)
        missing_keys = find_missing_keys(user_prompt_template, row)
        assert missing_keys == [], f"Missing keys in row {i}: {missing_keys}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        response_cur = None
        logging.info(f"Extract information with {args.model}")
        logging.info(f"messages:")
        for message in messages:
            _role = message["role"]
            _content = message["content"]
            logging.info(f"\t{_role}: {_content}")

        response_cur = client(model=model, message=messages, temperature=args.temperature, seed=args.random_seed, thinking_budget=args.thinking_budget)
        output = get_answer(response_cur)

        # Logging Token Information
        logging.info(f"GPT Output:\n " + output + "\n")
        output = re.search(r"\{.*\}", output, re.DOTALL).group()

        try:
            answer = ast.literal_eval(output)
            answer = json.dumps(answer)
            answer = json.loads(answer)
        except:
            answer = output

        final_results[hadm_id] = answer

    save_to_json(final_results, os.path.join(args.key_dir, f"{args.model}_filtering_results.json"))


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        choices=[
            "gpt-4o",
            "gemini-2.5-flash"
        ],
    )
    parser.add_argument("--model_api_type", type=str, default="genai", choices=["gpt_azure", "genai"])
    parser.add_argument("--temperature", type=float, default=0.0, help="model temperature")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--thinking_budget", type=int, default=1024)

    parser.add_argument("--data_dir", type=str, default="./data", help="save dir")
    parser.add_argument("--key_dir", type=str, default="./results/key_extraction", help="key_dir")
    parser.add_argument("--prompt_dir", type=str, default="./prompts/data_filtering", help="save directory")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logging.basicConfig(filename=os.path.join(args.key_dir, "filtering_log.log"), level=logging.INFO)

    main(args)
