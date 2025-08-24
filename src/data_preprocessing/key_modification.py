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
    "present_illness",
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
    filtering_results_path = os.path.join(args.key_dir, f"{args.model}_filtering_results.json")
    filtered_target_df = pd.read_json(filtering_results_path).T.reset_index().rename(columns={"index": "hadm_id"})
    filtered_target_df["hadm_id"] = filtered_target_df["hadm_id"].astype(str)

    with open(results_path, "r") as f:
        raw_results = json.load(f)

    results_df = pd.DataFrame.from_dict(raw_results, orient="index").reset_index().rename(columns={"index": "hadm_id"})
    demog_df = pd.json_normalize(results_df["demographics"])
    social_df = pd.json_normalize(results_df["social_history"])

    results_df = pd.concat([results_df.drop(["demographics", "social_history"], axis=1), demog_df, social_df], axis=1)
    results_df = results_df[results_df.hadm_id.isin(filtered_target_df[filtered_target_df.likelihood_rating > 2].hadm_id.values)]
    print(results_df.columns)
    merged = df.merge(results_df, on="hadm_id", how="inner")
    merged = merged.rename(columns={"mapped_icd_title": "diagnosis"})
    print(merged.columns)
    print(merged.shape)
    merged = merged[USED_COLUMNS]

    # Extract dataset using chatgpt model
    final_results = []
    for i, row in merged.iterrows():
        hadm_id = row["hadm_id"]
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

        def flatten_dict(d):
            flat = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    flat.update(flatten_dict(v))
                else:
                    flat[k] = v
            return flat

        init_data = merged[merged["hadm_id"] == hadm_id].to_dict(orient="records")[0]
        answer = flatten_dict(answer)

        final_output = {k: answer[k] if k in answer else v for k, v in init_data.items()}
        final_results.append(final_output)

    final_results = sorted(final_results, key=lambda x: x["hadm_id"])
    save_to_json(final_results, os.path.join(args.key_dir, f"{args.model}_mod_results.json"))


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
            "gemini-2.5-flash",
        ],
    )
    parser.add_argument("--model_api_type", type=str, default="genai", choices=["gpt_azure", "genai"])
    parser.add_argument("--temperature", type=float, default=0.0, help="model temperature")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--thinking_budget", type=int, default=1024)

    parser.add_argument("--data_dir", type=str, default="./data", help="save dir")
    parser.add_argument("--key_dir", type=str, default="./results/key_extraction/", help="key_dir")
    parser.add_argument("--prompt_dir", type=str, default="./prompts/key_modification", help="save directory")

    args = parser.parse_args()

    main(args)
