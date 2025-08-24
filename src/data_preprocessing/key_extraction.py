import os
import re
import sys
import ast
import json
import logging
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_to_string, find_missing_keys, save_to_json
from models import get_response_method, vllm_model_setup, get_answer


def main(args):
    model = args.model
    logging.info(f"Using LLM: {model}")

    # Load text prompts
    system_prompt = file_to_string(os.path.join(args.prompt_dir, "initial_system.txt"))
    user_prompt_template = file_to_string(os.path.join(args.prompt_dir, "initial_user.txt"))

    # Loading dataset
    df = pd.read_csv(os.path.join(args.data_dir, "sample_df.csv"), dtype={"hadm_id": str})
    if args.debug:
        df = df[:5]
        
    print(df.shape)
    
    print(f"{args.model_api_type} api call")
    client = get_response_method(args.model_api_type)
    model = vllm_model_setup(args.model) if "vllm" in args.model else args.model

    # Extract dataset using chatgpt model
    total_result = {}
    for i, row in df.iterrows():
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

        total_result[hadm_id] = answer
    
    total_result = dict(sorted(total_result.items()))
    save_to_json(total_result, os.path.join(args.save_dir, f"{args.model}_results.json"))


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
    parser.add_argument("--save_dir", type=str, default="./results/key_extraction", help="save dir")
    parser.add_argument("--prompt_dir", type=str, default="./prompts/key_extraction", help="save directory")
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if args.exp_name is None:
        if args.thinking_budget > 0:
            args.exp_name = f"{now}_{args.model}_thinking_{args.thinking_budget}"
        else:
            args.exp_name = f"{now}_{args.model}"
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    print(args.save_dir)

    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.save_dir, "key_extraction.log"), level=logging.INFO)

    main(args)
