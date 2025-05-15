import os
import re
import yaml
import json
import torch
import random
import logging
import jsonlines
import numpy as np


def prompt_valid_check(prompt, data_dict):
    missing_keys = find_missing_keys(prompt, data_dict)
    assert missing_keys == [], f"Missing keys: {missing_keys}"


def find_missing_keys(format_string, data):
    # Extract all keys enclosed in curly braces from the format string
    keys = re.findall(r'\{(.*?)\}', format_string)
    # Return a list of keys that are not present in the data dictionary
    return [key for key in keys if key not in data]


def check_all_patterns_present(text):
    patterns = [r"1\..*", r"2\..*", r"3\..*", r"4\..*", r"5\..*"]
    return all(re.search(pattern, text) for pattern in patterns)


def detect_termination(response):
    end_flag = False
    ddx_key = ["ddx ready:", "my top five differential diagnoses", "my top 5", "here are my top", "[ddx]", "[ddx", "here are some potential concerns we need to consider", "following differential diagnoses"]
    all_present = check_all_patterns_present(response)
    end_flag = any(key.lower() in response.lower() for key in ddx_key)
    return all_present or end_flag


def process_string(input_string):
    # Remove content inside parentheses and the parentheses themselves
    step1 = re.sub(r"\([^)]*\)", "", input_string)
    # Remove content inside asterisks and the asterisks themselves
    step2 = re.sub(r"\*\*[^*]*\*\*", "", step1)
    # Clean up extra spaces caused by the removal
    result = re.sub(r"\s+", " ", step2).strip()
    return result


def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def file_to_string(filename):
    with open(filename, "r", errors="ignore") as file:
        return file.read()


def load_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def load_jsonl(filename):
    with jsonlines.open(filename, "r") as file:
        data_list = [line for line in file]
    return data_list


def save_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_to_jsonl(data_list, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for req in data_list:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def save_to_dialogue(data, output_file):
    with jsonlines.open(output_file, mode="a") as writer:
        writer.write(data)


def get_profile(scenario_dict, trg_id):
    for profile in scenario_dict:
        if str(int(profile["hadm_id"])) == str(int(trg_id)):
            return profile


def log_and_print(message):
    """Logs a message to both the console and a log file."""
    print(message)  # Print to console
    logging.info(message)  # Log to file


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False