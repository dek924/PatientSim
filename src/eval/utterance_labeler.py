import os
import re
import ast
import sys
import json
import argparse

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from utils import load_json, load_jsonl, save_to_jsonl, file_to_string, set_seed


HISTORY_MODE = "history-patient"


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "labeler"


def process_answer(response):
    """Extract a JSON object from a model response."""
    if hasattr(response, "choices"):
        output = response.choices[0].message.content.strip()
    elif hasattr(response, "text"):
        output = response.text.strip()
    else:
        raise NotImplementedError("Failed to extract model response text.")

    output = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", output)
    output = re.sub(r"```\s*([\s\S]*?)\s*```", r"\1", output)

    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"({[\s\S]*})", output)
    if not match:
        raise ValueError(f"No valid JSON object found in: {output[:200]}")

    json_blob = match.group(1)
    try:
        return json.loads(json_blob)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(json_blob)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed response is not a JSON object.")
        return parsed


def get_valid_answer_with_retries(
    client, messages, model, temperature, random_seed, max_retries
):
    """Call the labeler model until a parseable JSON response is produced or retries are exhausted."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            seed = random_seed if attempt == 0 else None
            response = client(messages, model=model, temperature=temperature, seed=seed)
            return process_answer(response), None
        except Exception as exc:
            last_error = str(exc)
    return None, last_error


def format_patient_only_history(history_so_far):
    lines = []
    patient_turn_index = 0
    for turn in history_so_far:
        if turn.get("role") != "Patient":
            continue
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        patient_turn_index += 1
        lines.append(f"Patient[{patient_turn_index:02d}]: {content}")
    return "\n".join(lines)


def build_dialogue_rows(entries, source_dialogue_file):
    """Expand each saved dialogue into one row per patient utterance."""
    rows = []
    for entry in entries:
        dialog_history = entry.get("dialog_history") or []
        if not isinstance(dialog_history, list):
            continue

        base = {
            "hadm_id": entry.get("hadm_id"),
            "experiment_name": entry.get("experiment_name"),
            "source_dialogue_file": source_dialogue_file,
            "patient_engine_name": entry.get("patient_engine_name"),
            "doctor_engine_name": entry.get("doctor_engine_name"),
            "patient_api_type": entry.get("patient_api_type"),
            "doctor_api_type": entry.get("doctor_api_type"),
            "patient_prompt_file": entry.get("patient_prompt_file"),
            "doctor_prompt_file": entry.get("doctor_prompt_file"),
            "doctor_personality": entry.get("doctor_personality"),
            "cefr_type": entry.get("cefr_type"),
            "personality_type": entry.get("personality_type"),
            "recall_level_type": entry.get("recall_level_type"),
            "dazed_level_type": entry.get("dazed_level_type"),
            "diagnosis": entry.get("diagnosis"),
        }

        history_so_far = []
        patient_turn_index = 0

        for turn in dialog_history:
            role = turn.get("role") if isinstance(turn, dict) else None
            content = (
                str(turn.get("content", "")).strip() if isinstance(turn, dict) else ""
            )

            if role != "Patient":
                if isinstance(turn, dict):
                    history_so_far.append(turn)
                continue

            patient_turn_index += 1
            conversation_history = format_patient_only_history(history_so_far)

            rows.append(
                {
                    **base,
                    "turn": patient_turn_index,
                    "patient_utterance": content,
                    "conversation_history": conversation_history,
                }
            )

            if isinstance(turn, dict):
                history_so_far.append(turn)

    return rows


def format_label_prompt(template, dimension, patient_utterance, conversation_history):
    return (
        template.replace("{dimension_id}", dimension["id"])
        .replace("{question}", dimension["question"])
        .replace("{rubric}", "\n".join(dimension["rubric"]))
        .replace("{conversation_history}", conversation_history)
        .replace("{patient_utterance}", patient_utterance)
    )


def validate_dimension(parsed, dimension_id):
    """Validate the parsed JSON payload for one rubric dimension."""
    if not isinstance(parsed, dict):
        return None, None, "Parsed response is not a JSON object."

    score = parsed.get(dimension_id)
    if not isinstance(score, int):
        return None, None, f"Expected integer field '{dimension_id}'."
    if not 1 <= score <= 5:
        return None, None, f"'{dimension_id}' must be in the 1-5 range."

    rationale = parsed.get(f"{dimension_id}_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return (
            None,
            None,
            f"Expected non-empty rationale field '{dimension_id}_rationale'.",
        )

    return score, rationale.strip(), None


def build_output_path(result_path, labeler):
    labeler_tag = sanitize_filename(labeler)
    return os.path.join(
        result_path,
        "outputs",
        "eval",
        "utterance_labeler",
        f"labels__{labeler_tag}.jsonl",
    )


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--trg_exp_name", required=True)
    parser.add_argument("--labeler", required=True)
    parser.add_argument("--labeler_api_type", required=True)
    parser.add_argument("--prompt_dir", default="./prompts/eval/utterance_labeler")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(args):
    """Load dialogue outputs, label patient turns, and save flat JSONL results."""
    from models import get_response_method, vllm_model_setup

    set_seed(args.random_seed)

    result_path = os.path.join(args.result_dir, args.trg_exp_name)
    dialogue_path = os.path.join(result_path, "outputs", "dialogue.jsonl")
    output_path = build_output_path(result_path, args.labeler)
    prompt_template_path = os.path.join(args.prompt_dir, "utterance_label_template.txt")
    dimensions_path = os.path.join(args.prompt_dir, "utterance_label_dimensions.json")

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    prompt_template = file_to_string(prompt_template_path)
    dimensions = load_json(dimensions_path)
    dialogue_entries = load_jsonl(dialogue_path)
    rows = build_dialogue_rows(dialogue_entries, dialogue_path)

    client = get_response_method(args.labeler_api_type)
    model = (
        vllm_model_setup(args.labeler)
        if args.labeler_api_type == "vllm"
        else args.labeler
    )

    labeled_rows = []
    for row in tqdm(rows, desc="Labeling patient utterances"):
        labeled = {k: v for k, v in row.items() if k != "conversation_history"}
        labeled["labeling_model"] = args.labeler
        labeled["labeling_api_type"] = args.labeler_api_type
        labeled["labeling_mode"] = HISTORY_MODE
        labeled["label_prompt_template_file"] = prompt_template_path
        labeled["label_dimensions_file"] = dimensions_path

        patient_utterance = str(row.get("patient_utterance", ""))
        conversation_history = str(row.get("conversation_history", ""))
        labeling_errors = {}

        for dimension in dimensions:
            dim_id = dimension["id"]
            prompt = format_label_prompt(
                prompt_template,
                dimension,
                patient_utterance,
                conversation_history,
            )

            messages = [{"role": "user", "content": prompt}]
            parsed, request_error = get_valid_answer_with_retries(
                client,
                messages,
                model,
                args.temperature,
                args.random_seed,
                args.max_retries,
            )

            if request_error is not None:
                labeled[f"{dim_id}_score"] = None
                labeled[f"{dim_id}_rationale"] = None
                labeling_errors[dim_id] = request_error
                continue

            score, rationale, validation_error = validate_dimension(parsed, dim_id)
            labeled[f"{dim_id}_score"] = score
            labeled[f"{dim_id}_rationale"] = rationale
            if validation_error is not None:
                labeling_errors[dim_id] = validation_error

        labeled["labeling_errors"] = labeling_errors or None
        labeled_rows.append(labeled)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_to_jsonl(labeled_rows, output_path)
    print(f"Saved {len(labeled_rows)} labeled rows to {output_path}")


if __name__ == "__main__":
    parser = build_argument_parser()
    main(parser.parse_args())
