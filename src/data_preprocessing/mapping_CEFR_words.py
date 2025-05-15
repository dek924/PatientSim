import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_pickle(data_path):
    """
    Load a pickle file from the given data path.
    """
    print(f"Loading pickle file from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print("Pickle file loaded successfully.")
    return data


def save_to_json(data, output_file):
    """
    Save the given data to a JSON file with pretty formatting.
    """
    print(f"Saving data to JSON file: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Data saved to JSON successfully.")


def sample_words(words, num_sample):
    if len(words) >= num_sample:
        sampled = random.sample(words, num_sample)
    else:
        sampled = words
    return ", ".join(sampled)

def create_sampled_columns(df, word_dict, word_type, num_sample):
    columns = {}
    n = len(df)
    for level, words in word_dict.items():
        col_name = f"{word_type}_{level}"
        columns[col_name] = [sample_words(words, num_sample) for _ in range(n)]
    return pd.DataFrame(columns, index=df.index)


def main(args):
    # Seed fix
    set_seed(args.random_seed)

    # Data load
    data_path = os.path.join(args.data_dir, f"{args.data_file_name}.json")
    sample_df = pd.read_json(data_path, dtype={"hadm_id": str})
    present_illness_df = pd.json_normalize(sample_df["present_illness"]).add_prefix("present_illness_")
    sample_df = pd.concat([sample_df.drop(["present_illness"], axis=1), present_illness_df], axis=1)

    # CEFR words load
    cefr_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "CEFR_kaggle", "ENGLISH_CERF_WORDS.csv")
    medterm_dict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "cefr_word_dict.json")

    cefr_word_df = pd.read_csv(cefr_data_path)
    medterm_dict = json.load(open(medterm_dict_path))
    medterm_list = set(word.lower() for words in medterm_dict.values() for word in words)
    
    cefr_word_df['headword'] = cefr_word_df['headword'].str.lower()
    cefr_cnt = cefr_word_df.groupby("headword").nunique()
    valid_headwords = cefr_cnt[cefr_cnt.CEFR == 1].index

    cefr_word_df = cefr_word_df[
        cefr_word_df['headword'].isin(valid_headwords) &
        (cefr_word_df['headword'].str.len() > 3) &
        ~cefr_word_df['headword'].isin(medterm_list)
    ]
    cefr_word_dict = cefr_word_df.groupby("CEFR")["headword"].unique().apply(list).to_dict()

    print("Statistic of CEFR general vocabs")
    print(cefr_word_df.groupby("CEFR")["headword"].nunique())

    print("Statistic of CEFR medical terms")
    for key, val in medterm_dict.items():
        print(key, ": ", len(set(val)))

    # Mapping randomly sampled words per sample
    cefr_sampled_words_df = create_sampled_columns(sample_df, cefr_word_dict, "cefr", args.num_sample)
    med_sampled_words_df = create_sampled_columns(sample_df, medterm_dict, "med", args.num_sample)
    sample_df = pd.concat([sample_df, cefr_sampled_words_df, med_sampled_words_df], axis=1)

    # Save the results
    output_path = os.path.join(args.data_dir, f"{args.data_file_name}_w_cefr.json")
    save_to_json(sample_df.to_dict("records"), output_path)

    print(f"Sampled data saved to {output_path}")


if __name__ == "__main__":
    # Set up argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_name", type=str, default="sample_dict")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_sample", type=int, default=20) 
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
