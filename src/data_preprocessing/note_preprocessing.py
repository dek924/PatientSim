import os
import argparse
import pandas as pd


def print_statistic(df):
    for col in ["subject_id", "hadm_id", "note_id"]:
        print(f"""{col}: {df[col].nunique()}""")
    print()


def word_cnt_hpi(row):
    return len(row["text"][row["hpi_index"] : row["pmh_index"]].split())


def word_cnt(text):
    if isinstance(text, str):
        return len(text.split())
    else:
        0


def split_section(row):
    if (row["start_idx"] == -1) or (row["end_idx"] == -1):
        return None
    else:
        return row["text"][row["start_idx"] : row["end_idx"]]


def split_section_multi_key(row, start_key):
    if (row["end_idx_1"] == -1) and (row["end_idx_2"] == -1):
        return None
    end_index = min(pos for pos in [row["end_idx_1"], row["end_idx_2"]] if pos != -1)
    if (row["start_idx"] == -1) or (end_index == -1):
        return None
    else:
        return row["text"][row["start_idx"] + len(start_key) : end_index]


def split_history_section(data_df, start_key, end_key1, endkey2):
    data_df["start_idx"] = data_df.text.str.lower().str.find(start_key.lower())
    data_df["end_idx_1"] = data_df.text.str.lower().str.find(end_key1.lower())
    data_df["end_idx_2"] = data_df.text.str.lower().str.find(endkey2.lower())
    data_df[start_key.replace(":", "")] = data_df.apply(lambda x: split_section_multi_key(x, start_key=start_key), axis=1)
    data_df[start_key.replace(":", "")] = data_df[start_key.replace(":", "")].str.replace("\n", " ").str.strip()
    return data_df


def main(args):
    # Load discharge summaries
    print("Load note dataframes...")
    note_df = pd.read_csv(os.path.join(args.note_dir, "discharge.csv.gz"))
    print_statistic(note_df)
    print()

    # Filtering the note which contains minimum information about hpi, pmh
    note_df["hpi_index"] = note_df.text.str.lower().str.find("History of Present Illness:".lower())
    note_df["pmh_index"] = note_df.text.str.lower().str.find("Past Medical History:".lower())
    note_df["word_cnt"] = note_df.apply(word_cnt, axis=1)
    note_df = note_df[(note_df.hpi_index > 0) & (note_df.pmh_index > 0)]
    print("hpi & pmh section check")
    print_statistic(note_df)
    print()

    # Split the section of free-text report
    print("Split the note section...")
    token_list = ["Major Surgical or Invasive Procedure:", "History of Present Illness:"]
    for idx in range(len(token_list) - 1):
        print(token_list[idx][:-1])
        note_df["start_idx"] = note_df.text.str.lower().str.find(token_list[idx].lower())
        note_df["start_idx"] = note_df["start_idx"] + len(token_list[idx])
        note_df["end_idx"] = note_df.text.str.lower().str.find(token_list[idx + 1].lower())
        note_df[token_list[idx][:-1]] = note_df.apply(split_section, axis=1)
        note_df[token_list[idx][:-1]] = note_df[token_list[idx][:-1]].str.replace("\n", " ").str.strip()

    note_df = split_history_section(note_df, start_key="Allergies:", end_key1="Attending:", endkey2="Chief Complaint:")
    note_df = split_history_section(note_df, start_key="Complaint:", end_key1="Major Surgical or Invasive Procedure:", endkey2="History of Present Illness:")
    note_df = split_history_section(note_df, start_key="History of Present Illness:", end_key1="Past Medical History:", endkey2="PMH:")
    note_df = split_history_section(note_df, start_key="Past Medical History:", end_key1="Social History:", endkey2="Family History:")
    note_df = split_history_section(note_df, start_key="Social History:", end_key1="Family History:", endkey2="Physical Exam:")
    note_df = split_history_section(note_df, start_key="Family History:", end_key1="Physical Exam:", endkey2="Discharge exam:")
    note_df["pmi_word_cnt"] = note_df["Past Medical History"].apply(word_cnt)
    note_df["hpi_word_cnt"] = note_df["History of Present Illness"].apply(word_cnt)
    note_df["total_word_cnt"] = note_df["pmi_word_cnt"] + note_df["hpi_word_cnt"]

    print_statistic(note_df)
    print(note_df.hpi_word_cnt.describe())
    print(note_df.pmi_word_cnt.describe())
    print(note_df.total_word_cnt.describe())
    print()
    
    print("Remove nan section")
    note_df = note_df[~note_df["Complaint"].isna() & ~note_df["History of Present Illness"].isna() & ~note_df["Past Medical History"].isna()]
    print_statistic(note_df)
    print(note_df.hpi_word_cnt.describe())
    print(note_df.pmi_word_cnt.describe())
    print(note_df.total_word_cnt.describe())
    print()

    print("Remove outliers")
    note_df = note_df[(note_df.hpi_word_cnt > 10) & (note_df.hpi_word_cnt < 350) & (note_df.pmi_word_cnt < 80)]
    print_statistic(note_df)
    print(note_df.hpi_word_cnt.describe())
    print(note_df.pmi_word_cnt.describe())
    print(note_df.total_word_cnt.describe())
    print()

    # Filtered coma cases
    note_df = note_df[(~note_df.text.str.lower().str.contains("stupor|coma"))]
    print("Remove coma cases")
    print_statistic(note_df)
    print(note_df.hpi_word_cnt.describe())
    print(note_df.pmi_word_cnt.describe())
    print(note_df.total_word_cnt.describe())
    print()

    # Filtered dysarthria / memory issue patients
    note_df = note_df[(~note_df["Complaint"].str.lower().str.contains("ams|altered mental status|confusion"))]
    print("Remove confused cases")
    print_statistic(note_df)
    print(note_df.hpi_word_cnt.describe())
    print(note_df.pmi_word_cnt.describe())
    print(note_df.total_word_cnt.describe())
    print()

    note_df = note_df[(~note_df["History of Present Illness"].str.lower().str.contains("dysarthria|aphasia|slurred speech"))]
    print("Remove speech problem cases")
    print_statistic(note_df)
    print(note_df.hpi_word_cnt.describe())
    print(note_df.pmi_word_cnt.describe())
    print(note_df.total_word_cnt.describe())
    print()

    note_df = note_df[
        [
            "subject_id",
            "hadm_id",
            "Allergies",
            "Complaint",
            "Major Surgical or Invasive Procedure",
            "History of Present Illness",
            "Past Medical History",
            "Social History",
            "Family History",
        ]
    ].reset_index(drop=True)
    note_df.to_csv(os.path.join(args.save_dir, "note_section.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--note_dir", type=str, default="./physionet.org/files/mimic-iv-note/2.2/note")
    parser.add_argument("--save_dir", type=str, default="./preprocess")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
