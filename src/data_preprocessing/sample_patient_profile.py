import os
import json
import pickle
import argparse
import pandas as pd

from icdmappings import Mapper

# Mapping for diagnosis codes to standardized keys
DIAGNOSIS_MAPPING_KEYS = {
    "Pneumonia": [
        "pneumonia,organism unspecified",
        "pneumonia, unspecified organism",
        "other pneumonia, unspecified organism",
        "bronchopneumonia, unspecified organism",
        "bronchopneumonia,organism unspecified",
        "pneumococcal pneumonia",
    ],
    "Urinary tract infection": [
        "urin tract infection nos",
        "urinary tract infection, site not specified",
        "pyelonephritis nos",
        "acute pyelonephritis",
        "acute cystitis without hematuria",
        "cystitis, unspecified with hematuria",
        "acute cystitis with hematuria",
        "nonobstructive reflux-associated chronic pyelonephritis",
        "cystitis nos",
        "acute cystitis",
        "cystitis, unspecified without hematuria",
        "other cystitis without hematuria",
        "cystitis nec",
        "other cystitis with hematuria",
        "chr interstit cystitis",
        "irradiation cystitis with hematuria",
    ],
    "Myocardial infarction": [
        "non-st elevation (nstemi) myocardial infarction",
        "myocardial infarction nos, init episode of care",
        "subendocardial infarction, initial episode of care",
        "st elevation (stemi) myocardial infarction of unsp site",
        "stemi involving oth coronary artery of inferior wall",
        "ami inferior wall, initial episode of care",
        "stemi involving oth coronary artery of anterior wall",
        "stemi involving oth sites",
        "acute myocardial infarction, unspecified",
        "ami anterior wall nec, initial episode of care",
        "ami inferoposterior wall, initial episode of care",
        "myocardial infarction type 2",
    ],
    "Intestinal obstruction": [
        "intestinal obstruct nos",
        "unspecified intestinal obstruction",
        "other intestnl obst unsp as to partial versus complete obst",
        "unsp intestnl obst, unsp as to partial versus complete obst",
        "volvulus of intestine",
        "intussusception",
        "other partial intestinal obstruction",
        "partial intestinal obstruction, unspecified as to cause",
        "other intestinal obstruction",
        "intestinal obstruct nec",
    ],
    "Cerebral infarction": [
        "cereb infrc d/t unsp occls or stenos of left mid cereb art",
        "cereb infrc d/t unsp occls or stenos of right post cereb art",
        "cereb infrc due to unsp occls or stenos of left carotid art",
        "cerebral art occlus w/infarct",
        "cerebral infarction, unspecified",
        "cerebral infrc due to thombos of right post cerebral artery",
        "cerebral infrc due to thrombosis of right carotid artery",
        "occlus basilar art w/infarct",
        "occlus carotid art w/infarct",
        "occlus mult/bil art w/infarct",
        "occlus vertebral art w/infarct",
        "other cerebral infarction",
    ],
}


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


def main(args):
    """
    Main processing function that:
    - Loads various data files (notes, meta info, chart events, admissions, patients, ED stays, triage, and medication reconciliation data)
    - Processes and merges these datasets to create a comprehensive DataFrame containing patient information,
      diagnosis mapping, and additional clinical data.
    - Applies filtering criteria and sampling, and finally saves the sample DataFrame to a CSV file.
    """
    # Load admissions and patients data from hospital files
    print("Loading admissions and patients data...")
    admissions = pd.read_csv(
        os.path.join(args.mimic_dir, "hosp/admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime", "insurance", "language", "marital_status"],
    )
    print(f"Admissions data shape:")
    print(f"\tsubject_id: {admissions.subject_id.nunique()}")
    print(f"\thadm_id: {admissions.hadm_id.nunique()}")

    patients = pd.read_csv(os.path.join(args.mimic_dir, "hosp/patients.csv"), usecols=["subject_id", "anchor_age", "anchor_year"])
    print(f"Patients data shape:")
    print(f"\tsubject_id: {patients.subject_id.nunique()}")

    admissions = admissions.merge(patients, on="subject_id")
    admissions["age"] = pd.to_datetime(admissions["admittime"]).dt.year - admissions["anchor_year"] + admissions["anchor_age"]
    print(f"Admissions + Patients data shape: {admissions.shape}")
    print(f"\tsubject_id: {admissions.subject_id.nunique()}")
    print(f"\thadm_id: {admissions.hadm_id.nunique()}")

    # Remove outlier
    print("Remove NaN & Outliers...")
    admissions = admissions[(~admissions.marital_status.isna()) & (~admissions.insurance.isna())]
    print(f"Remove NaN & Outliers shape: {admissions.shape}")
    print(f"\tsubject_id: {admissions.subject_id.nunique()}")
    print(f"\thadm_id: {admissions.hadm_id.nunique()}")

    # Load emergency department (ED) related data
    print("Loading emergency department (ED) related data...")
    edstays = pd.read_csv(
        os.path.join(args.ed_dir, "edstays.csv"), usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender", "race", "arrival_transport", "disposition"]
    )
    triage = pd.read_csv(os.path.join(args.ed_dir, "triage.csv"), usecols=["stay_id", "chiefcomplaint", "pain"])
    medrecon = pd.read_csv(os.path.join(args.ed_dir, "medrecon.csv"))
    ed_diagnosis = pd.read_csv(os.path.join(args.ed_dir, "diagnosis.csv"), usecols=["subject_id", "stay_id", "seq_num", "icd_code", "icd_version", "icd_title"])
    print(f"ED stay data shape: {edstays.shape}")
    print(f"\tsubject_id: {edstays.subject_id.nunique()}")
    print(f"\thadm_id: {edstays.hadm_id.nunique()}")
    print(f"\tstay_id: {edstays.stay_id.nunique()}")

    print(f"ED diagnosis data shape: {ed_diagnosis.shape}")
    print(f"\tsubject_id: {ed_diagnosis.subject_id.nunique()}")
    print(f"\tstay_id: {ed_diagnosis.stay_id.nunique()}")
    ed_diagnosis["icd_title"] = ed_diagnosis["icd_title"].str.lower()
    diag_cnt = ed_diagnosis.groupby("stay_id").icd_title.nunique()
    ed_diagnosis = ed_diagnosis[ed_diagnosis.stay_id.isin(diag_cnt[diag_cnt == 1].index)].drop_duplicates()
    print(f"Remove diagnosis > 1: {ed_diagnosis.shape}")
    print(f"\tsubject_id: {ed_diagnosis.subject_id.nunique()}")
    print(f"\tstay_id: {ed_diagnosis.stay_id.nunique()}")

    # Merge ED stays with triage & diagnosis information
    print("Merging ED stays with triage & diagnosis data...")
    edstays = edstays.merge(triage, on=["stay_id"], how="inner")
    edstays = edstays.merge(ed_diagnosis, on=["subject_id", "stay_id"], how="inner")
    print(f"ED stay data shape: {edstays.shape}")
    print(f"\tsubject_id: {edstays.subject_id.nunique()}")
    print(f"\thadm_id: {edstays.hadm_id.nunique()}")
    print(f"\tstay_id: {edstays.stay_id.nunique()}")

    print("Merging ED stays with medication reconciliation data...")
    medrecon_list = pd.DataFrame(medrecon.groupby("stay_id").name.unique()).reset_index()
    edstays = edstays.merge(medrecon_list[["stay_id", "name"]].rename(columns={"name": "medication"}), on=["stay_id"], how="left")
    edstays["medication"] = edstays["medication"].str.join(", ")
    edstays["num_medication"] = edstays.medication.str.split(", ").str.len()

    # Remove outlier
    edstays["intime"] = pd.to_datetime(edstays["intime"])
    edstays["outtime"] = pd.to_datetime(edstays["outtime"])
    edstays["ed_stay_duration"] = edstays["outtime"] - edstays["intime"]
    edstays = edstays[edstays.ed_stay_duration.dt.total_seconds() > 0].drop_duplicates()
    print(f"Remove outlier (ED stay time < 0): {edstays.shape}")
    print(f"\tsubject_id: {edstays.subject_id.nunique()}")
    print(f"\thadm_id: {edstays.hadm_id.nunique()}")
    print(f"\tstay_id: {edstays.stay_id.nunique()}")

    # Remove outlier
    print("Remove nan...")
    edstays = edstays[(edstays.race != "UNKNOWN") & (~edstays.chiefcomplaint.isna()) & (edstays.arrival_transport != "UNKNOWN")]
    print(f"ED stay data shape: {edstays.shape}")
    print(f"\tsubject_id: {edstays.subject_id.nunique()}")
    print(f"\thadm_id: {edstays.hadm_id.nunique()}")
    print(f"\tstay_id: {edstays.stay_id.nunique()}")

    # Convert the 'pain' column to numeric values, coercing errors to NaN
    print("Converting 'pain' column to numeric values...")
    edstays["pain"] = pd.to_numeric(edstays["pain"], errors="coerce")
    edstays = edstays[~((edstays.pain > 10) | (edstays.pain < 0))]  # remove 13 (outliers)
    edstays = edstays.dropna(subset=["pain"])
    edstays = edstays[edstays.num_medication < 16]
    print(f"Remove pain & medication outlier: {edstays.shape}")
    print(f"\tsubject_id: {edstays.subject_id.nunique()}")
    print(f"\thadm_id: {edstays.hadm_id.nunique()}")
    print(f"\tstay_id: {edstays.stay_id.nunique()}")

    # Merge admissions with ED stays data and drop unnecessary columns
    print("Merging admissions data with ED stays...")
    admissions = admissions.merge(edstays, on=["subject_id", "hadm_id"], how="inner").drop(columns=["admittime", "anchor_age", "anchor_year"])
    print(f"Admissions data after merge shape: {admissions.shape}")
    print(f"\tsubject_id: {admissions.subject_id.nunique()}")
    print(f"\thadm_id: {admissions.hadm_id.nunique()}")

    _cnt = admissions.groupby("hadm_id").stay_id.nunique()
    admissions = admissions[admissions.hadm_id.isin(_cnt[_cnt == 1].index)].drop_duplicates()
    print(f"""Remove hadm with more than 1 ED stay: -> {admissions.shape}""")
    print(f"\tsubject_id: {admissions.subject_id.nunique()}")
    print(f"\thadm_id: {admissions.hadm_id.nunique()}")
    print(f"\tstay_id: {admissions.stay_id.nunique()}")

    print("Loading note sections and meta information...")
    # Load note sections and meta information DataFrames
    note_df = pd.read_csv(os.path.join(args.preprocess_dir, "note_section.csv"))
    print(f"Loaded note_df with shape: {note_df.shape}")
    print(f"\tsubject_id: {note_df.subject_id.nunique()}")
    print(f"\thadm_id: {note_df.hadm_id.nunique()}")

    print("Merging note sections with existing DataFrame...")
    admissions = admissions.merge(note_df, on=["subject_id", "hadm_id"])
    print(f"Merging note information: {admissions.shape}")
    print(f"\tsubject_id: {admissions.subject_id.nunique()}")
    print(f"\thadm_id: {admissions.hadm_id.nunique()}")
    print(f"\tstay_id: {admissions.stay_id.nunique()}")
    # print(filtered_df.groupby("mapped_icd_title").stay_id.nunique())

    admissions = admissions[(~admissions["chiefcomplaint"].str.lower().str.contains("transfer|abnormal"))].reset_index(drop=True)
    admissions["hadm_id"] = admissions["hadm_id"].astype(int).astype(str)
    output_path = os.path.join(args.save_dir, "filtered_all_df.csv")
    admissions.to_csv(output_path, index=False)

    # Create a mapping dictionary for diagnosis codes to their respective standard keys
    mapping_dict = {code: key for key, codes in DIAGNOSIS_MAPPING_KEYS.items() for code in codes}
    merged_icd_titles = [code for codes in DIAGNOSIS_MAPPING_KEYS.values() for code in codes]
    filtered_df = admissions[admissions.icd_title.isin(merged_icd_titles)]
    filtered_df["mapped_icd_title"] = filtered_df["icd_title"].map(mapping_dict)
    print(f"Filtered meta_info for mapped ICD titles. Shape: {filtered_df.shape}")
    print(f"\tsubject_id: {filtered_df.subject_id.nunique()}")
    print(f"\thadm_id: {filtered_df.hadm_id.nunique()}")
    print(f"\tstay_id: {filtered_df.stay_id.nunique()}")
    print(filtered_df.groupby("mapped_icd_title").stay_id.nunique())

    # Filter out records with chief complaints containing "transfer" (case-insensitive)
    # print("Applying filters on chief complaint and past medical history...")
    print("Remove slurred speech...")
    filtered_df = filtered_df[(~filtered_df["chiefcomplaint"].str.lower().str.contains("slurred speech|transfer|abnormal|dysarthria|aphasia|ams|altered mental status|confusion"))].reset_index(drop=True)
    print(f"Remove slurred speech / transfer: {filtered_df.shape}")
    print(f"\tsubject_id: {filtered_df.subject_id.nunique()}")
    print(f"\thadm_id: {filtered_df.hadm_id.nunique()}")
    print(f"\tstay_id: {filtered_df.stay_id.nunique()}")
    print(filtered_df.groupby("mapped_icd_title").stay_id.nunique())

    # Sample n records per 'mapped_icd_title'
    print("Sampling records per mapped ICD title...")
    filtered_df = (
        filtered_df.groupby("mapped_icd_title", group_keys=False).apply(lambda x: x.sample(n=min(args.num_sample, len(x)), random_state=args.random_seed)).reset_index(drop=True)
    )
    print(f"Final sampled data shape: {filtered_df.shape}")
    print(f"\tsubject_id: {filtered_df.subject_id.nunique()}")
    print(f"\thadm_id: {filtered_df.hadm_id.nunique()}")
    print(filtered_df.groupby("mapped_icd_title").subject_id.nunique())


    # Save the final sampled DataFrame to a CSV file
    filtered_df = filtered_df.drop(columns=["stay_id"])
    output_path = os.path.join(args.save_dir, "sample_df.csv")
    filtered_df.to_csv(output_path, index=False)
    print(f"Sampled data saved to {output_path}")


if __name__ == "__main__":
    # Set up argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_dir", type=str, default="./data/mimic-iv-3.1/")
    parser.add_argument("--ed_dir", type=str, default="./data/mimic-iv-ed-2.0/2.2/ed")
    parser.add_argument("--preprocess_dir", type=str, default="./preprocess")
    parser.add_argument("--save_dir", type=str, default="./profile_data")
    parser.add_argument("--num_sample", type=int, default=40)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="debug or not")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
