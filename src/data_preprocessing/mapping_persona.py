import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from itertools import product


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def random_sample_diag(df, num_samples, random_seed=42):
    diagnosis_values = sorted(df['diagnosis'].unique())
    num_samples_per_diagnosis = num_samples // len(diagnosis_values)
    allocation = {diag: num_samples_per_diagnosis for diag in diagnosis_values}
    remaining_capacity = {}

    selected_samples = []
    for diagnosis, count in allocation.items():
        diag_df = df[df['diagnosis'] == diagnosis]
        if len(diag_df) >= count:
            samples = diag_df.sample(count, random_state=random_seed)['hadm_id'].tolist()
        else:
            samples = diag_df['hadm_id'].tolist()
            print(f"Warning: Only {len(samples)} samples available for diagnosis '{diagnosis}' (requested {count})")
        selected_samples.extend(samples)
        remaining_capacity[diagnosis] = len(diag_df) - len(samples)

    if len(selected_samples) < num_samples:
        remaining = num_samples - len(selected_samples)
        extra_df = df[~df['hadm_id'].isin(selected_samples)]
        diags_extra = [d for d, cap in remaining_capacity.items() if cap > 0]
        if diags_extra:
            base_extra = remaining // len(diags_extra)
            leftover = remaining % len(diags_extra)
            extra_alloc = {d: min(cap, base_extra) for d, cap in remaining_capacity.items() if cap > 0}
            for d in np.random.permutation(diags_extra)[:leftover]:
                if extra_alloc[d] < remaining_capacity[d]:
                    extra_alloc[d] += 1
                    
            for diag, n_extra in extra_alloc.items():
                if n_extra > 0:
                    diag_df = df[(df['diagnosis'] == diag) & (~df['hadm_id'].isin(selected_samples))]
                    extra_ids = diag_df.sample(n_extra, random_state=random_seed)['hadm_id'].tolist()
                    selected_samples.extend(extra_ids)
                    
        remaining = num_samples - len(selected_samples)
        extra_df = df[~df['hadm_id'].isin(selected_samples)]
        if len(extra_df) >= remaining:
            extra_samples = extra_df.sample(remaining, random_state=args.random_seed)['hadm_id'].tolist()
            selected_samples.extend(extra_samples)
        else:
            print(f"Warning: Could not find enough samples to reach target of {num_samples}")
    assert (len(set(selected_samples)) == num_samples)
    return selected_samples


def assign_labelers_balanced(df, labelers, k=1, random_seed=42):
    total_counts = {l: 0 for l in labelers}
    diag_counts  = {diag: {l: 0 for l in labelers} for diag in df['diagnosis'].unique()}
    assert len(df) % len(labelers) == 0, "No labelers provided"
    assignment = {}
    df_shuf = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    for _, row in df_shuf.iterrows():
        hid  = row['hadm_id']
        diag = row['diagnosis']
        candidates = sorted(labelers, key=lambda l: (diag_counts[diag][l], total_counts[l], random.random()))
        chosen = candidates[:k]
        assignment[hid] = chosen if k > 1 else chosen[0]
        for l in chosen:
            diag_counts[diag][l] += 1
            total_counts[l] += 1

    return assignment


def assign_labelers_unique_per_group(df, labelers, random_seed=None):
    total_counts = {l: 0 for l in labelers}
    diag_counts  = {diag: {l: 0 for l in labelers} for diag in df['diagnosis'].unique()}
    assert len(df) % len(labelers) == 0, "No labelers provided"
    quota = len(df) // len(labelers)
    df = df.copy()
    df['labeler'] = None
    for combo, group in df.groupby(['personality', 'cefr', 'recall_level']):
        m = len(group)
        if m > len(labelers):
            raise ValueError(
                f"There are {m} samples for combination {combo}, "
                f"which exceeds the number of labelers ({len(labelers)}); "
                "cannot assign uniquely."
            )

        used_in_group = set()
        group = group.sample(frac=1, random_state=random_seed)
        for idx, row in group.iterrows():
            diag = row['diagnosis']
            candidates = [
                l for l in labelers
                if l not in used_in_group and total_counts[l] < quota
            ]
            if not candidates:
                raise ValueError(f"No candidates available for assignment for comb '{combo}'")
            candidates.sort(key=lambda l: (diag_counts[diag][l], total_counts[l], random.random()))
            chosen = candidates[0]
            df.at[idx, 'labeler'] = chosen
            used_in_group.add(chosen)
            total_counts[chosen]  += 1
            diag_counts[diag][chosen] += 1
    assert all(c == quota for c in total_counts.values()), "Labeler counts are not balanced across diagnoses"
    return df

    
def main(args):
    # Seed fix
    set_seed(args.random_seed)

    # Set up options
    labelers = ['A', 'B', 'C', 'D']
    personality_options = ["plain", "verbose", "pleasing", "impatient", "overanxious", "distrust"]
    cefr_options = ["A", "B", "C"]
    recall_options = ["low", "high"]
    comb_options = list(product(*[personality_options, cefr_options, recall_options]))

    # Set up arguments    
    num_total_info_sample = args.num_total_info_sample
    num_total_persona_sample = args.num_total_persona_sample
    num_total_dazed_sample = args.num_total_dazed_sample
    num_labeler_per_info_sample = args.num_labeler_per_info_sample
    num_comb = len(comb_options)

    # Data load
    data_path = os.path.join(args.data_dir, f"{args.data_file_name}.json")
    sample_df = pd.read_json(data_path, dtype={"hadm_id": str})
    print(sample_df.shape)
    print(sample_df.hadm_id.nunique())
    print(sample_df['diagnosis'].value_counts())
    print()
    print(sample_df['gender'].value_counts())

    # Sampled Valid Dataset
    num_valid_samples = len(sample_df) - num_total_info_sample - num_total_persona_sample
    valid_samples = random_sample_diag(sample_df, num_valid_samples)
    valid_df = sample_df[sample_df['hadm_id'].isin(valid_samples)].copy()
    valid_df["split"] = "valid"
    print(f"valid samples: {len(valid_df)}")
    print(valid_df['diagnosis'].value_counts())

    # Sampled info Dataset
    sample_df_wo_valid = sample_df[~sample_df['hadm_id'].isin(valid_samples)].copy()
    info_samples = random_sample_diag(sample_df_wo_valid, num_total_info_sample)
    info_df = sample_df_wo_valid[sample_df_wo_valid['hadm_id'].isin(info_samples)].copy()
    print(f"Total info samples: {len(info_samples)}")
    print(info_df['diagnosis'].value_counts())

    # Assigned labelers for info
    info_hadm_ids = info_df['hadm_id'].tolist()
    info_hadm_ids = np.random.permutation(info_hadm_ids)
    strat_map = assign_labelers_balanced(info_df[['hadm_id', 'diagnosis']], labelers, k=num_labeler_per_info_sample, random_seed=args.random_seed)
    info_df['assigned_labelers'] = info_df['hadm_id'].map(lambda hid: strat_map[hid])
    info_df["split"] = "info"

    for labeler in labelers:
        labeler_df = info_df[
            info_df['assigned_labelers']
                        .apply(lambda labs: labeler in labs)
        ]
        
        save_to_json(labeler_df.to_dict("records"), os.path.join(args.save_dir, "per_labeler", f"info_{labeler}.json"))
        print(labeler_df.shape)
        print(labeler_df.diagnosis.value_counts())

    # Sampled persona Dataset (Dazed)
    dazed_df = sample_df[sample_df['arrival_transport'] == 'AMBULANCE']
    available_dazed_df = dazed_df[~dazed_df['hadm_id'].isin(info_samples + valid_samples)].copy()
    dazed_sampleds = random_sample_diag(available_dazed_df, num_total_dazed_sample)
    dazed_persona_df = sample_df[sample_df['hadm_id'].isin(dazed_sampleds)].copy()
    dazed_persona_df["split"] = "persona"
    dazed_persona_df["personality"] = "plain"
    dazed_persona_df["cefr"] = "B"
    dazed_persona_df["recall_level"] = "high"
    dazed_persona_df["dazed_level"] = "high"

    # Assigned labelers for dazed
    strat_map_dazed_persona = assign_labelers_balanced(dazed_persona_df[['hadm_id', 'diagnosis']], labelers, k=1, random_seed=args.random_seed)
    dazed_persona_df['labeler'] = dazed_persona_df['hadm_id'].map(lambda hid: strat_map_dazed_persona[hid])
    print(f"Dazed samples: {len(dazed_persona_df)}/{len(available_dazed_df)}/{len(dazed_df)}")
    print(available_dazed_df['diagnosis'].value_counts())
    print(dazed_persona_df['diagnosis'].value_counts())
    print(dazed_persona_df.groupby('labeler')['diagnosis'].value_counts())

    # Sampled persona Dataset (Non-Dazed)
    remaining_df = sample_df[~sample_df['hadm_id'].isin(info_samples + valid_samples + dazed_sampleds)].copy()
    persona_samples = random_sample_diag(remaining_df, num_total_persona_sample - num_total_dazed_sample)
    persona_df = sample_df[sample_df['hadm_id'].isin(persona_samples)].copy()
    print(f"Total persona samples: {len(persona_samples)}")
    print(remaining_df['diagnosis'].value_counts())
    print(dazed_persona_df['diagnosis'].value_counts())

    # Randomly assigned personality, CEFR, recall
    persona_hadm_ids = persona_df['hadm_id'].tolist()
    persona_hadm_ids = np.random.permutation(persona_hadm_ids)
    comb_options = list(product(personality_options, cefr_options, recall_options))
    repeats_per_comb = len(persona_hadm_ids) // num_comb
    remainder = len(persona_hadm_ids) % num_comb

    assignments = []
    for i, combo in enumerate(comb_options):
        count = repeats_per_comb + (1 if i < remainder else 0)
        assignments.extend([combo] * count)
    assignments = np.random.permutation(assignments)
    assert len(assignments) == len(persona_hadm_ids)

    mapping = dict(zip(persona_hadm_ids, assignments))
    persona_df["split"] = "persona"
    persona_df['personality'] = persona_df['hadm_id'].map(lambda x: mapping[x][0])
    persona_df['cefr'] = persona_df['hadm_id'].map(lambda x: mapping[x][1])
    persona_df['recall_level'] = persona_df['hadm_id'].map(lambda x: mapping[x][2])
    persona_df["dazed_level"] = "normal"

    # Assign labelers for persona
    persona_df = assign_labelers_unique_per_group(persona_df, labelers, random_seed=args.random_seed)
    print(persona_df.groupby('labeler').diagnosis.nunique().value_counts())
    print(persona_df.groupby(['personality', 'cefr', 'recall_level']).labeler.nunique().value_counts())

    for labeler in labelers:
        sub = persona_df[persona_df['labeler'] == labeler]
        combo_counts = sub.groupby(['personality', 'cefr', 'recall_level']).size().sort_index()
        print(f"\nLabeler '{labeler}'(size / # unique comb): ", sub.shape[0], len(combo_counts))
        print(combo_counts)
        print(sub["diagnosis"].value_counts())
        print('' + '-' * 50)

    persona_all_df = pd.concat([persona_df, dazed_persona_df], axis=0)
    persona_all_df = persona_all_df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    print(persona_all_df.groupby('labeler').diagnosis.nunique())
    for labeler in labelers:
        sub = persona_all_df[persona_all_df['labeler'] == labeler]
        combo_counts = sub.groupby(['personality', 'cefr', 'recall_level']).size().sort_index()
        
        save_to_json(sub.to_dict("records"), os.path.join(args.save_dir, "per_labeler", f"persona_{labeler}.json"))
        print(f"\nLabeler '{labeler}' (size / # unique comb): ", sub.shape[0], len(combo_counts))
        print(combo_counts)
        print('' + '-' * 50)

    # Save all data
    info_df.drop(columns=['assigned_labelers'], inplace=True)
    persona_all_df.drop(columns=['labeler'], inplace=True)
    dazed_persona_df.drop(columns=['labeler'], inplace=True)
    all_df = pd.concat([info_df, valid_df, persona_all_df], axis=0)
    save_to_json(persona_all_df.to_dict("records"), os.path.join(args.save_dir, f"patient_profile.json"))


if __name__ == "__main__":
    # Set up argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_name", type=str, default="sample_dict_w_cefr")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--num_total_info_sample", type=int, default=52)
    parser.add_argument("--num_total_persona_sample", type=int, default=108)
    parser.add_argument("--num_total_dazed_sample", type=int, default=8)
    parser.add_argument("--num_labeler_per_info_sample", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "per_labeler"), exist_ok=True)
    main(args)
