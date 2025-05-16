#!/bin/bash

export GOOGLE_APPLICATION_CREDENTIALS=../google_credentials.json
export GOOGLE_PROJECT_ID="YOUR_PROJECT_ID"

# Capture the start time
start_time=$(date +%s)

# Read username and password (for PhysioNet)
read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD

# Define directories and file names
MIMIC_IV="https://physionet.org/files/mimiciv/3.1"
MIMIC_ED="https://physionet.org/files/mimic-iv-ed/2.2"
MIMIC_NOTE="https://physionet.org/files/mimic-iv-note/2.2"

# Define wget parameters for readability
WGET_PARAMS="-r -N -c -np --user $USERNAME --password $PASSWORD"

# Helper function to download and extract files
download_and_extract() {
    local file_url=$1
    local destination_dir=$2
    local file_name=$(basename "$file_url")

    # Download the file
    wget $WGET_PARAMS "$file_url"

    # Extract if it's a zip file
    if [[ "$file_name" == *.zip ]]; then
        unzip -o "$destination_dir/$file_name" -d "$destination_dir" # -o: overwrite
    fi

    # Extract if it's a gzip file
    if [[ "$file_name" == *.gz ]]; then
        gzip -d "$destination_dir/$file_name"
    fi
}

# Download MIMIC-IV hosp modules
download_and_extract "$MIMIC_IV/hosp/admissions.csv.gz" "data/physionet.org/files/mimiciv/3.1/hosp"
download_and_extract "$MIMIC_IV/hosp/patients.csv.gz" "data/physionet.org/files/mimiciv/3.1/hosp"

# Download MIMIC-IV-ED modules
download_and_extract "$MIMIC_ED/ed/diagnosis.csv.gz" "data/physionet.org/files/mimic-iv-ed/2.2/ed"
download_and_extract "$MIMIC_ED/ed/edstays.csv.gz" "data/physionet.org/files/mimic-iv-ed/2.2/ed"
download_and_extract "$MIMIC_ED/ed/medrecon.csv.gz" "data/physionet.org/files/mimic-iv-ed/2.2/ed"
download_and_extract "$MIMIC_ED/ed/triage.csv.gz" "data/physionet.org/files/mimic-iv-ed/2.2/ed"

# Download MIMIC-IV-NOTE icu modules
download_and_extract "$MIMIC_NOTE/note/discharge.csv.gz" "data/physionet.org/files/mimic-iv-note/2.2/note"

# Dowonload kaggle cefr words data
mkdir -p "data/CEFR_kaggle"
curl -L -o "data/10-000-english-words-cerf-labelled.zip" "https://www.kaggle.com/api/v1/datasets/download/nezahatkk/10-000-english-words-cerf-labelled"
unzip -o "data/10-000-english-words-cerf-labelled.zip" -d "data/CEFR_kaggle"

# Save currentdirectory
orig_dir=$(pwd)

# Preprocessing and generate dataset
SAVE_DIR="data/preprocessed_data"
mkdir -p "$SAVE_DIR"

python "data_preprocessing/note_preprocessing.py" \
    --note_dir "data/physionet.org/files/mimic-iv-note/2.2/note" \
    --save_dir "$SAVE_DIR"

python "data_preprocessing/sample_patient_profile.py" \
    --mimic_dir "data/physionet.org/files/mimiciv/3.1" \
    --ed_dir "data/physionet.org/files/mimic-iv-ed/2.2/ed" \
    --preprocess_dir "$SAVE_DIR" \
    --save_dir "$SAVE_DIR" 

python "data_preprocessing/key_extraction.py" \
    --data_dir "$SAVE_DIR" \
    --save_dir "$SAVE_DIR" \
    --exp_name "profile_extraction" \
    --prompt_dir "prompts/data_preprocessing/key_extraction" \
    --debug 

python "data_preprocessing/data_filtering.py" \
    --data_dir "$SAVE_DIR" \
    --key_dir "$SAVE_DIR/profile_extraction" \
    --prompt_dir "prompts/data_preprocessing/data_filtering" 
    

python "data_preprocessing/key_modification.py" \
    --data_dir "$SAVE_DIR" \
    --key_dir "$SAVE_DIR/profile_extraction" \
    --prompt_dir "prompts/data_preprocessing/key_modification" 

cp "$SAVE_DIR/profile_extraction/gemini-2.5-flash-preview-04-17_mod_results.json" "$SAVE_DIR/sample_dict.json"

python "data_preprocessing/mapping_CEFR_words.py" \
    --data_dir "$SAVE_DIR" \


SAVE_DIR="data/preprocessed_data"
FINAL_DATA_SAVEDIR="data/final_data"
mkdir -p "$FINAL_DATA_SAVEDIR"
python "data_preprocessing/mapping_persona.py" \
    --data_dir "$SAVE_DIR" \
    --save_dir "$FINAL_DATA_SAVEDIR" \

# Capture the end time
end_time=$(date +%s)

# Calculate the runtime
runtime=$((end_time - start_time))

# Display the runtime
echo "Script runtime: $runtime seconds"