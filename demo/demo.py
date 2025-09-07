import os
import sys
import json
import random
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from run_simulation import PatientAgent

CEFR_DICT = {
    "A": "Beginner\n\tCan make simple sentences.",
    "B": "Intermediate\n\tCan have daily conversations.",
    "C": "Advanced\n\tCan freely use even advanced medical terms.",
}

PERSONALITY_DICT = {
    "plain": "Neutral\n\tNo strong emotions or noticeable behavior.",
    "verbose": "Talkative\n\tSpeaks a lot, and provides highly detailed responses.",
    "distrust": "Distrustful\n\tQuestions the doctor’s expertise and care.",
    "pleasing": "Pleasing\n\tOverly positive and tend to minimize their problems.",
    "impatient": "Impatient\n\tEasily irritated and lacks patience.",
    "overanxious": "Overanxious\n\tExpresses concern beyond what is typical.",
}

RECALL_DICT = {"low": "\n\tOften forgetting even major medical events.", "high": "\n\tUsually recalls their medical events accurately."}

DAZED_DICT = {
    "normal": "\n\tClear mental status, with naturally reflect their own persona.",
    "high": "\n\tHighly dazed and extremely confused, struggle with conversation.",
}

###########################
# Helper Functions
###########################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


@st.cache_data
def load_patient_info(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        patient_info = json.load(f)
    return patient_info


# Styling functions (two variants for demo and NLI pages)
def stylize_text(
    text: str,
    font_size: int = 14,
    color: str = "black",
    bg_color: str = "#E6FFE6",
    border_color: str = "#999999",
    line_height: float = 1.2,
    min_height: str = None,
) -> str:
    text = text.replace("\n", "<br>").replace("\t", "&emsp;")
    style = (
        f"font-size:{font_size}px; color:{color}; background-color:{bg_color}; "
        f"border-radius:10px; padding:10px; "
        f"border:1px solid {border_color}; line-height:{line_height};"
    )
    if min_height:
        style += f" min-height:{min_height};"
    return f"<div style='{style}'>{text}</div>"


# Function to add a message to chat history
def add_message(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})


# Function to render chat history as HTML
def render_chat_html():
    chat_history = st.session_state.get("chat_history", [])
    html_content = "<h3>Consultation History</h3>"
    for msg in chat_history:
        if msg["role"] == "Doctor":
            html_content += f"<p><strong>Doctor:</strong> {msg['content']}</p>"
        elif msg["role"] == "Patient":
            html_content += f"<p><strong>Patient:</strong> {msg['content']}</p>"
        else:
            html_content += f"<p><strong>{msg['role']}:</strong> {msg['content']}</p>"
    return html_content


# Reset the session state keys
def reset_to_patient_selection():
    keys_to_keep = {"logged_in", "labeler"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            st.session_state.pop(key, None)
    st.rerun()  

###########################
# Page Functions
###########################
# 1. Patient selection page
def patient_selection_page():
    st.title("Patient Profile Configuration")
    patient_info_list = load_patient_info("./demo_data.json")
    patient_info_dict = {str(info["hadm_id"]): info for info in patient_info_list}

    display_options_to_hadm = {}  # mapping: "age:..../gender:.../disease:..." → patient info
    for hadm_id in sorted(patient_info_dict.keys()):
        info = patient_info_dict[hadm_id]
        display_str = f"age: {info['age']} / gender: {info['gender']} / disease: {info.get('diagnosis', 'Unknown')}"
        display_options_to_hadm[display_str] = hadm_id

    # Select box to show results
    st.markdown("### Select a Patient")
    selected_display = st.selectbox("Select Patient", display_options_to_hadm.keys())
    base_patient = patient_info_dict[display_options_to_hadm[selected_display]]

    st.markdown("### Customize Patient Persona")

    cefr = st.radio(
        "Select CEFR Level", 
        options=list(CEFR_DICT.keys()), 
        format_func=lambda k: f"{k.capitalize()} : {CEFR_DICT[k]}"
    )

    personality = st.radio(
        "Select Personality", 
        options=list(PERSONALITY_DICT.keys()), 
        format_func=lambda k: f"{k.capitalize()}: {PERSONALITY_DICT[k]}"
    )

    recall_level = st.radio(
        "Select Recall Level", 
        options=list(RECALL_DICT.keys()), 
        format_func=lambda k: f"{k.capitalize()}: {RECALL_DICT[k]}"
    )

    dazed_level = st.radio(
        "Select Dazed Level", 
        options=list(DAZED_DICT.keys()), 
        format_func=lambda k: f"{k.capitalize()}: {DAZED_DICT[k]}"
    )

    if st.button("Start Simulation"):
        base_patient["cefr"] = cefr
        base_patient["personality"] = personality
        base_patient["recall_level"] = recall_level
        base_patient["dazed_level"] = dazed_level

        st.session_state.selected_patient = base_patient
        st.rerun()


# 2. Demo Page (ED Consultation Simulation)
def demo_page():
    st.title("ED Consultation Simulation - Demo")
    selected_patient_profile = st.session_state.get("selected_patient", None)

    if selected_patient_profile is None:
        st.error("No patient selected. Please go back to the patient selection page.")
        if st.button("Go to Patient Selection"):
            reset_to_patient_selection()
        return
    
    # Build a config dictionary based on the selected labeler.
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load patient basic information
    set_seed(config["random_seed"])

    # Initialize session state keys if they do not exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    if "patient_agent" not in st.session_state:
        # Load patient basic info for initialization
        import copy
        selected_patient = copy.deepcopy(selected_patient_profile)
        client_params = {
            "temperature": 0,
            "random_seed": config["random_seed"],
        }
        st.session_state.patient_agent = PatientAgent(
            patient_profile=selected_patient,
            backend_str=config["model"],
            backend_api_type=config["backend_api_type"],
            prompt_dir=config["prompt_dir"],
            prompt_file=config["patient_prompt_file"],
            num_word_sample=10,
            cefr_type=selected_patient["cefr"],
            personality_type=selected_patient["personality"],
            recall_level_type=selected_patient["recall_level"],
            dazed_level_type=selected_patient["dazed_level"],
            client_params=client_params,
        )


    top_cols = st.columns([3, 1])
    with top_cols[-1]:
        if st.button("Return to Patient Selection", use_container_width=True):
            reset_to_patient_selection()

    # Layout: Two columns (left for patient info, right for conversation)
    cols = st.columns([2, 3])
    patient_info_height = st.sidebar.slider("Adjust Patient Info Height", min_value=400, max_value=1000, value=650)
    with cols[0]:
        basic_info_html = f"""
        <div style="margin-bottom: 0px; margin-top: 0px;">
            <h3 style="margin-top: 0px; margin-bottom: 0px;">Patient's Basic Information</h3>
            <strong>Age:</strong> {selected_patient_profile.get('age', 'N/A')}<br>
            <strong>Gender:</strong> {selected_patient_profile.get('gender', 'N/A')}<br>
            <strong>Arrival transport:</strong> {selected_patient_profile.get('arrival_transport', 'N/A')}<br>
        </div>
        """
        persona_html = f"""
        <div style="margin-bottom: 0px; margin-top: 0px; line-height:1.4;">
            <h3 style="margin-top: 0px; margin-bottom: 0px;">Patient's Persona</h3>
            <strong>Language proficiency:</strong> {CEFR_DICT[selected_patient_profile.get('cefr', 'N/A')]}<br>
            <strong>Personality:</strong> {PERSONALITY_DICT[selected_patient_profile.get('personality', 'N/A')]}<br>
            <strong>Medical History Recall Level:</strong> {RECALL_DICT[selected_patient_profile.get('recall_level', 'N/A')]}<br>
            <strong>Dazed Level:</strong> {DAZED_DICT[selected_patient_profile.get('dazed_level', 'N/A')]}
        </div>
        """
        patient_info_html = basic_info_html + persona_html
        patient_info_html = stylize_text(patient_info_html, min_height=f"{patient_info_height-45}px", line_height=1.3)
        components.html(patient_info_html, height=patient_info_height, scrolling=True)

        btn_cols = st.columns(2)
        if not st.session_state.get("chat_saved", False):
            if btn_cols[0].button("Reset Conversation", key="reset_demo"):
                st.session_state.chat_history = []
                st.session_state.patient_agent.reset()
                st.rerun()

    with cols[1]:
        # Conversation area
        conversation_html = render_chat_html()
        conversation_height = patient_info_height - 110
        conversation_html = stylize_text(conversation_html, font_size=14, bg_color="#ebf5fb", min_height=f"{conversation_height-45}px", line_height=1.5)
        components.html(conversation_html, height=conversation_height, scrolling=True)

        # Doctor's message input
        with st.form(key="chat_form", clear_on_submit=True):
            doctor_message = st.text_input("Enter doctor's message:")
            submitted = st.form_submit_button("Submit")

        if submitted and doctor_message:
            if len(st.session_state.chat_history) == 0:
                st.session_state.conversation_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            add_message("Doctor", doctor_message)
            with st.spinner("Patient agent is generating a response..."):
                patient_response = st.session_state.patient_agent.inference(doctor_message)

            add_message("Patient", patient_response)
            st.rerun()



###########################
# Main App Routing
###########################

def main():
    # Show the patient selection page.
    if "selected_patient" not in st.session_state:
        patient_selection_page()
    else:
        demo_page()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
