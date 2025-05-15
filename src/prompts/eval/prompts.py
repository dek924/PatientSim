ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

SCORE_RUBRIC_TEMPLATE = """
[{criteria}]
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
""".strip()

PATIENT_PROFILE_TEMPLATE = """
Demographics:
    Age: {age}
    Gender: {gender}
    Race: {race}

Social History:
    Tobacco: {tobacco}
    Alcohol: {alcohol}
    Illicit drug use: {illicit_drug}
    Exercise: {exercise}
    Marital status: {marital_status}
    Children: {children}
    Living Situation: {living_situation}
    Occupation: {occupation}
    Insurance: {insurance}

Previous Medical History:
    Allergies: {allergies}
    Family medical history: {family_medical_history}
    Medical devices used before this ED admission: {medical_device}
    Medical history prior to this ED admission: {medical_history}

Current Visit Information:
    Present illness: 
        positive: {present_illness_positive}
        negative (denied): {present_illness_negative}
    ED chief complaint: {chiefcomplaint}
    Pain level at ED Admission (0 = no pain, 10 = worst pain imaginable): {pain}
    Current medications they are taking: {medication}
    ED Arrival Transport: {arrival_transport}
    ED disposition: {disposition}
    ED Diagnosis: {diagnosis}
"""

PATIENT_PROFILE_TEMPLATE_UTI = """
Demographics:
    Age: {age}
    Gender: {gender}
    Race: {race}

Social History:
    Tobacco: {tobacco}
    Alcohol: {alcohol}
    Illicit drug use: {illicit_drug}
    Sexual History: {sexual_history}
    Exercise: {exercise}
    Marital status: {marital_status}
    Children: {children}
    Living Situation: {living_situation}
    Occupation: {occupation}
    Insurance: {insurance}

Previous Medical History:
    Allergies: {allergies}
    Family medical history: {family_medical_history}
    Medical devices used before this ED admission: {medical_device}
    Medical history prior to this ED admission: {medical_history}

Current Visit Information:
    Present illness: 
        positive: {present_illness_positive}
        negative (denied): {present_illness_negative}
    ED chief complaint: {chiefcomplaint}
    Pain level at ED Admission (0 = no pain, 10 = worst pain imaginable): {pain}
    Current medications they are taking: {medication}
    ED Arrival Transport: {arrival_transport}
    ED disposition: {disposition}
    ED Diagnosis: {diagnosis}
"""

PATIENT_PERSONA_TEMPLATE = """
Persona:
    Personality: {personality}
    Language Proficiency: {cefr}
    Medical History Recall Ability: {memory_recall_level}
    Dazedness level: {dazed_level}
"""