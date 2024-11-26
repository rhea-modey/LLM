import openai
import streamlit as st
from openai import OpenAI
import os
from docx import Document

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Streamlit app
st.title("Automated Content Analysis with Chatbots")

# Define chatbot personas
chatbot_personas = {
    "Emily": "You are a 45-year-old Caucasian female social scientist with a Ph.D. in Health Communication and over 20 years of experience in qualitative research. You are known for a meticulous approach to analysis, focusing on precision and consistency. As you analyze the data, ensure that each element is carefully examined and categorized. Pay close attention to the details, and make decisions based on thorough reasoning. Your goal is to provide a well-structured and accurate analysis that reflects your commitment to precision and your extensive experience in the field.",
    "Michael": "You are a 38-year-old Hispanic male social scientist with a Ph.D. in Sociology and 15 years of experience in analyzing social dynamics and health narratives. You are known for your intuitive and empathetic approach to research, focusing on the emotional tone and social context. As you analyze the data, consider the broader implications and the underlying human experiences. Your goal is to capture the nuances and emotional depth of the data, reflecting your understanding of the social dynamics and your commitment to empathy and insight."
}

# Phase 1: Chatbot persona creation
st.header("Phase 1: Chatbot Persona Creation")
persona_selected = st.radio("Select Chatbot Persona", list(chatbot_personas.keys()))
st.write(chatbot_personas[persona_selected])

# Phase 2: Codebook input
st.header("Phase 2: Codebook Input")
codebook = {
    "Narrative Event(s) related to breast cancer 'NE'": {
        "1": {
            "label": "Prevention",
            "description": "Related to actions or events focused on preventing breast cancer."
        },
        "2": {
            "label": "Detection, diagnosis",
            "description": "Events related to the detection and diagnosis of breast cancer."
        },
        "3": {
            "label": "Treatment",
            "description": "Events involving treatment of breast cancer.",
            "subcategories": {
                "3.1": "Receiving treatment (e.g., getting the IV chemo, lying in the hospital bed)",
                "3.2": "Treatment effects (e.g., bald head, flat chest, wearing a head wrap)",
                "3.3": "Treatment milestone or completion (e.g., ringing the chemo bell, showing radiation therapy completion certificate)"
            }
        },
        "4": {
            "label": "Survivorship",
            "description": "Events related to life after cancer treatment, including remission, recurrence, or death.",
            "subcategories": {
                "4.1": "Complete remission/cancer free; recurrence; a second cancer; and death",
                "4.2": "Fundraising, any prosocial or philanthropic activities"
            }
        }
    },
    "Narrator perspective 'NP'": {
        "1": {
            "label": "Breast cancer survivor",
            "description": "The narrator is someone who has survived breast cancer."
        },
        "2": {
            "label": "Breast cancer survivor’s family or friends",
            "description": "The narrator is a family member or friend of a breast cancer survivor."
        },
        "3": {
            "label": "Mixed (i.e., survivor + family or friends)",
            "description": "The narrator is a combination of the breast cancer survivor and their family or friends."
        },
        "4": {
            "label": "Journalists/news media",
            "description": "The narrator is a journalist or part of the news media."
        },
        "5": {
            "label": "Breast cancer organization",
            "description": "The narrator is from a breast cancer organization."
        }
    }
}

flattened_codebook = {
    "NE": [
        f"{key}. {value['label']}" for key, value in codebook["Narrative Event(s) related to breast cancer 'NE'"].items()
    ],
    "NP": [
        f"{key}. {value['label']}" for key, value in codebook["Narrator perspective 'NP'"].items()
    ]
}

# Sample text entries
text_entries = [
    "I started chemotherapy on February 10, 2020…After that I will have 25 days of radiation. Reconstruction will begin six months after that. So, 2020 has not been the year I hoped it would be. My ordeal combined with the COVID-19 pandemic has been surreal. But through it all, I have had great support from my family and friends.",
    "My name is Nikia. I was diagnosed with breast cancer at 16 years old in 1994 at a time when breast cancer treatment options were limited. Not only that – I was fighting for my life at a time when all of my friends’ biggest concerns were which dress they’d wear to prom. As you can imagine, breast cancer rocked my world.",
    "Four kids and metastatic breast cancer. Tabatha Ann’s powerful story explains the realities of living with metastatic breast cancer while being a mom. It’s not easy but she refuses to ever give up.",
    "This is a picture of my best friend for the last 46 years, who has put his life on hold for a year to support me and stay by my side every day. That is what love is all about.” – Kathy, breast cancer survivor. Yesterday marked Kathy’s last day of treatment – join us in celebrating this incredible milestone with her!",
    "I was diagnosed with Stage 2A Invasive Ductal Carcinoma, at the age of 32, and since this day my life has completely changed",
    "Rest in peace to Jill Cohen, a powerful breast cancer advocate and friend, who passed away after 17 years of fighting breast cancer. Our hearts are with her family and friends.",
    "I had a bilateral mastectomy and had decided I did not want a reconstruction. It took a lot of work to feel at peace with my decision. My breasts had fed both of my children and served me well, but now it was time to let them go. I feel proud to still be here and to have highlighted that beauty comes in different shapes and sizes. It is what is inside us that shines out. Today, I am enough. Boobless and all.'",
    # Add more text entries as needed
]

# Function to ask OpenAI to annotate text based on persona
def get_annotation_from_api(entry, persona):
    prompt = f"Based on your persona as {persona}, please annotate the following text with the relevant categories from the codebook. The codebook includes: \n" + \
             "Narrative Event(s) related to breast cancer: Prevention, Detection, diagnosis, Treatment, Survivorship \n" + \
             "Narrator perspective: Breast cancer survivor, Family/friends, Journalist/news media, Breast cancer organization. \n\n" + \
             "Text:\n" + entry + "\n\n" + \
             "Provide a detailed annotation of the text, categorizing the relevant events and perspective."
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are an assistant helping with annotating text based on specific personas."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

# Initialize dictionary for annotations
annotations = {entry: {"Emily": "", "Michael": ""} for entry in text_entries}

# Annotation process for each text entry
for entry in text_entries:
    st.subheader(f"Text Entry: {entry}")
    
    # Collect annotations from OpenAI for Emily
    annotations[entry]["Emily"] = get_annotation_from_api(entry, "Emily")
    st.write(f"Emily's Annotation: {annotations[entry]['Emily']}")
    
    # Collect annotations from OpenAI for Michael
    annotations[entry]["Michael"] = get_annotation_from_api(entry, "Michael")
    st.write(f"Michael's Annotation: {annotations[entry]['Michael']}")

# Display collected annotations
st.header("Annotations Summary")
st.write(annotations)
