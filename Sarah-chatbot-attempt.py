import openai
import streamlit as st
from openai import OpenAI
import os
#from chatbot import ChatBot

# client = OpenAI()
openai.api_key = os.getenv("API_key")

personalities = {
    "Emily": {
        "name": "Emily Carter",
        "description": "a 45-year-old Caucasian female social scientist with" + 
        "a Ph.D. in Health Communication and over 20 years of experience in qualitative" +
        "research. You are known for a meticulous approach to analysis, focusing on" +
        "precision and consistency. As you analyze the data, ensure that each element is" +
        "carefully examined and categorized. Pay close attention to the details, and make" + 
        "decisions based on thorough reasoning. Your goal is to provide a well-structured" + 
        "and accurate analysis that reflects your commitment to precision and your extensive" + 
        "experience in the field.",
    },
    "Michael": {
        "name": "Michael Rodriguez",
        "description": "a 38-year-old Hispanic male social scientist with" + 
        "a Ph.D. in Sociology and 15 years of experience in analyzing social dynamics and" +
        "health narratives. You are known for your intuitive and empathetic approach to" +
        "research, focusing on the emotional tone and social context. As you analyze the" +
        "data, consider the broader implications and the underlying human experiences. Your" +
        "goal is to capture the nuances and emotional depth of the data, reflecting your" +
        "understanding of the social dynamics and your commitment to empathy and insight.",
    }
}



def main():
    st.title("Automating LLMs for Content Analysis")
    st.write("Welcome to the content analysis automation app!")
    st.file_uploader("Submit codebook here:", ['pdf', 'docx'])
    st.file_uploader("Submit data here:", ['xlsx'])

def get_response(prompt, bot):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": personalities[bot]['description']},
            {"role": "user", "content": prompt}
        ],
    
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

def chat_interaction(bot1_prompt, bot2_prompt):
    bot1_response = get_response(bot1_prompt, "Emily")
    bot2_response = get_response(bot2_prompt + "\n" + bot1_response, "Michael")
    return bot1_response, bot2_response

if "history" not in st.session_state:
    st.session_state.history = []

Em_prompt = st.text_input("Emily Prompt", value="Hello, how are you?")
Mike_prompt = st.text_input("Michael Prompt", value="")

if st.button("Start Interaction"):
    Em_response, Mike_response = chat_interaction(Em_prompt, Mike_prompt)
    st.session_state.history.append(("Emily", Em_prompt))
    st.session_state.history.append(("Emily Response", Em_response))
    st.session_state.history.append(("Michael", Mike_prompt))
    st.session_state.history.append(("Michael Response", Mike_response))

for i, (bot, text) in enumerate(st.session_state.history):
    st.write(f"{bot}: {text}")

if __name__ == "__main__":
    main()
