import openai
import streamlit as st
from openai import OpenAI
import os

# client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

if st.button("Generate Haiku"):
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a haiku about recursion in programming."
                }
            ]
        )
        
        haiku = completion.choices[0].message.content
        st.write(haiku)

def main():
    st.title("Automating LLMs for Content Analysis")
    st.write("Welcome to the content analysis automation app!")
    st.file_uploader("Submit codebook here:", ['pdf', 'docx']) # adds widget to upload the codebook
        st.file_uploader("Submit data here:", ['xlsx']) #widget for uploading data

if __name__ == "__main__":
    main()
