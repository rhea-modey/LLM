import openai
import streamlit as st
from openai import OpenAI
import os

# client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

codebook = {
    "Positive": "Text expressing optimism, agreement, or good feelings.",
    "Negative": "Text expressing pessimism, disagreement, or bad feelings.",
    "Neutral": "Text expressing neither clear positive nor negative sentiment."
}

# Function to display the codebook on Streamlit
def display_codebook():
    st.write("### Codebook for Text Annotations")
    for key, value in codebook.items():
        st.write(f"**{key}**: {value}")

chatbot_1 = {"role": "system", "content": "You are a strict and analytical social scientist who follows the codebook to the letter."}
chatbot_2 = {"role": "system", "content": "You are a flexible and open-minded social scientist who is willing to consider different interpretations of the text."}


def annotate_text(text, persona):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            persona,
            {"role": "system", "content": f"Use this codebook for annotation: {codebook}"},
            {"role": "user", "content": f"Please annotate this text: {text}"}
        ]
    )
    return response.choices[0].message.content

# if st.button("Generate Haiku"):
#         completion = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {
#                     "role": "user",
#                     "content": "Write a haiku about recursion in programming."
#                 }
#             
#         )
        
#         haiku = completion.choices[0].message.content
#         st.write(haiku)

def main():
    st.title("Automating LLMs for Content Analysis")

    display_codebook()

    text_entry = st.text_area("Text for Annotation", "More text!")
    
    if st.button("Annotate Text (Chatbot 1)"):
        annotation = annotate_text(text_entry, chatbot_1)
        st.write(f"Chatbot 1 Annotation: {annotation}")

    if st.button("Annotate Text (Chatbot 2)"):
        annotation = annotate_text(text_entry, chatbot_2)
        st.write(f"Chatbot 2 Annotation: {annotation}")

if __name__ == "__main__":
    main()