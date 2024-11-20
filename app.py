import cohere
from gtts import gTTS
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Cohere client with API key from environment variables
API_KEY = os.getenv('jpm5qccrCwtRoGSNY93Ws9iyJLeOgvTgAGbExRLo')
co = cohere.Client(API_KEY) if API_KEY else None

# Streamlit app
st.title('Text-to-Story with Cohere and Streamlit')

# Input field for the user to type a prompt
user_input = st.text_area("Enter a prompt to generate a story:")

# Check if Cohere API is initialized
if not co:
    st.error("Cohere API key is missing! Please configure it properly.")
else:
    # Option to generate a story with Cohere
    if st.button('Generate Story with Cohere'):
        if user_input.strip():
            try:
                response = co.generate(
                    model='command-xlarge-nightly',  # Use appropriate Cohere model
                    prompt=user_input,
                    max_tokens=300,  # Adjust token limit for desired story length
                    temperature=0.8,  # Adjust for creativity
                    k=0,  # Top-k sampling (0 for no restriction)
                    p=0.9,  # Top-p sampling
                    stop_sequences=["\n"],  # Define stop sequences to end the story
                )
                story = response.generations[0].text.strip()
                st.subheader("Generated Story:")
                st.write(story)
                user_input = story  # Use the story for further TTS conversion
            except Exception as e:
                st.error(f"Error generating story: {e}")
        else:
            st.warning("Please enter a prompt to generate a story.")


