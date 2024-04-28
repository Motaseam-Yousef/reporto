import os
import streamlit as st
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import warnings

# Ignore specific warnings from pydub
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", category=RuntimeWarning, module='pydub.utils')

import requests
# URL to the raw image file on GitHub
url = "https://raw.githubusercontent.com/Motaseam-Yousef/reporto/main/reporto.png"

# Fetch the image
response = requests.get(url)

def generate_content(img=None):
    """
    Generates content based on medical examination data presented in an image.
    
    This function configures the GenAI API to analyze medical images and provide an interpretation.
    It aims to provide recommendations with a positive outlook and ensures the advice of a specialist doctor
    is mentioned for detailed information. The response is tailored to be simple and direct in Arabic.
    
    Parameters:
    img (str, optional): An image file that contains medical examination data. Defaults to None.
    
    Returns:
    str: The generated content based on the medical data in the image, or an error message if the process fails.
    """
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=gemini_api_key)
    config = genai.types.GenerationConfig(temperature=0)
    model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
    try:
        genai_client = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
        text_prompt = '''Please provide me with all the details in the image if its related to healthcare field, focusing on the gender, age, all test names, results, and the ranges for these results.
If the image is not related to healthcare, as it is neither an X-ray nor a medical report, then reply only with: "Please provide valid data."'''
        response = model.generate_content([text_prompt, img],generation_config=config)

        # analysis model

        ana_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        text_prompt_ana = f'''if the "{response.text}" as Please provide me by valid data then answer only by -> "Please provide me by valid data" ignore all after.

You are a professional in reading medical reports.
The user will provide you with a report in text form.
You will respond according to these roles:
1. First, you will write a welcome message for the user without using their name or any personal information, Never use any personal information as name or gender or age.
2. Then, you will identify and mention any abnormalities in the report, using the Arabic names for these records. Only mention the abnormal findings; do not include any normal results.
3. As an expert doctor experienced in interpreting reports, provide your conclusion about the user’s health state. If you recommend a doctor’s visit, or suggest special actions like drinking more fluids or avoiding certain foods, include these recommendations.
4. If any findings are abnormal, inform the user that they should visit a doctor.
Your Answer in Arabic Language ONLY.'''
        response_ana = ana_model.generate_content([text_prompt_ana],generation_config=config)

        return [response_ana.text , response.text]
    except Exception as e:
        st.error("Failed to generate content: {}".format(e))
        return None

# Streamlit app setup

import streamlit as st
from PIL import Image
import io
def main():
    if response.status_code == 200:
        # Open the image from the binary content
        im = Image.open(BytesIO(response.content))

        # Display the image using Streamlit
        st.image(im, width=100)
    else:
        st.error("Failed to retrieve image. Status code: {}".format(response.status_code))
    st.title("Reporto")
    st.markdown("##### Skip the Wait, Not the Detail: Fast AI Lab Analysis")
    st.markdown("### Overview")
    st.markdown("""
    In many regions, the manual analysis of lab reports is slow, error-prone, and often hindered by the scarcity of healthcare providers. 
    This project addresses these challenges by introducing an AI-powered application designed to automate and enhance the analysis and interpretation of lab reports, reducing wait times and the anxiety associated with them.
    """)

    img_file_buffer = st.file_uploader("Upload an image (jpg, png):", type=["jpg", "png"])
    img = None
    if img_file_buffer is not None:
        # Convert the file buffer to an image object
        img = Image.open(io.BytesIO(img_file_buffer.getvalue()))

    if st.button("Generate Report"):
        if img:
            # Generate content based on text and image
            processed_text = generate_content(img)
            st.markdown(f"<div style='direction: rtl; text-align: lest;'>{processed_text[0]}</div>", unsafe_allow_html=True)
            st.markdown("--------------------------------------------------------------------------")
            st.markdown(f"<div style='direction: rtl; text-align: right;'>{processed_text[1]}</div>", unsafe_allow_html=True)  # Display the result from generate_content
            # Display the result from generate_content

if __name__ == "__main__":
    main()
