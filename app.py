import os
import io
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import warnings
import requests
from io import BytesIO
import time
from openai import OpenAI
import base64

# Ignore specific warnings from pydub
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", category=RuntimeWarning, module='pydub.utils')

## Set the API key
def main():
    # Load environment variables
    load_dotenv()
    
    # Streamlit app setup
    st.title("Reporto")
    st.markdown("##### Skip the Wait, Not the Detail: Fast AI Lab Analysis")
    st.markdown("""
    In many regions, the manual analysis of lab reports is slow, error-prone, and often hindered by the scarcity of healthcare providers. 
    This project addresses these challenges by introducing an AI-powered application designed to automate and enhance the analysis and interpretation of lab reports, reducing wait times and the anxiety associated with them.
    """)

    # Azure credentials
    openai_api = os.getenv('OPENAI_API')
    openai_client = OpenAI(api_key=openai_api)
    azure_key = os.getenv('AZURE_VISION_KEY')
    azure_endpoint = os.getenv('AZURE_ENDPOINT')
    credentials = CognitiveServicesCredentials(azure_key)
    client = ComputerVisionClient(azure_endpoint, credentials)

    def extract_text_from_stream(image_stream, client):
        response = client.read_in_stream(image_stream, raw=True)
        operation_location = response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        while True:
            result = client.get_read_result(operation_id)
            if result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        extracted_text = ""
        if result.status == OperationStatusCodes.succeeded:
            for text_result in result.analyze_result.read_results:
                for line in text_result.lines:
                    extracted_text += line.text + "\n"

        return extracted_text.strip()

    def generate_content(image_stream):
        # Configure GenAI API with your API key
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=gemini_api_key)

        try:
            extracted_text = extract_text_from_stream(image_stream, client)

            # Check if any text was extracted
            if not extracted_text:
                return "No text could be extracted from the image."

            # analysis model
            ana_model = genai.GenerativeModel('gemini-1.5-pro-latest')
            config = genai.types.GenerationConfig(temperature=0)
            text_prompt_ana = f'''The report says: "{extracted_text}". 
            and You are a professional in reading medical reports.
            The user will provide you with a report in text form.
            You will respond according to these roles:
            1. First, you will write a welcome message for the user without using their name or any personal information, Never use any personal information as name or gender or age.
            2. Then, you will identify and mention any abnormalities in the report, using the Arabic names for these records. Only mention the abnormal findings; do not include any normal results.
            3. As an expert doctor experienced in interpreting reports, provide your conclusion about the user’s health state. If you recommend a doctor’s visit, or suggest special actions like drinking more fluids or avoiding certain foods, include these recommendations.
            4. If any findings are abnormal, inform the user that they should visit a doctor.
            Your Answer in Arabic Language ONLY.'''

            response_ana = ana_model.generate_content([text_prompt_ana], generation_config=config)
            return [response_ana.text, extracted_text]
        except Exception as e:
            st.error(f"Failed to generate content: {e}")
            return None

    def generate_gpt_content(image_path):
        # Encode the image as a base64 string
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        response = openai_client.chat.completions.create(
        model='gpt-4o',
        messages=[
        {"role": "system", "content": "You are a helpful assistant in healthcare and medical reports and images"},
        {"role": "user", "content": [
            {"type": "text", "text": '''If this image is a medical report please analyze it and provide a detailed explanation of whether the results are good or not. Explain the reasons behind your assessment, including any specific findings such as "1) 2) etc" and provide more detailed results. Additionally, offer recommendations based on your analysis, finally always sure to say "الرجاء التواصل مع طبيب لمعلومات أكثر" Answer ONLY in Arabic.
        else (The given Text not related to any medical information) then reponse "لا يمكنني المساعدة بذلك'''},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    temperature=0.0,
)
        return response.choices[0].message.content

    img_file_buffer = st.file_uploader("Upload an image (jpg, png, jpeg):", type=["jpg", "png", "jpeg"])

    if st.button("Generate Report"):
        if img_file_buffer:
            # Convert the file buffer to an image object
            image_stream = BytesIO(img_file_buffer.getvalue())

            # Save image to a temporary file
            temp_image_path = "temp_image.png"
            with open(temp_image_path, "wb") as temp_image_file:
                temp_image_file.write(img_file_buffer.getvalue())

            # Generate content based on text and image
            processed_text = generate_content(image_stream)
            gpt_response = generate_gpt_content(temp_image_path)

            if processed_text:
                # Display the result from generate_content
                st.markdown("### Doctor Ahmad")
                st.markdown(f"<div style='direction: rtl; text-align: right;'>{processed_text[0]}</div>", unsafe_allow_html=True)
                st.markdown(f"Extracted text: {processed_text[1]}")
                
                # Display the result from GPT response
                st.markdown("### Doctor Mohammad")
                st.markdown(f"<div style='direction: rtl; text-align: right;'>{gpt_response}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
