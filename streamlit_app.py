import streamlit as st
import requests
from PIL import Image
import io
import zipfile
import tempfile
import os

st.title('PDF Processor')

st.write('Upload a PDF for processing.')

# Endpoint URL of your FastAPI application
API_ENDPOINT = 'http://localhost:8000/visualize_pdf/'

uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:
    files = {'file': uploaded_file.getvalue()}
    
    # Make a POST request to your FastAPI server with the uploaded file
    response = requests.post(API_ENDPOINT, files=files)
    
    if response.status_code == 200:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_file.extractall(temp_dir)
            # loop through image_* files in the ZIP file in order and display them
            nfiles = len([name for name in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, name))])
            for i in range(nfiles):
                image = Image.open(os.path.join(temp_dir, f'image_{i}.png'))
                st.image(image, caption=f'Image {i}', use_column_width=True)
    else:
        st.write("An error occurred during processing.")
