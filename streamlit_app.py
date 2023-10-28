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
API_ENDPOINT = 'http://localhost:9000/visualize_pdf/'

uploaded_file = st.file_uploader("Choose a file")
from_page = st.number_input('From page', min_value=1)
to_page = st.number_input('To page', min_value=1)

if uploaded_file is not None:
    files = {'file': uploaded_file.getvalue()}
    data = {'from_page': from_page, 'to_page': to_page}
    
    response = requests.post(API_ENDPOINT, files=files)
    
    if response.status_code == 200:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_file.extractall(temp_dir)
            nfiles = len([name for name in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, name))])
            for i in range(nfiles):
                image = Image.open(os.path.join(temp_dir, f'image_{i}.png'))
                st.image(image, caption=f'Image {i}', use_column_width=True)
    else:
        st.write("An error occurred during processing.")
