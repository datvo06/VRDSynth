import streamlit as st
import requests
from PIL import Image
import io
import zipfile
import os

st.title('PDF Processor')

st.write('Upload a PDF for processing.')

# Endpoint URL of your FastAPI application
API_ENDPOINT = f"{os.environ.get('API_SERVICE_URL', 'http://127.0.0.1:8000')}/visualize_pdf/"

uploaded_file = st.file_uploader("Choose a file")
from_page = st.number_input('From page', min_value=1)
to_page = st.number_input('To page', min_value=1)

if st.button("Run"):
    if uploaded_file:
        files = {'file': uploaded_file.getvalue()}
        payload = {'from_page': from_page, 'to_page': to_page}
        response = requests.post(API_ENDPOINT, files=files, data=payload)

        if response.status_code == 200:
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            members = zip_file.namelist()

            for member in members:
                buf = zip_file.read(member)
                image = Image.open(io.BytesIO(buf))
                st.image(image, caption=f'{member.title()}', use_column_width=True)
        else:
            st.error("An error occurred during processing.")
    else:
        st.error("Missing file!")
