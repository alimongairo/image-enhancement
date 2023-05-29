import streamlit as st
import os
from PIL import Image
import time

from main import main as pipeline

st.markdown(
    """
    <style>
        div[data-testid="column"]
        {
            display: flex;
            justify-content: center;
            align-items: center;
        } 
    </style>
    """,unsafe_allow_html=True
)

st.title('Image Enhancement Pipeline')

input_dir = './input/'

uploaded_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    with open(os.path.join(input_dir, 'input.jpg'), "wb") as f:
        f.write(uploaded_file.getbuffer())

    original_image = Image.open(input_dir + 'input.jpg')
    inferenced_image = Image.open(input_dir + 'input.jpg')

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption='Original image', width=250)
        with col2:
            with st.spinner('Wait for it...'):
                pipeline(input_image=input_dir + 'input.jpg')
            st.image(inferenced_image, caption='Inferenced image', width=250)


