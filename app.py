import streamlit as st
import os
from PIL import Image

from pipeline import main as pipeline

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

input_filename = './input/input.jpg'
output_filename = './output/output.jpg'

uploaded_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    with open(os.path.join(input_filename), "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            original_image = Image.open(input_filename)
            st.image(original_image, caption='Original image', width=250)
        with col2:
            with st.spinner('Wait for it...'):
                pipeline()
                inferenced_image = Image.open(output_filename)
                st.image(inferenced_image, caption='Inferenced image', width=250)


