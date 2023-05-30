import streamlit as st
import os
from PIL import Image
import cv2
import logging
import shutil
from datetime import datetime
import torch

from NAFNet.basicsr.demo import main as deblur
from ShadowNet.main_test import main as shadowRemoval
from DewarpingCV2.DewarpScript import main as dewarping

def inference(classifier):
    logger = logging.getLogger(__name__)

    input_image = './input/input.jpg'
    interim_path = './output/'
    interim_image = interim_path + 'output.jpg'
    shutil.copy(input_image, interim_image)

    total_start_time = datetime.now()
    deblur_time, shrem_time, dewarp_time = 0, 0, 0

    if classifier['deblur']:
        start_time = datetime.now()
        logger.info('Debluring...')
        deblur(interim_image, interim_image, logger)
        deblur_time = datetime.now() - start_time
        torch.cuda.empty_cache()

    if classifier['shrem']:
        start_time = datetime.now()
        logger.info('Shadow removing...')
        shutil.copy(interim_image, interim_path + 'testA/1.png')
        shutil.copy(interim_image, interim_path + 'testC/1.png')
        img = cv2.imread(interim_image)
        dimensions = img.shape
        shadowRemoval(interim_path, interim_path, dimensions[0], dimensions[1], logger)
        shutil.copy(interim_path +'ISTD/600000/outputB/1.png', interim_image)
        shrem_time = datetime.now() - start_time
        torch.cuda.empty_cache()

    if classifier['dewarp']:
        start_time = datetime.now()
        logger.info('Dewarping...')
        dewarping(interim_image, interim_image)
        dewarp_time = datetime.now() - start_time

    total_time = datetime.now() - total_start_time

    return total_time, deblur_time, shrem_time, dewarp_time

def app():
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

    classifier = {'deblur': False, 'shrem': False, 'dewarp': False}

    torch.cuda.empty_cache()
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader('Upload An Image', type=['png', 'jpeg', 'jpg'])
        with col2:
            st.markdown('Settings')
            classifier['deblur'] = st.checkbox('Deblur')
            classifier['shrem'] = st.checkbox('Remove shadows')
            classifier['dewarp'] = st.checkbox('Dewarp')

    if uploaded_file is not None:
        with open(os.path.join(input_filename), "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                original_image = Image.open(input_filename)
                st.image(original_image, caption='Original image', width=250)
            with col2:
                with st.spinner('Wait for it...'):
                    total_time, deblur_time, shrem_time, dewarp_time = inference(classifier)
                inferred_image = Image.open(output_filename)
                st.image(inferred_image, caption='Inferred image', width=250)

        st.subheader(f'Total time: {total_time} sec')
        st.markdown('')
        if deblur_time:
            st.markdown(f'Deblurring time: {deblur_time} sec')
        if shrem_time:
            st.markdown(f'Shadow remvoal time: {shrem_time} sec')
        if dewarp_time:
            st.markdown(f'Dewarping time: {dewarp_time} sec')

if __name__ == '__main__':
    app()
