import os
import sys
import cv2
import logging
import click
import shutil

from NAFNet.basicsr.demo import main as deblur
from ShadowNet.main_test import main as shadowRemoval
from DewarpingCV2.DewarpScript import main as dewarping

@click.command()
# @click.argument('input_image', type=click.Path(exists=True))

def main():
    logger = logging.getLogger(__name__)

    input_image = './input/input.jpg'

    interim_path = './output/'
    interim_image = interim_path + 'output.jpg'
    shutil.copy(input_image, interim_image)

    # to do classifier
    classifier = {'deblur': False, 'shrem': False, 'dewarp': True}

    if classifier['deblur']:
        logger.info('Debluring...')
        deblur(interim_image, interim_image, logger)

    if classifier['shrem']:
        logger.info('Shadow removing...')
        shutil.copy(interim_image, interim_path + 'testA/1.png')
        shutil.copy(interim_image, interim_path + 'testC/1.png')
        img = cv2.imread(interim_image)
        dimensions = img.shape

        shadowRemoval(interim_path, interim_path, dimensions[0], dimensions[1], logger)
        shutil.copy(interim_path +'ISTD/600000/outputB/1.png', interim_image)

    if classifier['dewarp']:
        logger.info('Dewarping...')
        dewarping(interim_image, interim_image)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
