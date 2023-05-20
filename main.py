import os
import sys
import logging
import click
# from dotenv import find_dotenv, load_dotenv

from NAFNet.basicsr.demo import main as deblur


@click.command()
@click.argument('input_image', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
def main(input_image, output_path):
    logger = logging.getLogger(__name__)
    output_path += '\\1.jpg'

    # todo classifier
    classifier = {'deblur': True}
    if classifier['deblur']:
        logger.info('debluring...')
        deblur(input_image, output_path, logger)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load_dotenv(find_dotenv())

    main()
