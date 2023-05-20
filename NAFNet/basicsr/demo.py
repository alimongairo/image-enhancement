# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from datetime import datetime
import torch
import cv2
import os

from NAFNet.basicsr.models import create_model
from NAFNet.basicsr.train import parse_options
from NAFNet.basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite


def main(img_path, output_path, logger):
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)

    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    start_time = datetime.now()
    model.test()
    torch.cuda.synchronize()
    print(f'time taken 256x256: {datetime.now() - start_time}')


    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, output_path)

    print(f'inference {img_path} .. finished. saved to {output_path}')


if __name__ == '__main__':
    main()
