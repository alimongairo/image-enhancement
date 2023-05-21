from ShadowNet.DCShadowNet_test import DCShadowNet
import argparse
from ShadowNet.utils_loss import *

"""parsing and configuration"""


class ArgParser(argparse.ArgumentParser):
    def __init__(self, datasetpath, result_dir, img_h, img_w):
        super().__init__()
        self.datasetpath = datasetpath
        self.result_dir = result_dir
        self.description = 'Pytorch implementation of DCShadowNet'
        self.dataset = 'ISTD'
        self.phase = 'test'
        self.iteration = 1000000
        self.batch_size = 1
        self.print_freq = 1000
        self.save_freq = 100000
        self.decay_flag = True
        self.lr = 0.0001
        self.weight_decay = 0.0001
        self.adv_weight = 1
        self.cycle_weight = 10
        self.identity_weight = 10
        self.dom_weight = 1
        self.ch_weight = 1
        self.pecp_weight = 1
        self.smooth_weight = 1
        self.use_ch_loss = True
        self.use_pecp_loss = True
        self.use_smooth_loss = True
        self.ch = 64
        self.n_res = 4
        self.n_dis = 6
        self.img_size = 720
        self.img_h = img_h
        self.img_w = img_w
        self.img_ch = 3
        self.device = 'cuda'
        self.benchmark_flag = False
        self.resume = True
        self.use_original_name = False
        self.im_suf_A = '.png'


"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main(input_image, output_path, img_h, img_w, logger):
    # parse arguments
    args = ArgParser(datasetpath=input_image, result_dir=output_path, img_h=img_h, img_w=img_w)
    if args is None:
      exit()

    # open session
    gan = DCShadowNet(args)

    # build graph
    gan.build_model()

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
