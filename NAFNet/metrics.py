import cv2
from ssim import ssim as SSIM
from SSIM_PIL import compare_ssim
from PIL import Image
import os


def main():
    blur_dir = '../NAFNet/datasets/test'
    sharp_dir = '../NAFNet/output'
    blur_paths = []
    sharp_paths = []
    psnr_values = []
    ssim_values = []

    for dir in os.listdir(blur_dir):
        for img in os.listdir(blur_dir + '/' + dir + '/blur'):
            blur_paths.append(blur_dir + '/' + dir + '/blur' + '/' + img)
            sharp_paths.append(sharp_dir + '/' + dir + '/' + img)

    for i, blur_path in enumerate(blur_paths):
        sharp_path = sharp_paths[i]

        original = cv2.imread(blur_path)
        compressed = cv2.imread(sharp_path)
        value1 = cv2.PSNR(original, compressed)
        psnr_values.append(value1)

        original = Image.open(blur_path)
        compressed = Image.open(sharp_path)
        value2 = compare_ssim(original, compressed)
        ssim_values.append(value2)

    psnr_val = sum(psnr_values) / len(psnr_values)
    ssim_val = sum(ssim_values) / len(ssim_values)

    print(f"PSNR value is {psnr_val} dB")
    print(f"SSIM value is {ssim_val}")


if __name__ == "__main__":
    main()
