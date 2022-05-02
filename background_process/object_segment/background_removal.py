"""""""""""""""""""""""
Author : NGUYEN DINH HAI
VER    : 1.16
DATE   : 2021, MAR 12
"""""""""""""""""""""""
import os
import errno
import time
import io
import configparser

import numpy as np
from numpy import asarray

from PIL import Image, ImageFilter
# from skimage import transform

import torch

from torchvision import transforms  # , utils

try:
    from .u2net import utils, model
except:
    from u2net import utils, model


# Add padding for image which we need to change
def add_margin(pil_img, top, right, bottom, left, color):
    """

    :param pil_img:
    :param top:
    :param right:
    :param bottom:
    :param left:
    :param color:
    :return:
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def load_model(path, model_name: str = 'u2net'):

    if model_name == "u2netp":
        net = model.U2NETP(3, 1)
    elif model_name == "u2net":
        net = model.U2NET(3, 1)
    else:
        print("Choose between u2net or u2netp")
    print(f"INFO:root: Loaded {model_name}")

    try:
        with open(path, 'rb') as f:
            buffer = io.BytesIO(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT)
            )

    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(buffer))
            net.to(torch.device("cuda"))
        else:
            net.load_state_dict(torch.load(buffer, map_location="cpu"))
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT)
        )

    net.eval()

    return net


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def predict(net, item, show_img=False):

    sample = preprocess(item)

    with torch.no_grad():

        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(sample["image"].unsqueeze(0).float())
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = d1[:, 0, :, :]

        predict = norm_pred(pred)
        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        blur_data = np.copy(predict_np)
        for i in range(blur_data.shape[0]):
            for j in range(blur_data.shape[1]):
                if blur_data[i][j] < 0.45:
                    blur_data[i][j] = 0

        img = Image.fromarray(predict_np * 255).convert("RGB")
        img_blur = Image.fromarray(blur_data * 255).convert("RGB")
        img_blur = img_blur.filter(ImageFilter.GaussianBlur(1))
        # img_blur = img_blur.filter(ImageFilter.BLUR)

        if show_img:
            img.show()
            img_blur.show()

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample

        return img, img_blur
        # return img


# Get object from image
def get_object(img, net, show_img=False, debug=False, auto=False, top_song=False):

    im = Image.open(img)

    pred, pred_blur = predict(net, np.array(im), show_img=show_img)

    _pred = pred.resize(im.size, resample=Image.BILINEAR)
    # _pred.show()
    pred = _pred.copy()

    _blur_mask = pred_blur.resize(im.size, resample=Image.BILINEAR)
    pred_blur = _blur_mask.copy()

    # if debug:
    #     pred.show()

    # Create new blue image
    empty_img = Image.new("RGB", im.size, 0)
    # Create new blue transprent image
    empty_trans = Image.new('RGBA', im.size, (0, 0, 0, 0))

    new = Image.composite(im, empty_img, _pred.convert("L"))

    # Create black img
    black_img = Image.composite(im, empty_img, _blur_mask.convert("L"))

    # Create green image
    _green_img = Image.new('RGB', im.size, 0x00FF00)
    green_img = Image.composite(im, _green_img, _blur_mask.convert('L'))

    # Create transprent image
    transprent_img = Image.composite(im, empty_trans, _blur_mask.convert('L'))
  
    del im, empty_img, new, _blur_mask
 
    return pred, pred_blur, black_img, green_img, transprent_img



if __name__ == "__main__":
    
    img = '/home/haind/Documents/VFAST/Background_removal/dev_env/test.png'
    path = "/home/haind/Documents/VFAST/Background_removal/model/u2net_human_seg.pth"

    net = load_model(path, 'u2net')

    get_object(img=img, net=net, debug=True)
