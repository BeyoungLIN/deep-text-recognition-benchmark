import random
from PIL import Image, ImageFilter, ImageFont, ImageDraw
import numpy as np
from torch import nn
from torchvision import transforms


def salt_and_pepper_noise(img, proportion=0.05):
    noise_img = np.array(img)
    height, width = noise_img.shape
    num = int(height * width * proportion)  # how many points to be salt and peper noise point
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    noise_img = Image.fromarray(noise_img)
    return noise_img


def gauss_noise(image):
    image = np.array(image)
    img = image.astype(np.int16)  # 此步是为了避免像素点小于0，大于255的情况
    mu = 0
    sigma = 10
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j] + random.gauss(mu=mu, sigma=sigma)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img


def img_augment(img, crop_threshold=0.3, blur_threshold=0.3, salt_pepper_threshold=0.5, gauss_threshold=0.5):
    w, h = img.size
    if random.random() < crop_threshold:
        multiplier = random.uniform(1.0, 1.2)
        # add an eps to prevent cropping issue
        nw = int(multiplier * w) + 1
        nh = int(multiplier * h) + 1
        img = img.resize((nw, nh), Image.BICUBIC)

        shift_x = random.randint(0, max(nw - w - 1, 0))
        shift_y = random.randint(0, max(nh - h - 1, 0))
        img = img.crop((shift_x, shift_y, shift_x + w, shift_y + h))
    if random.random() < blur_threshold:
        sigma_list = [1, 1.5, 2]
        sigma = random.choice(sigma_list)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    if random.random() < salt_pepper_threshold:
        img = salt_and_pepper_noise(img)
    if random.random() < gauss_threshold:
        img = gauss_noise(img)
    return img


def draw_single_char(ch, font, canvas_size, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
    # 需要填充区域，如果宽大于高则上下填充，否则左右填充
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    # 填充像素常值
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #直接填0
    img = img.squeeze(0)  # 去轴
    img = transforms.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img
