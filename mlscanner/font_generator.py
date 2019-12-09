import os
import shutil
from enum import Enum, auto

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlscanner.splitchar.image_splitter import ImageSplitter

FULL_SIZE = 18
TEXT_SIZE = 16
excludedFonts = {"symbol.ttf", "wingding.ttf", "webdings.ttf", "MTEXTRA.TTF", "BSSYM7.TTF"}
FONT_FOLDER = '../assets/trainingfonts'


class ReScale(Enum):
    Normal = auto()
    UpDown = auto()
    DownUp = auto()


class RePlace(Enum):
    Normal = auto()
    Center = auto()
    Up = auto()
    Down = auto()


def generate_augmented_font_image(text, fontname):
    for scale in ReScale:
        image = generate_font_image(text, fontname, scale)
        for place in RePlace:
            yield place_font_image(image, True, place)


def get_font_text_size(text, font):
    font_width, _ = font.getsize(text)
    _, font_height = font.getsize(text + " lpgf")  # normalize with 'tall' characters
    return font_width, font_height


def generate_font_image(text, fontname, scale=ReScale.Normal):
    # Draw with scale
    image_size = FULL_SIZE
    text_size = TEXT_SIZE
    if scale == ReScale.UpDown:
        image_size = FULL_SIZE + 4
        text_size = TEXT_SIZE + 4
    elif scale == ReScale.DownUp:
        image_size = FULL_SIZE - 4
        text_size = TEXT_SIZE - 4
    image = Image.new("L", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(fontname, text_size)
    font_width, font_height = get_font_text_size(text, font)
    # font_x_offset, font_y_offset = font.getoffset(text)  # <<<< MAGIC!
    draw.text(((image_size-font_width)/2, (image_size-font_height)/2), text, font=font, fill="black")
    if image_size != FULL_SIZE:
        image = image.resize((FULL_SIZE, FULL_SIZE), Image.ANTIALIAS)

    return image


def place_font_image(image, horiz_center, place):
    if place != RePlace.Normal:
        # Alter placement
        im_data = np.array(image)
        splitter = ImageSplitter(im_data, 1)
        (top, height, char_data) = splitter.full_section()
        image = resize_sample_image(char_data, image.size, horiz_center, place)

    return image


def resize_sample_image(im_data, original_size, horiz_center: True, place=RePlace.Center):
    # build src image
    (height, width) = im_data.shape
    src_im = Image.new('L', (width, height))
    src_im.putdata(im_data.reshape(-1))
    # center into blank image
    image = Image.new("L", original_size, 'white')
    x = int((FULL_SIZE - width) / 2) if horiz_center else 0
    y = 0
    if place == RePlace.Center:
        y = int((FULL_SIZE - height) / 2)
    elif place == RePlace.Up:
        y = 0
    elif place == RePlace.Down:
        y = FULL_SIZE - height
    image.paste(src_im, (x,y))
    return image


def write_font_image(text, fontname, outputFolder):
    image = generate_font_image(text, fontname)
    filename = outputFolder + '/' + fontname + '_' + text + '.png'
    image.save(filename, "PNG")


def list_fonts(from_os=False):
    dir = 'C:/Windows/Fonts' if from_os else FONT_FOLDER
    fonts = []
    for file in os.listdir(dir):
        if file.endswith(".ttf") and file not in excludedFonts:
            fonts.append(dir + '/' + file)
    return fonts


def copy_fonts():
    for file in list_fonts(True):
        shutil.copy2(file, FONT_FOLDER)


def true_array(size, trueIndex):
    ta = np.zeros(size)
    ta[trueIndex] = 1
    return ta


def chars_dataset_generator(features):
    def generator():
        fsize = len(features)
        for f in list_fonts():
            print(f)
            for idx, itm in enumerate(features):
                for im in generate_augmented_font_image(itm, f):
                    x = np.array(im).reshape((18,18,1))
                    y = true_array(fsize, idx)
                    yield (x, y)
    return generator


def generate_data_augmented_sample():
    sample_text = "Inthelastvideo,youlearnedhowtouseconvolutionalimplementationofslidingwindows."
    line_image = Image.new("L", ((FULL_SIZE + 1) * len(sample_text), (FULL_SIZE + 1) * 12))
    for idx, c in enumerate(sample_text):
        for idx2, char_im in enumerate(generate_augmented_font_image(c, "../assets/OpenSans-Regular.ttf")):
            x = int((FULL_SIZE + 1) * idx)
            y = int((FULL_SIZE + 1) * idx2)
            line_image.paste(char_im, (x, y))
    line_image.save("../out/generated_sample.png", "PNG")


def generate_test_sample(sample_text, max_font=None):
    text_size = 14
    fonts = list_fonts()
    if max_font is not None:
        fonts = fonts[:max_font]
    image = Image.new("L", ((FULL_SIZE + 1) * len(sample_text), (FULL_SIZE + 1) * len(fonts)), "white")
    draw = ImageDraw.Draw(image)
    for idx, fontname in enumerate(fonts):
        font = ImageFont.truetype(fontname, text_size)
        font_width, font_height = font.getsize(sample_text)
        y = int((FULL_SIZE + 1) * idx)
        draw.text((0, y + (FULL_SIZE - font_height) / 2), sample_text, font=font, fill="black")
        print(fontname)
    image.save("../out/generated_test.png", "PNG")


def main():
    copy_fonts()
    # sample_text = "In the last video, you learned how to use 125x250 convolutional sliding windows. THAT WAS FUN!"
    # generate_test_sample(sample_text, 13)


if __name__ == "__main__":
    main()
