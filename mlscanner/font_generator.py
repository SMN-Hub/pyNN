import os
from enum import Enum, auto

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlscanner.image_splitter import ImageSplitter

FULL_SIZE = 18
TEXT_SIZE = 16
excludedFonts = {"symbol.ttf", "wingding.ttf", "webdings.ttf", "MTEXTRA.TTF", "BSSYM7.TTF"}


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
            yield place_font_image(image, place)


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
    font_width, font_height = font.getsize(text)
    _, font_height = font.getsize(text + " lpgf")
    # font_x_offset, font_y_offset = font.getoffset(text)  # <<<< MAGIC!
    draw.text(((image_size-font_width)/2, (image_size-font_height)/2), text, font=font, fill="black")
    if image_size != FULL_SIZE:
        image = image.resize((FULL_SIZE, FULL_SIZE), Image.ANTIALIAS)

    return image


def place_font_image(image, place):
    if place != RePlace.Normal:
        # Alter placement
        im_data = np.array(image)
        splitter = ImageSplitter(im_data, 1)
        (top, height, char_data) = splitter.full_section()
        image = resize_sample_image(char_data, False, place)

    return image


def resize_sample_image(im_data, resize=True, place=RePlace.Center):
    # build src image
    (height, width) = im_data.shape
    src_im = Image.new('L', (width, height))
    src_im.putdata(im_data.reshape(-1))
    if resize:
        # resize with maintain aspect ratio to 16x16
        big_size = max(height, width)
        ratio = big_size / TEXT_SIZE
        new_height = round(height * ratio)
        if new_height == 0:
            new_height = 1
        new_width = round(width * ratio)
        if new_width == 0:
            new_width = 1
        if new_height != height:
            src_im = src_im.resize((new_width, new_height), Image.ANTIALIAS)
            height = new_height
            width = new_width
    # center into blank image
    image = Image.new("L", (FULL_SIZE, FULL_SIZE), 'white')
    x = int((FULL_SIZE - width) / 2)
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


def list_fonts():
    for file in os.listdir('C:\\Windows\\Fonts'):
        if file.endswith(".ttf") and file not in excludedFonts:
            yield file


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


def main():
    fonts = list_fonts()
    for f in fonts:
        print(f)
    # write_font_image("A", 'arial.ttf', '.')
    sample_text = "Inthelastvideo,youlearnedhowtouseconvolutionalimplementationofslidingwindows."
    line_image = Image.new("L", ((FULL_SIZE + 1) * len(sample_text), (FULL_SIZE + 1) * 12))
    for idx, c in enumerate(sample_text):
        for idx2, char_im in enumerate(generate_augmented_font_image(c, 'OpenSans-Regular.ttf')):
            x = int((FULL_SIZE + 1) * idx)
            y = int((FULL_SIZE + 1) * idx2)
            line_image.paste(char_im, (x, y))
    line_image.save("../out/generated_sample.png", "PNG")


if __name__ == "__main__":
    main()
