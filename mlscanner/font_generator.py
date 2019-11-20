import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FULL_SIZE = 18
TEXT_SIZE = 16
excludedFonts = {"symbol.ttf", "wingding.ttf", "webdings.ttf", "MTEXTRA.TTF", "BSSYM7.TTF"}


def generate_font_image(text, fontname):
    image = Image.new("L", (FULL_SIZE, FULL_SIZE))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(fontname, TEXT_SIZE)
    font_width, font_height = font.getsize(text)
    # print("text", text, "size", font_width, font_height)

    # font_x_offset, font_y_offset = font.getoffset(text)  # <<<< MAGIC!
    # print("text", text, "offset", font_x_offset, font_y_offset)

    draw.rectangle(((0, 0), (FULL_SIZE, FULL_SIZE)), fill="white")
    draw.text(((FULL_SIZE-font_width)/2, (FULL_SIZE-font_height)/2), text, font=font, fill="black")
    return image


def resize_sample_image(im_data):
    # build src image
    (height, width) = im_data.shape
    src_im = Image.new('L', (width, height))
    src_im.putdata(im_data.reshape(-1))
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
    # center into blank image
    image = Image.new("L", (FULL_SIZE, FULL_SIZE), 'white')
    x = int((FULL_SIZE - new_width) / 2)
    y = int((FULL_SIZE - new_height) / 2)
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
                im = generate_font_image(itm, f)
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
    line_image = Image.new("L", ((FULL_SIZE + 1) * len(sample_text), FULL_SIZE))
    for idx, c in enumerate(sample_text):
        char_im = generate_font_image(c, 'OpenSans-Regular.ttf')
        x = int((FULL_SIZE + 1) * idx)
        line_image.paste(char_im, (x, 0))
    line_image.save("../out/generated_sample.png", "PNG")


if __name__ == "__main__":
    main()
