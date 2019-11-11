import os

from PIL import Image, ImageDraw, ImageFont

FULL_SIZE = 100
textsize = 40
excludedFonts = {"symbol.ttf", "wingding.ttf", "webdings.ttf", "MTEXTRA.TTF", "BSSYM7.TTF"}


def generate_font_image(text, fontname, outputFolder):
    image = Image.new("RGBA", (FULL_SIZE, FULL_SIZE))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(fontname, textsize)
    font_width, font_height = font.getsize(text)

    font_y_offset = font.getoffset(text)[1]  # <<<< MAGIC!

    draw.rectangle(((0, 0), (font_width, font_height)), fill="black")
    draw.text((0, 0 - font_y_offset), text, font=font, fill="red")
    filename = outputFolder + '/' + fontname + '_' + text + '.png'
    image.save(filename, "PNG")


def list_fonts():
    for file in os.listdir('C:\\Windows\\Fonts'):
        if file.endswith(".ttf") and file not in excludedFonts:
            yield file


def main():
    fonts = list_fonts()
    for f in fonts:
        print(f)
    generate_font_image("A", 'arial.ttf', '.')


if __name__ == "__main__":
    main()
