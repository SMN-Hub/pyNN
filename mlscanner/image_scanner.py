"""
Image scanner reads formatted text from image

Steps:
* Resize image to reasonable size
* Turn to gray scale
* Contrast enhance
  * Detect font color(s) vs background color(s) (background should should be numerous)
  * Extreme contrast: turn into bitmap (font black, background white)
* Convolution for paragraph detection
* Fix rotation problems
  * pitch (slightly rotate) image
  * redo paragraph detection         => repeat until paragraph size is smaller
  * rotate source image too
* Detect / remove Table lines
  * each cell is considered as a separate paragraph
* For each paragraph
  * Detect lines
    - top spacing (full blank lines)
    - text height
    - line floor position (most black line)
  * Detect Chars
    - left margin
    - text with (! care about attached fonts)
* CNN for Char interpretation
  * Resize/scale Chars at CNN input size
  * Evaluate Chars
  * Resolve ambiguities (1 vs l, 0 vs O, ...) according to context
  * Build words with white spaces according to left margins
* CNN for font name detection
* Font colors from original image
"""
from mlscanner.image_processor import process_image


def scan_text_from_image(file, debug=False):
    """
    Scan an image to determined formatted text inside
    :param debug: if true, produces an output debug image in 'out/detected_debug.png'
    :param file: the image file to scan
    :return: the scanned text
    """
    text = process_image(file, debug)
    print(text)
    return text


if __name__ == "__main__":
    scan_text_from_image("..\\assets\\text.png")
