import numpy as np
from PIL import Image

from mlscanner.char_interpreter import CharInterpreter
from mlscanner.font_generator import resize_sample_image, FULL_SIZE
from mlscanner.image_splitter import ImageSplitter
from mlscanner.text_structure import Line, Char


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    X_pad = np.pad(X, ((pad, pad), (pad, pad)), mode='constant', constant_values=(0, 0))

    return X_pad


def conv_single_step(a_slice_prev, conv_filter):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f)
    conv_filter -- Weight parameters contained in a window - matrix of shape (f, f)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (filter) on a slice x of the input data
    """

    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev * conv_filter
    # Sum over all entries of the volume s.
    Z = s.sum()

    return Z


def convolution(im_data, conv_filter):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    im_data -- output activations of the previous layer,
        numpy array of shape (n_H, n_W)
    conv_filter -- Weights, numpy array of shape (f, f)

    Returns:
    Z -- conv output, numpy array of shape (n_H, n_W)
    """

    # Retrieve dimensions from A_prev's shape (≈1 line)
    (n_H, n_W) = im_data.shape

    # Retrieve dimensions from W's shape (≈1 line)
    f = conv_filter.shape[0]

    # pad for "same" padding
    stride = 1
    pad = int((f-1)/2)

    # Initialize the output volume Z with zeros. (≈1 line)
    conv = np.zeros((n_H, n_W))

    # Create A_prev_pad by padding A_prev
    im_data_pad = zero_pad(im_data, pad)

    for h in range(n_H):  # loop over vertical axis of the output volume
        # Find the vertical start and end of the current "slice" (≈2 lines)
        vert_start = h * stride
        vert_end = vert_start + f

        for w in range(n_W):  # loop over horizontal axis of the output volume
            # Find the horizontal start and end of the current "slice" (≈2 lines)
            horiz_start = w * stride
            horiz_end = horiz_start + f

            # Use the corners to define the (2D) slice of a_prev_pad (See Hint above the cell).
            a_slice = im_data_pad[vert_start:vert_end, horiz_start:horiz_end]

            # Convolve the slice with the correct filter
            conv[h, w] = conv_single_step(a_slice, conv_filter)

    return conv


def process_convolution(data, size):
    hori_filter = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    vert_filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    conv_data = convolution(data, hori_filter)
    conv_data = convolution(conv_data, vert_filter)
    conv_im = Image.new('L', size)
    conv_im.putdata(conv_data.reshape(-1))
    conv_im.show()


def process_image(file):
    with Image.open(file) as im:
        im = im.convert('L')
        print(im.format, im.size, im.mode)
        # im.show()
        data = np.array(im)
        print(data.shape)
        # split in lines
        line_splitter = ImageSplitter(data, 1)
        (top, height, line_data) = next(line_splitter.sections_generator())
        line = Line(top, height, line_data)
        # split in chars
        chars = []
        left_average = 0
        char_splitter = ImageSplitter(line_data, 0)
        for (left, width, char_data) in char_splitter.sections_generator():
            chars.append(Char(left, width, char_data))
            left_average += left
        left_average = left_average / len(chars) + 1
        # predict
        interpreter = CharInterpreter()
        line_image = Image.new("L", ((FULL_SIZE+1)*len(chars), FULL_SIZE))
        char_data_array = []
        for idx, char in enumerate(chars):
            char_im = resize_sample_image(char.data)
            x = int((FULL_SIZE+1) * idx)
            line_image.paste(char_im, (x, 0))
            char_data = np.array(char_im, dtype=float).reshape((18, 18, 1))
            char_data_array.append(char_data)
        predictions = interpreter.predict(np.array(char_data_array))
        line_image.save("../out/detected_sample.png", "PNG")
        # interpret
        text = ""
        for idx, char in enumerate(chars):
            if char.left > left_average:
                text += " "
            text += predictions[idx]
        return text
