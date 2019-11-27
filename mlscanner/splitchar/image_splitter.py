import numpy as np


def check_blank(x):
    return x >= 200


class ImageSplitter:
    def __init__(self, data, axis):
        self.data = data
        self.axis = axis
        self.line_mins = np.min(data, axis)
        self.line_count = len(self.line_mins)
        self.top_blank, self.top_fill = 0, 0

    def split(self, line):
        margin_up = 1 if self.top_fill > 0 and self.line_mins[self.top_fill - 1] < 255 else 0
        margin_dn = 1 if line < self.line_count - 1 and self.line_mins[line + 1] < 255 else 0
        top = self.top_fill - self.top_blank
        height = line - self.top_fill
        if self.axis == 1:
            line_data = self.data[(self.top_fill - margin_up):(line + margin_dn + 1), :]
        else:
            line_data = self.data[:, (self.top_fill - margin_up):(line + margin_dn)]
        return top, height, line_data

    def sections_generator(self):
        self.top_blank, self.top_fill = 0, 0

        is_blank = check_blank(self.line_mins[0])
        for line, val in enumerate(self.line_mins):
            new_is_blank = check_blank(val)
            if new_is_blank and not is_blank:  # text => blank
                (top, height, line_data) = self.split(line)
                self.top_blank = line
                yield (top, height, line_data)
            elif not new_is_blank and is_blank:  # blank => text
                self.top_fill = line
            is_blank = new_is_blank
        if not is_blank:
            yield self.split(line)

    def full_section(self):
        self.top_fill, last_fill = -1, -1

        for line, val in enumerate(self.line_mins):
            is_blank = check_blank(val)
            if not is_blank:
                if self.top_fill == -1:
                    self.top_fill = line
                last_fill = line

        return self.split(last_fill)
