class Line:
    def __init__(self, top, height, data,  chars, left_average):
        self.top = top
        self.height = height
        self.data = data
        self.chars = chars
        self.left_average = left_average


class Char:
    def __init__(self, left, width, data):
        self.left = left
        self.width = width
        self.data = data


class Paragraph:
    def __init__(self, lines):
        self.lines = lines

    def get_shape(self):
        lines = len(self.lines)
        columns = 0
        for line in self.lines:
            columns = max(columns, len(line.chars))
        return lines, columns

