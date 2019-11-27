from enum import Enum, auto


class SlidingMethod(Enum):
    """
    Method used to identify characters on line
    """
    SplitChar = auto()  # hand split characters (faster but un-precised with some fonts)
    ConvSlide = auto()  # convolutional horizontal anchor box sliding (more precise but slower - and very more difficult to train))
