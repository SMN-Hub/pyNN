import unittest

import mlscanner
from mlscanner.sliding_method import SlidingMethod


class MyTestCase(unittest.TestCase):
    def test_splitchar(self):
        # Generate test image
        sample_text = "In the last video, you learned how to use 125x250 convolutional sliding windows. THAT WAS FUN!"
        nb_fonts = 12
        # generate_test_sample(sample_text, nb_fonts)
        # Scan image
        text = mlscanner.scan_text_from_image('../out/generated_test.png', SlidingMethod.SplitChar, debug=True)
        # Check results
        self.assertIsNotNone(text, 'Scan should output a text')
        self.assertIsNot(text, "", 'Scan should output an empty text')
        lines = text.splitlines()
        self.assertEqual(len(lines), nb_fonts, f"should have {nb_fonts} lines")
        for idx, line in enumerate(lines):
            self.assertEqual(line, sample_text, f"Font #{idx} should be correct")


if __name__ == '__main__':
    unittest.main()
