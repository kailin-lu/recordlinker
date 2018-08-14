from unittest import TestCase

from recordlinker import preprocess

class TestCleanNames(TestCase):
    def test_lower_and_strip(self):
        f = preprocess.lower_and_strip()
        self.assertEqual(f('Kailin!', 'kailin'))

    def test_clean_name(self):
        f = preprocess.clean_names()
        pass

class TestEmbedLetters(TestCase):
    def __init__(self):
        self.max_length = 12

    def test_letters_normalized(self):
        f = preprocess.embed_letters(max_length=self.max_length,
                                     normalize=True)
        self.assertEqual(len(f('kailin'), self.max_length))

    # Not normalized


class TestDisembedLetters(TestCase):
    # Disembed

    # Disembed normalized
    pass

class TestEmbedShingles(TestCase):
    # Embed shingle

    # Embed consecutive shingle
    pass

class TestDisembedShingles(TestCase):
    pass