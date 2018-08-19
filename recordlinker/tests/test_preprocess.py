from unittest import TestCase

from recordlinker import preprocess
import numpy as np

class TestEmbedLetters(TestCase):
    def test_embed_letters(self):
        embedded = preprocess.embed_letters('abcdefghijklmnopqrstuvwxyz ',
                                            max_length=27,
                                            normalize=False)
        self.assertEqual(len(embedded), 27)

    def test_embed_letters_normalized(self):
        embedded =  preprocess.embed_letters('abcdefghijklmnopqrstuvwxyz ',
                                             max_length=27,
                                             normalize=True)
        self.assertTrue(np.max(embedded) <= 1.0)

    def test_disembed_letters(self):
        reconstructed = preprocess.disembed_letters(np.array([1,2,3,4,5,6,7,8,9,10,
                                                              11,12,13,14,15,16,17,18,19,20,
                                                              21,22,23,24,25,26,27]))
        self.assertTrue(reconstructed, 'abcdefghijklmnopqrstuvwxyz ')

    def test_disembed_letters_normalized(self):
        # reconstructed = preprocess.disembed_letters()
        pass

    def test_disembed_letters_onehot(self):
        # reconstructed = preprocess.disembed_letters(onehot=True)
        pass


class TestEmbedShingles(TestCase):
    def test_embed_shingles(self):
        pass

    def test_embed_shingles_normalized(self):
        pass

    def test_disembed_shingles(self):
        pass

    def test_disembed_shingles_normalized(self):
        pass

    def test_disembed_shingles_onehot(self):
        pass


class TestCleanNames(TestCase):
    def test_lower_and_strip(self):
        lowered = preprocess.lower_and_strip('KAILIN')
        no_whitespace = preprocess.lower_and_strip('kailin l')
        no_punc = preprocess.lower_and_strip("kai'lin.lu!")

        self.assertEqual(lowered, 'kailin')
        self.assertTrue(no_whitespace, 'kailinl')
        self.assertTrue(no_punc, 'kailinlu')

