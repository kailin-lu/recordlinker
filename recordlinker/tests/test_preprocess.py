from unittest import TestCase

import recordlinker

class TestCleanNames(TestCase):
    def test_clean_name(self):
        d = recordlinker.preprocess.clean_names()
        pass

class TestEmbedLetters(TestCase):
    # Normalize

    # Not normalized

    # With length

    # Without length
    pass

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