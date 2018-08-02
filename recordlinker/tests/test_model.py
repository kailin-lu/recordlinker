from unittest import TestCase
import factories

from recordlinker.model import VariationalAutoencoder

class TestVariationalAutoencoder(TestCase):
    def setUp(self):
        self.vae = factories.VarationalAutoencoder()

    def test_variational(self):
        return isinstance(self.vae, VariationalAutoencoder)
