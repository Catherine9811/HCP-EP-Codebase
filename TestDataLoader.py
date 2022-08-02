import unittest

from DataLoader.BrainDataLoader import BrainDataLoader
from DataLoader.CognitiveDataLoader import CognitiveDataLoader
import sys


class DataLoaderTestCase(unittest.TestCase):
    def test_brain_loader(self):
        loader = BrainDataLoader()
        data = loader.get_processed_data()
        print(data.head())

    def test_cognitive_loader(self):
        loader = CognitiveDataLoader()
        data = loader.get_processed_data()
        print(data.head())


if __name__ == '__main__':
    unittest.main()
