import unittest

from DataAnalyzer.BoxAnalyzer import BoxAnalyzer
from DataAnalyzer.CorrelationAnalyzer import CorrelationAnalyzer
from DataAnalyzer.DistributionAnalyzer import DistributionAnalyzer
from DataAnalyzer.FeatureAnalyzer import FeatureAnalyzer
from DataAnalyzer.HistogramAnalyzer import HistogramAnalyzer
from DataAnalyzer.OutlierAnalyzer import OutlierAnalyzer
from DataAnalyzer.AbnormalyAnalyzer import AbnormalyAnalyzer
from DataLoader.BrainDataLoader import BrainDataLoader
from DataLoader.CognitiveDataLoader import CognitiveDataLoader


class AnalyzerTestCase(unittest.TestCase):
    def test_corr(self):
        loader = BrainDataLoader()
        data = loader.get_processed_data()
        analyzer = CorrelationAnalyzer(data, loader.get_labels())
        analyzer.analyze()

    def test_tsne(self):
        loader = CognitiveDataLoader()
        analyzer = DistributionAnalyzer(loader.get_features(), loader.get_labels())
        analyzer.analyze()

    def test_cognitive_feature(self):
        loader = CognitiveDataLoader()
        analyzer = FeatureAnalyzer(loader.get_columns(), loader.get_labels())
        analyzer.analyze()
        """
Selected features:
[ True False  True False False False  True False  True  True False False
 False  True  True  True]
Top features:
1. feature 15 (0.169059)
2. feature 9 (0.141011)
3. feature 6 (0.088750)
4. feature 2 (0.068796)
5. feature 5 (0.068059)
6. feature 3 (0.066535)
7. feature 8 (0.055675)
8. feature 1 (0.055214)
9. feature 4 (0.051334)
10. feature 7 (0.045583)
11. feature 11 (0.036732)
12. feature 12 (0.035367)
13. feature 13 (0.033434)
14. feature 14 (0.031787)
15. feature 0 (0.031429)
        """

    def test_brain_feature(self):
        loader = BrainDataLoader()
        analyzer = FeatureAnalyzer(loader.get_columns(), loader.get_labels())
        analyzer.analyze()
        """
Selected features:
[ True  True False  True  True  True False  True  True  True  True  True
  True  True  True False False False  True False False False False  True
 False False False False  True False  True False False False False False
 False False False  True  True  True False  True False False False False
  True False False  True  True  True  True  True False False  True  True
  True  True  True False  True  True False  True  True  True False  True
  True  True  True  True  True False  True  True False False  True  True
 False False  True  True False False  True False False False  True False
 False False False False False False]
Top features:
1. feature 10 (0.049801)
2. feature 8 (0.038202)
3. feature 41 (0.029638)
4. feature 24 (0.024587)
5. feature 23 (0.023225)
6. feature 86 (0.020352)
7. feature 0 (0.020009)
8. feature 7 (0.017552)
9. feature 57 (0.016605)
10. feature 9 (0.016553)
11. feature 3 (0.015744)
12. feature 16 (0.015708)
13. feature 78 (0.015149)
14. feature 42 (0.014788)
15. feature 98 (0.014749)
        """

    def test_brain_outliers(self):
        loader = BrainDataLoader()
        analyzer = OutlierAnalyzer(loader.get_features(), loader.get_labels())
        analyzer.analyze()
        """
174 Normal Points
4 Outliers
2.24%
        """

    def test_cognitive_outliers(self):
        loader = CognitiveDataLoader()
        analyzer = OutlierAnalyzer(loader.get_features(), loader.get_labels())
        analyzer.analyze()
        """
140 Normal Points
18 Outliers
11.39%
        """

    def test_cognitive_abnormaly(self):
        loader = CognitiveDataLoader()
        analyzer = AbnormalyAnalyzer(loader.get_columns(), loader.get_labels())
        analyzer.analyze()

    def test_cognitive_box(self):
        loader = CognitiveDataLoader()
        analyzer = BoxAnalyzer(loader.get_columns(), loader.get_labels())
        analyzer.analyze()

    def test_brain_box(self):
        loader = BrainDataLoader()
        analyzer = BoxAnalyzer(loader.get_columns(), loader.get_labels())
        analyzer.analyze()

    def test_brain_histogram(self):
        loader = BrainDataLoader()
        analyzer = HistogramAnalyzer(loader.get_columns(), loader.get_labels())
        analyzer.analyze()

    def test_cognitive_histogram(self):
        loader = CognitiveDataLoader()
        analyzer = HistogramAnalyzer(loader.get_columns(), loader.get_labels())
        analyzer.analyze()


if __name__ == '__main__':
    unittest.main()
