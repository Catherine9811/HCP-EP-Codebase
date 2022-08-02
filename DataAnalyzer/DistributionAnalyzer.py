import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from DataAnalyzer.BaseAnalyzer import BaseAnalyzer


class DistributionAnalyzer(BaseAnalyzer):
    @staticmethod
    def plot_tsne_customized(z_embedded, labels, num_points=4000):
        axes = plt.subplot()
        axes.set_title("T-SNE Visualization Result")
        coords = np.interp(z_embedded, (z_embedded.min(), z_embedded.max()), (0, 1))
        reference = np.random.choice(list(range(len(labels))), num_points)
        for index in reference:
            plt.scatter(coords[index, 0], coords[index, 1], color=plt.cm.Pastel1(labels[index]))

    def analyze(self):
        tsne = TSNE(n_components=2, random_state=1000, init='pca', verbose=0, perplexity=8, n_iter=30000)
        z_embedded = tsne.fit_transform(self.DATA)
        DistributionAnalyzer.plot_tsne_customized(z_embedded, self.LABEL, num_points=200)
        plt.show()
