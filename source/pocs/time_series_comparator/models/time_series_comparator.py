import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
import antropy as ant
import pandas as pd
import seaborn as sns

class TimeSeriesComparator:
    def __init__(self, series1, series2, series1_name="Series1", series2_name="Series2"):
        self.s1 = np.array(series1)
        self.s2 = np.array(series2)
        self.name1 = series1_name
        self.name2 = series2_name

    def basic_statistics(self, series):
        return {
            'mean': np.mean(series),
            'std': np.std(series),
            'skewness': skew(series),
            'kurtosis': kurtosis(series),
            'min': np.min(series),
            'max': np.max(series)
        }

    def compare_basic_stats(self):
        stats1 = self.basic_statistics(self.s1)
        stats2 = self.basic_statistics(self.s2)
        return stats1, stats2

    def plot_distributions(self):
        sns.kdeplot(self.s1, label=self.name1)
        sns.kdeplot(self.s2, label=self.name2)
        plt.title("Distribution Comparison")
        plt.legend()
        plt.show()

    def fft_magnitude(self, series):
        fft_vals = np.fft.fft(series)
        mag = np.abs(fft_vals)
        return mag[:len(mag)//2]  # take positive freqs only

    def plot_fft(self):
        mag1 = self.fft_magnitude(self.s1)
        mag2 = self.fft_magnitude(self.s2)
        plt.plot(mag1, label=f'{self.name1} FFT Magnitude')
        plt.plot(mag2, label=f'{self.name2} FFT Magnitude', alpha=0.7)
        plt.title("FFT Magnitude Comparison")
        plt.legend()
        plt.show()

    def autocorrelation(self, series, nlags=100):
        return acf(series, nlags=nlags, fft=True)

    def plot_autocorrelation(self, nlags=100):
        acf1 = self.autocorrelation(self.s1, nlags=nlags)
        acf2 = self.autocorrelation(self.s2, nlags=nlags)
        plt.plot(acf1, label=f'{self.name1} Autocorrelation')
        plt.plot(acf2, label=f'{self.name2} Autocorrelation')
        plt.title("Autocorrelation Comparison")
        plt.legend()
        plt.show()

    def sample_entropy(self, series):
        return ant.sample_entropy(series)

    def compare_entropy(self):
        ent1 = self.sample_entropy(self.s1)
        ent2 = self.sample_entropy(self.s2)
        return ent1, ent2

    def similarity_report(self):
        stats1, stats2 = self.compare_basic_stats()
        ent1, ent2 = self.compare_entropy()

        report = {
            "Basic Statistics": {
                self.name1: stats1,
                self.name2: stats2,
            },
            "Sample Entropy": {
                self.name1: ent1,
                self.name2: ent2
            }
        }
        return report

    def display_similarity_report(self):
        stats1, stats2 = self.compare_basic_stats()
        ent1, ent2 = self.compare_entropy()

        # Format basic stats
        stats_df = pd.DataFrame({
            self.name1: stats1,
            self.name2: stats2
        })

        print("\n📊 Basic Statistics Comparison:")
        print(stats_df.round(4))

        # Format entropy
        entropy_df = pd.DataFrame({
            'Sample Entropy': [ent1, ent2]
        }, index=[self.name1, self.name2])

        print("\n🧠 Sample Entropy Comparison:")
        print(entropy_df.round(4))

    def plot_series(self):
        """
        Plot both time series for visual comparison.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(self.s1, label=self.name1)
        plt.plot(self.s2, label=self.name2, alpha=0.7)
        plt.title("Time Series Comparison")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

