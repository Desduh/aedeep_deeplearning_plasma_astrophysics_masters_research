from models.time_series_comparator import TimeSeriesComparator
from models.pmodel_generation import generate_and_prepare_series
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load Tokamak data from CSV
tokamak_path = r"C:\Users\Carlos\Documents\INPE\master\aedeep_deeplearning_plasma_astrophysics_masters_research\source\data\jorek\heat_flux_iter.csv"
df_tokamak = pd.read_csv(tokamak_path)

# Extract 'Amplitude' column
tokamak_series_raw = df_tokamak['Amplitude'].values

# Normalize Tokamak data to range [0, 1]
scaler = MinMaxScaler()
tokamak_series = scaler.fit_transform(tokamak_series_raw.reshape(-1, 1)).flatten()

# Define list of p values to test
p_values = np.linspace(0.15, 0.45, 10)  # [0.15, 0.20, ..., 0.45]

# Store results
results = []

for p in p_values:
    print(f"\n=============================")
    print(f"🔍 Evaluating P-model with p = {p:.2f}")

    # Generate P-model series
    pmodel_df = generate_and_prepare_series(length=len(tokamak_series), p_value=p, peak_threshold=0.5)
    pmodel_series = pmodel_df['normalized'].values

    # Compare with Tokamak
    ts_comp = TimeSeriesComparator(tokamak_series, pmodel_series, series1_name="Tokamak", series2_name=f"P-model p={p:.2f}")

    # Generate report
    stats1, stats2 = ts_comp.compare_basic_stats()
    ent1, ent2 = ts_comp.compare_entropy()
    kl_pq, kl_qp = ts_comp.kullback_leibler()  # <<< ADICIONAR ESTA LINHA

    # ts_comp.compare_dfa_psd()
    # ts_comp.display_similarity_report()
    # ts_comp.plot_distributions()
    # ts_comp.plot_fft()
    ts_comp.plot_autocorrelation()

    # Append metrics to results
    results.append({
        'p_value': p,
        'tokamak_mean': stats1['mean'],
        'pmodel_mean': stats2['mean'],
        'tokamak_std': stats1['std'],
        'pmodel_std': stats2['std'],
        'tokamak_skewness': stats1['skewness'],
        'pmodel_skewness': stats2['skewness'],
        'tokamak_kurtosis': stats1['kurtosis'],
        'pmodel_kurtosis': stats2['kurtosis'],
        'tokamak_min': stats1['min'],
        'pmodel_min': stats2['min'],
        'tokamak_max': stats1['max'],
        'pmodel_max': stats2['max'],
        'tokamak_entropy': ent1,
        'pmodel_entropy': ent2,
        'kl_tokamak_pmodel': kl_pq,
        'kl_pmodel_tokamak': kl_qp
    })


# Função para plotar todas as métricas (já existente)
def plot_all_metrics_vs_p(results_df):
    basic_metrics = ['mean', 'std', 'skewness', 'kurtosis']
    entropy_metric = 'entropy'
    kl_metrics = ['kl_tokamak_pmodel', 'kl_pmodel_tokamak']

    # Agora o total de métricas:
    n_metrics = len(basic_metrics) + 1 + 1  # +1 para entropia e +1 para KL divergence
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    plt.figure(figsize=(5*n_cols, 4*n_rows))

    # Plot basic metrics
    for i, metric in enumerate(basic_metrics):
        plt.subplot(n_rows, n_cols, i+1)
        plt.plot(results_df['p_value'], results_df[f'pmodel_{metric}'], marker='o', label='P-model')
        tok_val = results_df[f'tokamak_{metric}'].iloc[0]
        plt.axhline(y=tok_val, color='r', linestyle='--', label='Tokamak')
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xlabel('p value')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

    # Plot sample entropy
    plt.subplot(n_rows, n_cols, len(basic_metrics)+1)
    plt.plot(results_df['p_value'], results_df[f'pmodel_{entropy_metric}'], marker='o', label='P-model')
    tok_val = results_df[f'tokamak_{entropy_metric}'].iloc[0]
    plt.axhline(y=tok_val, color='r', linestyle='--', label='Tokamak')
    plt.title('Sample Entropy Comparison')
    plt.xlabel('p value')
    plt.ylabel('Sample Entropy')
    plt.legend()
    plt.grid(True)

    # Plot KL divergences
    plt.subplot(n_rows, n_cols, len(basic_metrics)+2)
    plt.plot(results_df['p_value'], results_df['kl_tokamak_pmodel'], marker='o', label='KL Tokamak → P-model')
    plt.plot(results_df['p_value'], results_df['kl_pmodel_tokamak'], marker='o', label='KL P-model → Tokamak')
    plt.title('KL Divergence')
    plt.xlabel('p value')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Salvar resultados e plotar
results_df = pd.DataFrame(results)
results_df.to_csv("pmodel_comparison_results.csv", index=False)
plot_all_metrics_vs_p(results_df)
print("\n✅ All comparisons completed. Results saved to 'pmodel_comparison_results.csv'.")
