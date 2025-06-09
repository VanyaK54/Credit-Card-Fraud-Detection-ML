import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(data, save_path):
    corrmat = data.corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.8, square=True)
    plt.title("Correlation Matrix")
    plt.savefig(save_path)
    plt.close()
