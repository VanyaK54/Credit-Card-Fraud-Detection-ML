from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(yTest, yPred):
    metrics = {
        "accuracy": accuracy_score(yTest, yPred),
        "precision": precision_score(yTest, yPred),
        "recall": recall_score(yTest, yPred),
        "f1": f1_score(yTest, yPred),
        "mcc": matthews_corrcoef(yTest, yPred)
    }
    return metrics

def plot_confusion_matrix(yTest, yPred, save_path):
    conf_matrix = confusion_matrix(yTest, yPred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(save_path)
    plt.close()
