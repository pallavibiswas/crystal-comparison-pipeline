import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_true = np.loadtxt('data/y_all.dat').astype(int)
y_pred = np.loadtxt('model/y_pred_all.dat').astype(int)

split_point = int(0.8 * len(y_true))
y_true = y_true[split_point:]

label_names = [0, 1, 2, 3, -1]

def evaluate_model(y_true, y_pred, name):
    print(f"\nEvaluation Report for {name}")
    accuracy = accuracy_score(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, labels=label_names)
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=label_names)

    with open(f'evaluation/{name.lower()}_eval.txt', 'w') as f:
        f.write(f"\nEvaluation Report for {name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report: \n")
        f.write(clf_report + "\n")
        f.write("Confusion Matrix: ")
        f.write(np.array2string(cnf_matrix, separator=', ')+ "\n")

    plt.figure(figsize=(6,5))
    sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=label_names, yticklabels=label_names)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"evaluation/{name.lower()}_cm.png")
    plt.close()

evaluate_model(y_true, y_pred, "DC3")


