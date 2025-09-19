import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Visualizer:
    @staticmethod
    def plot_confusion(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
