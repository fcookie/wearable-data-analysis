from sklearn.metrics import accuracy_score, classification_report

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Classification Report:\n", classification_report(y_true, y_pred))
