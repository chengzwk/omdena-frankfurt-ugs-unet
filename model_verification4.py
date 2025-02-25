# Perform various experiments to verify the model's correctness

# Experiment 3: Constant-output model
# Make a constant-output model that always predicts 1 or 0 and evaluate its performance metrics

import numpy as np
from model_evaluation_and_prediction import evaluate_model, compute_iou, plot_roc_curve


class AlwaysPredictor:
    """A simple model that always predicts a fixed value."""

    def __init__(self, constant_value=0, alpha=0.25, gamma=2.0):
        self.constant_value = constant_value
        self.alpha = alpha
        self.gamma = gamma

    def predict(self, X_test):
        """Returns a constant prediction for all inputs."""
        return np.full((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1), self.constant_value, dtype=np.float32)

    def binary_focal_loss(self, y_true, y_pred):
        """Computes binary focal loss manually."""
        epsilon = 1e-7  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)  # Ensure numerical stability

        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)  # Get p_t for true class
        alpha_factor = np.where(y_true == 1, self.alpha, 1 - self.alpha)  # Apply alpha
        modulating_factor = (1.0 - p_t) ** self.gamma  # Apply gamma

        focal_loss = -alpha_factor * modulating_factor * np.log(p_t)
        return np.mean(focal_loss)  # Average over all samples

    def evaluate(self, X_test, y_test, verbose=0):
        """Returns accuracy, precision, recall, and loss based on the constant predictions."""
        y_pred = self.predict(X_test)
        y_pred_thresholded = (y_pred >= 0.5).astype(int)

        # Compute basic metrics
        accuracy = np.mean(y_pred_thresholded == y_test)
        precision = np.sum((y_pred_thresholded == 1) & (y_test == 1)) / max(np.sum(y_pred_thresholded == 1), 1)
        recall = np.sum((y_pred_thresholded == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
        loss = self.binary_focal_loss(y_test, y_pred)  # Compute focal loss

        return loss, accuracy, precision, recall


def evaluate_baseline(X_test, y_test):
    """Evaluates both the Always 0 and Always 1 models."""
    print("\nEvaluating Always 0 Model")
    print("-" * 30)
    model_zero = AlwaysPredictor(0)
    y_pred_zeros = model_zero.predict(X_test)
    y_pred_thresholded_zeros = (y_pred_zeros >= 0.5).astype(int)

    evaluate_model(model_zero, X_test, y_test)
    compute_iou(y_pred_thresholded_zeros, y_test)
    plot_roc_curve(y_pred_thresholded_zeros, y_test)

    print("\nEvaluating Always 1 Model")
    print("-" * 30)
    model_one = AlwaysPredictor(1)
    y_pred_ones = model_one.predict(X_test)
    y_pred_thresholded_ones = (y_pred_ones >= 0.5).astype(int)

    evaluate_model(model_one, X_test, y_test)
    compute_iou(y_pred_thresholded_ones, y_test)
    plot_roc_curve(y_pred_thresholded_ones, y_test)


# Example usage
if __name__ == "__main__":
    # Replace with actual test data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    evaluate_baseline(X_test, y_test)


# Result: established performance metrics for constant-output models