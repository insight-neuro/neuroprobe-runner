from typing import Any

import chz
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from neuroprobe import BrainTreebankDataset, NeuroprobeRunner


def calculate_metrics(
    clf, x_test: np.ndarray, y_test: np.ndarray, prefix: str = ""
) -> dict[str, Any]:
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)[:, 1]
    accuracy = np.mean(y_pred == y_test)
    auroc = roc_auc_score(y_test, y_score)
    return {f"{prefix}_accuracy": accuracy, f"{prefix}_auroc": auroc}


class Runner(NeuroprobeRunner):
    model_name = "ExampleModel"
    description = "This is an example model for demonstration purposes."
    author = "Your Name"
    organization = "Your Organization"
    organization_url = "https://www.yourorganization.com"

    @classmethod
    def evaluate_fold(
        cls,
        train_ds: BrainTreebankDataset,
        val_ds: BrainTreebankDataset,
        test_ds: BrainTreebankDataset,
    ) -> dict[str, Any]:
        model = LogisticRegression()

        X_train = train_ds.signals  # shape: (num_samples, num_channels, num_timepoints)
        y_train = train_ds.labels  # shape: (num_samples,)

        # Flatten the features to shape: (num_samples, num_channels * num_timepoints)
        X_train = X_train.reshape(X_train.shape[0], -1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        model.fit(X_train, y_train)
        train_metrics = calculate_metrics(model, X_train, y_train, "train")

        X_test = test_ds.signals.reshape(test_ds.signals.shape[0], -1)
        X_test = scaler.transform(X_test)
        y_test = test_ds.labels

        test_metrics = calculate_metrics(model, X_test, y_test, "test")

        return {**train_metrics, **test_metrics}


if __name__ == "__main__":
    chz.nested_entrypoint(Runner.run)
