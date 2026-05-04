"""
This file provides an example of how to set up a training and evaluation pipeline using the
`NeuroprobeRunner` class. It trains a simple logistic regression model, returning accuracy
and AUROC metrics on the train and test sets, as expected by the leaderboard.

You can use this as a template for setting up your own training and evaluation pipelines
with different models and metrics.

To run this example, make sure you have the Neuroprobe library installed and the data downloaded,
and the `ROOT_DIR_BRAINTREEBANK` environment variable set to the path of the data directory.

Then, you can run this script with:
```bash
python examples/logistic_regression_runner.py [config overrides]
```

Using the `chz` configuration framework, any overrides of the sort `xxx=yyy`
 will be passed as arguments to the `NeuroprobeConfig` object, overriding the default values.

For example, to override the data directory, you can run:
```bash
python examples/logistic_regression_runner.py data_dir="/path/to/data"
```

All outputs can be found in the `eval_results` folder, ready to be submitted to the leaderboard.
"""

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
