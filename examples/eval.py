from typing import Any

import chz
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neuroprobe import BrainTreebankDataset, NeuroprobeRunner


class Runner(NeuroprobeRunner):
    model_name = "ExampleModel"
    description = "This is an example model for demonstration purposes."
    author = "Your Name"
    organization = "Your Organization"
    organization_url = "https://www.yourorganization.com"

    @classmethod
    def finetune(
        cls,
        train_ds: BrainTreebankDataset,
        val_ds: BrainTreebankDataset,
    ) -> tuple[StandardScaler, LogisticRegression]:
        model = LogisticRegression()

        x_train = np.array(
            [feature.ieeg.float().numpy() for feature in train_ds]
        )  # shape: (num_samples, num_channels, num_timepoints)
        y_train = np.array(
            [feature.label for feature in train_ds]
        )  # shape: (num_samples,)

        x_train = x_train.reshape(
            x_train.shape[0], -1
        )  # Flatten the features to shape: (num_samples, num_channels * num_timepoints)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        model.fit(x_train, y_train)

        return scaler, model

    @classmethod
    def evaluate(
        cls,
        ctx: tuple[StandardScaler, LogisticRegression],
        test_ds: BrainTreebankDataset,
    ) -> dict[str, Any]:
        scaler, model = ctx
        x_test = np.array(
            [feature.ieeg.float().numpy() for feature in test_ds]
        )  # shape: (num_samples, num_channels, num_timepoints)
        y_test = np.array(
            [feature.label for feature in test_ds]
        )  # shape: (num_samples,)

        x_test = x_test.reshape(
            x_test.shape[0], -1
        )  # Flatten the features to shape: (num_samples, num_channels * num_timepoints)
        x_test = scaler.transform(x_test)

        y_pred = model.predict(x_test)
        accuracy = np.mean(y_pred == y_test)
        return {"accuracy": accuracy}


if __name__ == "__main__":
    chz.nested_entrypoint(Runner.run)
