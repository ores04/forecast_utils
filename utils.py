import pandas as pd
import numpy as np
import jax.numpy as jnp

def train_test_split(data: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets."""
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def transform_data_to_training_timeseries_np(data: pd.DataFrame, window_size: int = 100) -> tuple[np.array, np.array]:
    """Transform the DataFrame into a time series format for training."""
    # Convert DataFrame to JAX array
    y = np.array(data['Target'])
    x_unformatted = np.array(data.drop(columns=['Target']))
    x = np.array([x_unformatted[i:i + window_size] for i in range(len(x_unformatted) - window_size)])
    y = y[window_size:] # drop label for series that are not complete
    return x, y

def transform_data_to_training_timeseries_jax(data: pd.DataFrame, window_size: int = 100, columns_to_drop=['Target']) -> \
    tuple[jnp.ndarray, jnp.ndarray]:
        """Transform the DataFrame into a time series format for training."""
        # Convert DataFrame to JAX array
        y = jnp.array(data['Target'])
        x_unformatted = jnp.array(data.drop(columns=columns_to_drop))
        x = []
        for i, row in enumerate(x_unformatted):
            if i < window_size:
                continue  # Skip rows that are too short

            # Create a sliding window of the data
            x.append(x_unformatted[i - window_size:i])

        x = jnp.array(x)  # Convert list of arrays to a JAX array
        y = y[window_size - 1:-1]  # Adjust y to match the length of x

        return x, y


import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin


class RobustScaler05(BaseEstimator, TransformerMixin):
    def __init__(self, quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True):
        self.quantile_range = quantile_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.scaler = RobustScaler(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling
        )

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        # Apply standard robust scaling (centers around 0)
        X_scaled = self.scaler.transform(X)
        # Shift to center around 0.5
        return X_scaled + 0.5

    def inverse_transform(self, X):
        # Shift back to 0-centered, then apply inverse transform
        X_shifted = X - 0.5
        return self.scaler.inverse_transform(X_shifted)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


