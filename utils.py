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

def transform_data_to_training_timeseries_jax(data: pd.DataFrame, window_size: int = 100) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Transform the DataFrame into a time series format for training using JAX."""
    # Convert DataFrame to JAX array
    y = jnp.array(data['Target'])
    x_unformatted = jnp.array(data.drop(columns=['Target']))
    x = jnp.array([x_unformatted[i:i + window_size] for i in range(len(x_unformatted) - window_size)])
    y = y[window_size:]  # drop label for series that are not complete
    return x, y

