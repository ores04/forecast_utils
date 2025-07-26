import pathlib

import pandas as pd
import numpy as np
import jax.numpy as jnp
from forecast_utils.base_fn import get_current_path
from forecast_utils import preprocessing
from forecast_utils.models import LSTM
from flax import nnx
import jax
import orbax.checkpoint as ocp
import joblib

def train_test_split(data: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets."""
    split_index = int(len(data) * (1 - test_size))
    data = data[:split_index]
    test_data = data[split_index:]
    return data, test_data

def transform_data_to_training_timeseries_np(data: pd.DataFrame, window_size: int = 100) -> tuple[np.array, np.array]:
    """Transform the DataFrame into a time series format for training."""
    # Convert DataFrame to JAX array
    y = np.array(data['Target'])
    x_unformatted = np.array(data.drop(columns=['Target']))
    x = np.array([x_unformatted[i:i + window_size] for i in range(len(x_unformatted) - window_size)])
    y = y[window_size:] # drop label for series that are not complete
    return x, y

def transform_data_to_training_timeseries_jax(data: pd.DataFrame, window_size: int = 100, columns_to_drop=['Target'], target_column="Target") -> \
    tuple[jnp.ndarray, jnp.ndarray]:
        """Transform the DataFrame into a time series format for training."""
        # Convert DataFrame to JAX array
        y = jnp.array(data[target_column])
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


    
def scale_data(data) -> tuple[pd.DataFrame, RobustScaler05]:

    scaler = fit_scaler_on_saved_data('close_return')
    data['close_return'] = scaler.transform(data[['close_return']])
    scaler = fit_scaler_on_saved_data('realized_volatility')
    data['realized_volatility'] = scaler.transform(data[['realized_volatility']])
    scaler_target = fit_scaler_on_saved_data('target')
    data['target'] = scaler_target.transform(data[['target']])

    data['close_return'] = preprocessing.reduce_high_frequency_components(data['close_return'])
    data['realized_volatility'] = preprocessing.reduce_high_frequency_components(data['realized_volatility'])
    data['target'] = preprocessing.reduce_high_frequency_components(data['target'])
    print(data.head())
    return data, scaler

def fit_scaler_on_saved_data(column: str) -> RobustScaler05:

    path = get_current_path()
    btc = pd.read_csv(path + "/data/BTC_USD-15min.csv", parse_dates=['Open time'], index_col='Open time')
    data = preprocessing.preprocess_15_min_data_for_daily_predict(btc)
    return RobustScaler05().fit(data[[column]])

def get_current_forecast_model() -> LSTM:
    """Loads the model from the curent checkpoint """
    checkpointer = ocp.PyTreeCheckpointer()

    abstract_lstm = nnx.eval_shape(lambda: LSTM(features=2, hidden_features=[32,32,16], special_last_layer=True,rngs=nnx.Rngs(jax.random.PRNGKey(0)), use_dropout=False))
    graph_def, state = nnx.split(abstract_lstm)
    path = pathlib.Path(get_current_path() + '/' + 'checkpoints_double_4_to_4_utc_feature_pytree')

    state_restored = checkpointer.restore(path / 'checkpoint', state)
    model = nnx.merge(graph_def, state_restored)
    return model

if __name__ == "__main__":
    model = get_current_forecast_model()
    nnx.display(model)
