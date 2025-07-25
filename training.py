from functools import partial

import jax
import optax
from flax import nnx
from jax import numpy as jnp

from forecast_utils.loss import mean_squared_error
from forecast_utils.models import LSTM



def train_step_factory(loss_fn):
    grad_fn = nnx.value_and_grad(loss_fn)

    @nnx.jit
    def train_step(model: LSTM, optimizer, x, y):
        """Perform a single training step."""
        loss, grads = grad_fn(model, x, y)
        optimizer.update(grads)  # In place updates.
        return loss, grads
    return train_step

def train_model(model: nnx.Module, optimizer, X_train: jnp.ndarray, Y_train: jnp.ndarray, loss_fn, epochs: int = 10, batch_size: int = 32):
    """Train the LSTM model."""
    train_step = train_step_factory(loss_fn)
    for epoch in range(epochs):
        aggregate_loss = 0.0
        loss_counter = 0
        for i in range(0, len(X_train), batch_size):
            x_batch: jnp.array = X_train[i:i + batch_size]
            y_batch: jnp.array = Y_train[i:i + batch_size]
            loss, grads = train_step(model, optimizer, x_batch, y_batch)
            aggregate_loss += loss
            loss_counter += 1
        average_loss = aggregate_loss / loss_counter
        print(f"Epoch {epoch + 1}, AVG Loss: {average_loss:.8f}")

    return model

def evaluate_model(pred, y_test: jnp.ndarray):
    """The function calculate the metrics for the model on the test set. For now we will use a simple MSE"""
    metrics = compute_metrics(pred, y_test)
    # return epsilon loss
    return metrics['epsilon_loss']

def compute_metrics(predictions: jnp.ndarray, targets: jnp.ndarray) -> dict:
    """Compute metrics for the model predictions."""
    diff = predictions - targets
    # Calculate epsilon loss, which is a modified MSE that ignores small differences
    epsilon = 0.05
    epsilon_loss = jnp.mean(jnp.where(jnp.abs(diff) > epsilon , jnp.exp(jnp.abs(diff)), 0))  # Adding a small epsilon to avoid division by zero
   # epsilon_loss = jnp.mean((predictions - targets) ** 2 if jnp.abs(predictions - targets) > 5e-2 else 0)  # Adding a small epsilon to avoid division by zero
    mse = jnp.mean((predictions - targets) ** 2)
    mae = jnp.mean(jnp.abs(predictions - targets))
    rmse = jnp.sqrt(jnp.mean(jnp.square(predictions - targets)))
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'epsilon_loss': epsilon_loss}