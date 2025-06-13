from functools import partial

import jax
import optax
from flax import nnx
from jax import numpy as jnp

from forcast_utils.loss import mean_squared_error
from forcast_utils.models import LSTM



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

def evaluate_model(model, x_test: jnp.ndarray, y_test: jnp.ndarray):
    """The function calculate the metrics for the model on the test set. For now we will use a simple MSE"""
    y_pred = model(x_test)
  #  y_test = jnp.reshape(y_test, (-1, 1))
    mse = jnp.mean((y_pred - y_test) ** 2)
    return mse