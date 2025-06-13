from forcast_utils.models import LSTM
from jax import numpy as jnp


def t_loss_fn(model: LSTM, x: jnp.ndarray, y: jnp.ndarray, v: int) -> jnp.ndarray:
    """ This function implements the T-loss function. Which is a loss function modeled after the Student's t-distribution.
    v is the degrees of freedom parameter. The lower the v, the heavier the tails of the distribution."""
    assert v > 2, "Degrees of freedom v must be greater than 2 for the T-loss function to be well-defined."
    y_pred = model(x)
    regularization_term = jnp.log(jnp.square(y_pred))/2  # Regularization term to prevent overfitting
    error_term = (v+1)/2 * jnp.log(1+(jnp.square(y)/((v -2) * jnp.square(y_pred))))
    loss = jnp.mean(error_term + regularization_term)  # Mean of the negative log-likelihood
    return loss


def mean_absolute_error(model: LSTM, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Calculate the Mean Absolute Error (MAE) between predictions and true values."""
    y_pred = model(x)
    return jnp.mean(jnp.abs(y_pred - y))

def mean_squared_error(model: LSTM, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Calculate the Mean Squared Error (MSE) between predictions and true values."""
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)