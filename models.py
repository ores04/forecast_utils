# These are all imports specific to the LSTM cell implementation
# Therefore they are not in the top of the file
from typing import Any
from collections.abc import Callable
from functools import partial

import jax
from flax import nnx
import jax.numpy as jnp

from flax.nnx import rnglib, LSTMCell, RNN
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import Linear
from flax.nnx.nn.activations import sigmoid
from flax.nnx.nn.activations import tanh
from flax.typing import (
    Dtype,
    Initializer,
    Shape, Array
)

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


def modified_orthogonal(key: Array, shape: Shape, dtype: Dtype = jnp.float32) -> Array:
    """Modified orthogonal initializer for compatibility with half precision."""
    initializer = initializers.orthogonal()
    return initializer(key, shape).astype(dtype)

class CustomLSTMCell(nnx.RNNCellBase):
    r"""LSTM cell.

    A good part of this code is stolen from the JAX NNX library, but because I need to modify the
    activation function to depend on the input and the hidden state, I had to copy it here.
    But that should be fine as the FLAX NNX library is under the Apache 2.0 license.

    The mathematical definition of the cell is as follows

    .. math::
      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}

    where x is the input, h is the output of the previous time step, and c is
    the memory.
    """

    __data__ = ('ii', 'if_', 'ig', 'io', 'hi', 'hf', 'hg', 'ho', 'rngs')

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: Callable[..., Any] = sigmoid,
        activation_fn: Callable[..., Any] = tanh,
        kernel_init: Initializer = default_kernel_init,
        recurrent_kernel_init: Initializer = modified_orthogonal,
        bias_init: Initializer = initializers.zeros_init(),
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        rngs: rnglib.Rngs,
        ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.rngs = rngs

        # input and recurrent layers are summed so only one needs a bias.
        dense_i = partial(
          Linear,
          in_features=in_features,
          out_features=hidden_features,
          use_bias=False,
          kernel_init=self.kernel_init,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          rngs=rngs,
        )

        dense_h = partial(
          Linear,
          in_features=hidden_features,
          out_features=hidden_features,
          use_bias=True,
          kernel_init=self.recurrent_kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          rngs=rngs,
        )

        self.garch = GARCHModel()

        self.ii = dense_i()
        self.if_ = dense_i()
        self.ig = dense_i()
     #   self.io = dense_i() this is not used due to the fact that the GARCH model is used instead of the output gate
        self.hi = dense_h()
        self.hf = dense_h()
        self.hg = dense_h()
       # self.ho = dense_h() same reason as above, the GARCH model is used instead of the output gate

        # The output gate is replaced by the GARCH model
        self.ho = Linear(
            in_features=hidden_features,
            out_features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs
        )

        # Initialize the carry state
        self.carry_init = initializers.zeros_init()

    def __call__(
        self, carry: tuple[Array, Array], inputs: Array
        ) -> tuple[tuple[Array, Array], Array]:  # type: ignore[override]
        r"""A long short-term memory (LSTM) cell.

        Args:
          carry: the hidden state of the LSTM cell,
            initialized using ``LSTMCell.initialize_carry``.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """
        c, h = carry
        i = self.gate_fn(self.ii(inputs) + self.hi(h))
        f = self.gate_fn(self.if_(inputs) + self.hf(h))
        g = self.activation_fn(self.ig(inputs) + self.hg(h))
        # Apply the GARCH model to the input and hidden state.
        o = self.garch(inputs, h)
        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h

    def initialize_carry(
        self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None
        ) -> tuple[Array, Array]:  # type: ignore[override]
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.
        Returns:
          An initialized carry for the given RNN cell.
        """
        batch_dims = input_shape[:-1]
        if rngs is None:
          rngs = self.rngs
        mem_shape = batch_dims + (self.hidden_features,)
        c = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
        h = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
        return c, h

    @property
    def num_feature_axes(self) -> int:
        return 1



class GARCHModel(nnx.Module):
    """This model implements a GARCH(1, 1) model layer for volatility prediction.
    The idea is that as indicated in the paper "From GARCH to Neural Network for Volatility Forecast" we can enhance
    the LSTM model with a GARCH layer to improve the volatility prediction. """
    def __init__(self):
        initial_param_value = jnp.zeros((1, 1)) + 0.1  # Initialize parameters to zero
        self.alpha_raw = nnx.Param(initial_param_value)
        self.beta_raw = nnx.Param(initial_param_value)
        self.omega_raw = nnx.Param( initial_param_value)

    def __call__(self, x: jnp.ndarray, h:jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the GARCH model. x is the input data at time t and h is the hidden state.
        So h could be seen as the previous volatility prediction."""
        # Calculate the GARCH(1, 1) volatility
        # x does not need to be squared as it is already defined as such
        # for now we assume that x is of shape (batch_size, 1)
        assert x.shape == (x.shape[0], 1), "Input x must be of shape (batch_size, 1)"
        h_squared = jnp.square(h)
        omega = self.omega_raw.value
        alpha = self.alpha_raw.value
        beta =  self.beta_raw.value

        # lets constrain omega to be positive
        omega = jnp.maximum(omega, jnp.array([1e-6]))

        # enforce alpha, beta sum to be less than 1
        # ie 0 < alpha + beta < 1
        s = sigmoid(alpha)
        alpha = s * (1 - beta)  # Scale alpha to be in the range [0, 1 - beta]
        beta = s * beta  # Scale beta to be in the range [0, 1 - alpha]

        return omega + alpha * x + beta * h


# now lets define the LSTM model
class LSTM(nnx.Module):

    def __init__(self, features: int, hidden_features: list[int],special_last_layer: bool, rngs: nnx.Rngs | None = None):
        """This function initializes the LSTM model.
            features: int - the number of features in the input data, this means the number of features each time step has.
            hidden_features: list[int] - the number of hidden features in the LSTM cell, the lists length is the number of layers in the LSTM while the last layer
            will be a CustomLSTM cell with a specific activation function.
            rngs: nnx.Rngs | None - the random number generator, if None it will use the default one.
        """

        self.lstm_cells = self.construct_lstm_cells(
            features=features,
            hidden_features=hidden_features,
            sepecial_last_layer=special_last_layer,
            rngs=rngs
        )

        self.reccurent_layers = self.construct_reccurent_layers(
            lstm_cells=self.lstm_cells,
            rngs=rngs,
            special_last_layer=special_last_layer,
        )


        self.linear_layer1 = nnx.Linear(
            in_features=hidden_features[-1],
            out_features=1,  # Output a single value for volatility prediction
            use_bias=True,
            rngs=rngs
        )

    def __call__(self, x):
        """Forward pass of the LSTM"""
        for recurrent, linear in self.reccurent_layers:
            x = recurrent(x)  # Pass the input through the recurrent layer
            if linear is not None:
                x = linear(x)
        # now feed it to the linear layer
        x = x[:, -1, :]
        x = self.linear_layer1(x)
        return x  # Return the last output of the LSTM, which is the prediction for the last time step

    @staticmethod
    def construct_lstm_cells(features: int, hidden_features: list[int], sepecial_last_layer:bool, rngs: nnx.Rngs | None = None) -> list[nnx.LSTMCell]:
        """Construct the LSTM cells for the model."""
        lstm_cells = []
        for i in range(len(hidden_features)):
            if i == len(hidden_features) - 1 and sepecial_last_layer:
                # Use the CustomLSTMCell for the last layer
                lstm_cells.append(CustomLSTMCell(
                    in_features=1,
                    hidden_features=hidden_features[i],
                    rngs=rngs
                ))
            else:
                # Use the standard LSTMCell for other layers
                lstm_cells.append(LSTMCell(
                    in_features=features if i == 0 else hidden_features[i],
                    hidden_features=hidden_features[i],
                    rngs=rngs
                ))
        return lstm_cells
    @staticmethod
    def construct_reccurent_layers(lstm_cells: list[nnx.LSTMCell], rngs: nnx.Rngs | None = None, special_last_layer=False) -> list[tuple[nnx.RNN, nnx.Linear]]:
        """Construct the recurrent layers for the model."""
        layers = []
        for i in range(len(lstm_cells)):
            # Create a recurrent layer for each LSTM cell
            rnn_layer = nnx.RNN(
                cell=lstm_cells[i],
                rngs=rngs,
            )
            if i == len(lstm_cells) - 1:
                # If it's the last layer, we don't need a linear layer after it
                layers.append((rnn_layer, None))
                break
            # Add a linear layer after each RNN layer
            if i == len(lstm_cells) - 2 and special_last_layer:
                linear_layer = linear_layer = nnx.Linear(
                in_features=lstm_cells[i].hidden_features,
                out_features=1,
                use_bias=True,
                rngs=rngs
                )
            else:
                linear_layer = nnx.Linear(
                    in_features=lstm_cells[i].hidden_features,
                    out_features=lstm_cells[i + 1].hidden_features,
                    use_bias=True,
                    rngs=rngs
                )
            layers.append((rnn_layer, linear_layer))
        return layers





if __name__ == "__main__":
    model = GARCHModel()
    nnx.display(model)
    lstm = LSTM(4, [256, 128],True, nnx.Rngs(jax.random.PRNGKey(0)))
    nnx.display(lstm)
