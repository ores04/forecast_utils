import random

import jax.numpy as jnp

import jax.random
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from sklearn.linear_model import Ridge

from flax import nnx

from forecast_utils.models import LSTM


class LIMEExplainer:

    f_persist = None
    noise_ts = None
    explanation = None
    original_prediction = None
    scaler = None

    def __init__(self, timeseries, timeseries_for_backgound_noise, model: nnx.module, sample_size=1000, random_seed=None, scaler=None):
        self.timeseries = timeseries
        self.model = model
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.timeseries_for_backgound_noise = timeseries_for_backgound_noise

        timeseries_for_lstm_pred = jnp.array([timeseries])
        timeseries_for_lstm_pred = jnp.expand_dims(timeseries_for_lstm_pred, axis=-1)  # Add a channel to fix the input shape for the model (batch_size, time_steps, features)
        self.original_prediction = model(timeseries_for_lstm_pred)
        self.scaler = scaler

    def test(self):
        timeseries_for_lstm_pred = jnp.array([self.timeseries])
        timeseries_for_lstm_pred = jnp.expand_dims(timeseries_for_lstm_pred, axis=-1)
        y = self.model(timeseries_for_lstm_pred)
        # remove value 20 from the timeseries
        first_half = np.copy(self.timeseries)[:20]
        second_half = np.copy(self.timeseries)[21:]
        ts = np.concatenate((first_half, second_half), axis=0)
        timeseries_for_lstm_pred = jnp.array([ts])
        timeseries_for_lstm_pred = jnp.expand_dims(timeseries_for_lstm_pred, axis=-1)
        y_new = self.model(timeseries_for_lstm_pred)
        print(y_new, y)

    def determine_background_frequency(self, timeseries):
        """
        Correctly implements the described algorithm to find and reconstruct the
        most persistent background frequency in a time series.
        """
        # STFT parameters from the original code
        win = hann(128)
        hop = 32
        fs = 1000  # Sampling frequency
        sftf = ShortTimeFFT(win, hop, fs)

        # 1. Perform the STFT
        X_sftf = sftf.stft(timeseries)

        # 2. Get the magnitude response |f_t|
        X_magnitude = np.abs(X_sftf)

        # 3. Find the most persistent frequency
        # We iterate over the frequency bins (rows) to check their stability over time (columns).
        f_persist_idx = -1
        max_ratio = -1

        for i in range(X_magnitude.shape[0]):  # Iterate over FREQUENCY BINS (rows)
            frequency_band_over_time = X_magnitude[i, :]

            mean_mag = np.mean(frequency_band_over_time)
            std_dev_mag = np.std(frequency_band_over_time)

            # Avoid division by zero for silent frequency bands
            if std_dev_mag == 0:
                continue

            # This is the ratio from the paper: μ(|f_t|) / σ(|f_t|)
            ratio = mean_mag / std_dev_mag

            if ratio > max_ratio:
                max_ratio = ratio
                f_persist_idx = i

        if f_persist_idx == -1:
            # Handle case where no valid frequency was found
            print("Warning: Could not determine a persistent frequency.")
            return np.zeros_like(timeseries)

        # 4. Perform inverse STFT using ONLY the persistent frequency
        reconstruction_matrix = np.zeros_like(X_sftf, dtype=complex)
        # Copy the chosen FREQUENCY ROW (the persistent frequency across all time)
        reconstruction_matrix[f_persist_idx, :] = X_sftf[f_persist_idx, :]

        base_ts = sftf.istft(reconstruction_matrix)

        self.f_persist = f_persist_idx
        self.noise_ts = base_ts

        # The iSTFT can sometimes return a slightly different length
        # due to windowing, so we trim it to match the original.
        return f_persist_idx, base_ts[:len(timeseries)]



    def generate_indexes_for_perturbation_random(self) -> list:
        indexes_for_perturbation = []

        for i in range(self.sample_size):
            # Generate a random number of indexes to perturb
            num_indexes = random.randint(1, 4)
            # Generate unique random indexes
            indexes = random.sample(range(len(self.timeseries)), num_indexes)
            # add if not already in the list

            if indexes not in indexes_for_perturbation:
                indexes_for_perturbation.append(indexes)

        return indexes_for_perturbation





    def generate_indexes_for_perturbation(self) -> list:
        """Generates random indexes for perturbation."""
        indexes_for_perturbation = []
        # replace one value in the time series with noise
        indexes_for_perturbation = [i for i in range(len(self.timeseries))]
        # replace two values in the time series with noise
        temp = range(len(self.timeseries))
        cross_prod = [(i, j) for i in temp for j in temp if i != j]
        # add the cross product indexes to the list
        indexes_for_perturbation.extend(cross_prod)
        if len(self.timeseries) > 20:
            # replace three values in the time series with noise
            temp = range(len(self.timeseries))
            cross_prod_3 = [(i, j, k) for i in temp for j in temp for k in temp if i != j and i != k and j != k]
            # add the cross product indexes to the list
            indexes_for_perturbation.extend(cross_prod_3)
        if len(self.timeseries) > 30:
            # replace four values in the time series with noise
            temp = range(len(self.timeseries))
            cross_prod_4 = [(i, j, k, l) for i in temp for j in temp for k in temp for l in temp if i != j and i != k and i != l and j != k and j != l and k != l]
            # add the cross product indexes to the list
            indexes_for_perturbation.extend(cross_prod_4)
        return indexes_for_perturbation



    def explain(self, random_indexes=False):

        if random_indexes:
            pertubation_indexes = self.generate_indexes_for_perturbation_random()
        else:
            pertubation_indexes = self.generate_indexes_for_perturbation()
        perturbed_ts = [self.perturb_timeseries(self.timeseries, indexes) for indexes in pertubation_indexes]

        # get the model predictions for the perturbed time series
        # calculate weight vector w
        w = [self.exponential_kernal(x, 100) for x in perturbed_ts]


        # normalize the weights
        w = np.array(w)
        w /= np.sum(w)

        # convert to jax array
        perturbed_ts_for_lstm = jnp.array(perturbed_ts)
        perturbed_ts_for_lstm = jnp.expand_dims(perturbed_ts_for_lstm, axis=-1) # Add a channel to fix the input shape for the model (batch_size, time_steps, features)

        if self.scaler:
            y = [self.predict_timeseries(np.array([ts])) for ts in perturbed_ts_for_lstm]
        else:
            y = self.model(perturbed_ts_for_lstm)


        # calculate the explanations using ridge regression
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(perturbed_ts, y, sample_weight=w)

        self.explanation = ridge.coef_

        return ridge.coef_

    def visualize_explanation(self, title='LIME Explanation'):
        """
        This function plots the explanation of the timeseries prediction.
        The plot contains the original time series, the explanation and the prediction of the model.
        """
        import matplotlib.pyplot as plt

        # Check if explanation has been computed
        if self.explanation is None:
            print("No explanation available. Please run explain() first.")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)

        # Time axis for plotting
        time_axis = np.arange(len(self.timeseries))

        # Plot 1: Original Time Series
        axes[0].plot(time_axis, self.timeseries, 'b-', linewidth=2, label='Original Time Series')
        axes[0].set_title('Original Time Series')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # plot the prediction into the first plot
        axes[0].axhline(y=float(self.original_prediction[0][0]), color='orange', linestyle='--', linewidth=1, label='Model Prediction')

        # Plot 2: LIME Explanation (Feature Importance)
        # The explanation shows how much each time point contributes to the prediction
        colors = ['red' if coef < 0 else 'green' for coef in self.explanation]
        axes[1].bar(time_axis, self.explanation, color=colors, alpha=0.7)
        axes[1].set_title('LIME Explanation (Feature Importance)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Coefficient Value')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add legend for positive/negative contributions
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Positive Contribution'),
                           Patch(facecolor='red', alpha=0.7, label='Negative Contribution')]
        axes[1].legend(handles=legend_elements)

        # Plot 3: Model Prediction Information
        axes[2].text(0.1, 0.7, f'Original Prediction: {float(self.original_prediction[0][0]):.4f}',
                     transform=axes[2].transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        axes[2].text(0.1, 0.5, f'Sum of Explanations: {float(np.sum(self.explanation)):.4f}',
                     transform=axes[2].transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="purple"))

        axes[2].text(0.1, 0.3, f'Most Important Time Point: {int(np.argmax(np.abs(self.explanation)))}',
                     transform=axes[2].transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        axes[2].text(0.1, 0.1, f'Max Absolute Contribution: {float(np.max(np.abs(self.explanation))):.4f}',
                     transform=axes[2].transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

        axes[2].set_title('Model Prediction Summary')
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\n" + "=" * 50)
        print("LIME Explanation Summary")
        print("=" * 50)
        print(f"Original Prediction: {float(self.original_prediction[0][0]):.6f}")
        print(f"Sum of Explanations: {float(np.sum(self.explanation)):.6f}")
        print(f"Most Important Time Point: {int(np.argmax(np.abs(self.explanation)))}")
        print(f"Strongest Positive Contribution: {float(np.max(self.explanation)):.6f}")
        print(f"Strongest Negative Contribution: {float(np.min(self.explanation)):.6f}")
        print(f"Mean Absolute Contribution: {float(np.mean(np.abs(self.explanation))):.6f}")
        print("=" * 50)



    def visualize_stft_matrix(self, matrix, title='STFT Matrix'):
        """Visualizes the Short-Time Fourier Transform (STFT) matrix."""
        import matplotlib.pyplot as plt
        plt.imshow(np.abs(matrix), aspect='auto', origin='lower')
        plt.title(title)
        plt.xlabel('Frequency Bins')
        plt.ylabel('Time Frames')
        plt.colorbar(label='Magnitude')
        plt.show()

    def perturb_timeseries(self, timeseries, indexes):
        """"""
        # sanity check
        assert len(timeseries) > np.max(indexes)

        noise_ts = self.noise_ts
        perturbed_timeseries = np.copy(timeseries)

        noise_ts_multiple_of_ts = len(noise_ts) // len(timeseries) if len(noise_ts) > len(timeseries)  else 1

        random_mult = random.randint(1,noise_ts_multiple_of_ts)

        if isinstance(indexes, int):
            indexes = [indexes]

        # replace the values at the specified indexes with noise at the same indexes multiplied by a random factor
        for index in indexes:
            perturbed_timeseries[index] = 0.45666224 # Try mean for now TODO

        return perturbed_timeseries




    @staticmethod
    def timeseries_distance(ts1, ts2):
        """ This function implements a distance measure for time series data.
        The distance is calucalte as definded in the LIMESegment Paper"""

        mean_ts1 = float(np.mean(ts1))
        mean_ts2 = float(np.mean(ts2))

        variance_ts1 = np.var(ts1)
        variance_ts2 = np.var(ts2)

        expected_value = lambda timeseries, mean: np.sum([(1/len(timeseries))*(x-mean) for x in timeseries])

        expected_value_ts1 = expected_value(ts1, mean_ts1)
        expected_value_ts2 = expected_value(ts2, mean_ts2)

        return (expected_value_ts2 * expected_value_ts1) / (variance_ts1 * variance_ts2)

    def predict_timeseries(self, timeseries):


        # get the last timeseries
        vola_prediction = self.model(timeseries)
        vola_prediction = vola_prediction[-1].reshape(1, -1)

        # inverse the scaling
        prediction = self.scaler.inverse_transform(vola_prediction)

        pred_mean = jnp.mean(prediction)

        p = prediction - pred_mean
        p = p * 3
        prediction = p + pred_mean
        value = prediction[0]


        return value[0]

    def exponential_kernal(self, x, l):
        """ This implements the exponential kernal function as defined in the LIME paper."""
        if l <= 0:
            raise ValueError("Length scale 'l' must be positive.")
        return np.exp(np.square(self.dtw_z_norm(x, self.timeseries)) / l)

    @staticmethod
    def dtw_z_norm(ts1, ts2):
        """
        Computes the Dynamic Time Warping (DTW) distance between two z-normalized time series.

        This function first applies z-normalization to each input time series to have
        a mean of 0 and a standard deviation of 1. It then calculates the DTW
        distance between the normalized series.

        Args:
            ts1 (np.ndarray): The first time series.
            ts2 (np.ndarray): The second time series.

        Returns:
            float: The DTW distance between the z-normalized time series.
        """
        # Z-normalization
        ts1_norm = (ts1 - np.mean(ts1)) / np.std(ts1)
        ts2_norm = (ts2 - np.mean(ts2)) / np.std(ts2)

        n = len(ts1_norm)
        m = len(ts2_norm)

        # Initialize the cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        # Compute the DTW distance
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(ts1_norm[i - 1] - ts2_norm[j - 1])
                last_min = np.min([dtw_matrix[i - 1, j],  # Insertion
                                   dtw_matrix[i, j - 1],  # Deletion
                                   dtw_matrix[i - 1, j - 1]])  # Match
                dtw_matrix[i, j] = cost + last_min

        return dtw_matrix[n, m]

def generate_random_points_around_sin_wave():
    """
    Generates random points around a sine wave with added noise.
    Returns:
        np.ndarray: Array of points around the sine wave.
    """
    x = np.linspace(0,  8*np.pi, 1000)
    y = np.sin(x)
    return y



if __name__ == "__main__":



    # Example usage
    timeseries = np.random.rand(7)  # Example time series data
    timeseries_for_background_noise = generate_random_points_around_sin_wave()
    model = LSTM(features=1, hidden_features=[12], special_last_layer=False, rngs=nnx.Rngs(jax.random.PRNGKey(0)))  # Placeholder for the model

    lime_explainer = LIMEExplainer(timeseries, timeseries_for_background_noise, model)
    f_persist, base_ts = lime_explainer.determine_background_frequency(timeseries_for_background_noise)
    coefs = lime_explainer.explain()
    lime_explainer.visualize_explanation()

    print(f"Persisted frequency: {f_persist}, Base time series: {base_ts}")
