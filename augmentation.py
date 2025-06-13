import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def augment_data(data: np.array, label: np.array) -> tuple[np.array, np.array]:

    augmented_data_x = []
    augmented_data_y = []
    assert len(data) == len(label), "Data and label must have the same length."
    for i in range(len(data)):
        # Apply augmentations to each time series in the DataFrame
        augmented_series = augment_timeseries(
            data[i].reshape(-1, 1),
            do_jitter=True,
            do_scale=False,
            do_flip=False,
            jitter_sigma=0.011,
            scale_factor=1.2,
            crop_ratio=0.8,
            warp_knots=5,
            permute_segments=10
        )
        #  add the augmented series to the list
        for series in augmented_series:
            augmented_data_x.append(series)
            augmented_data_y.append(label[i])

    augmented_data_x = np.array(augmented_data_x)
    augmented_data_y = np.array(augmented_data_y)

    return np.concat((data, augmented_data_x), ), np.concat((label, augmented_data_y))

def jitter(x, sigma=0.03):
    """
    Adds random Gaussian noise to a time series.

    Args:
        x (np.ndarray): Input time series of shape (n_timesteps, 1).
        sigma (float): Standard deviation of the Gaussian noise to be added.

    Returns:
        np.ndarray: The jittered time series.
    """
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scale(x, factor=1.1):
    """
    Scales a time series by a random factor.

    The scaling factor is chosen uniformly from the range [1/factor, factor].

    Args:
        x (np.ndarray): Input time series of shape (n_timesteps, 1).
        factor (float): The maximum scaling factor. Must be >= 1.

    Returns:
        np.ndarray: The scaled time series.
    """
    if factor < 1:
        raise ValueError("The scaling factor must be greater than or equal to 1.")
    scaling_factor = np.random.uniform(1 / factor, factor)
    return x * scaling_factor


def flip(x):
    """
    Flips the sign of the time series (multiplies by -1).

    Args:
        x (np.ndarray): Input time series of shape (n_timesteps, 1).

    Returns:
        np.ndarray: The flipped time series.
    """
    return -x


def crop(x, crop_ratio=0.9):
    """
    Randomly crops a segment and pads it back to the original length.

    A random segment of `crop_ratio` of the original length is selected. This
    segment is then placed at a random location within a new array of the
    original size, with the rest of the values being zero.

    Args:
        x (np.ndarray): Input time series of shape (n_timesteps, 1).
        crop_ratio (float): The ratio of the original length to keep (e.g., 0.9
                            means 90% of the series is kept). Must be in (0, 1].

    Returns:
        np.ndarray: The cropped and padded time series.
    """
    original_len = x.shape[0]
    crop_len = int(original_len * crop_ratio)

    start_idx = np.random.randint(0, original_len - crop_len + 1)
    cropped_segment = x[start_idx: start_idx + crop_len]

    # Create a new array of zeros and place the cropped segment randomly
    padded_x = np.zeros_like(x)
    pad_start_idx = np.random.randint(0, original_len - crop_len + 1)
    padded_x[pad_start_idx: pad_start_idx + crop_len] = cropped_segment

    return padded_x


def time_warp(x, n_knots=4):
    """
    Warps the time dimension of a series using a smooth random curve.

    This is done by generating a smooth cubic spline that maps the time
    axis to a new, distorted time axis.

    Args:
        x (np.ndarray): Input time series of shape (n_timesteps, 1).
        n_knots (int): The number of knots to use for the warping spline.

    Returns:
        np.ndarray: The time-warped series.
    """
    n_timesteps = x.shape[0]

    # Generate random knots and their displacements.
    # Knots are points in time, displacements are how much they move.
    knot_indices = np.linspace(0, n_timesteps - 1, n_knots)
    knot_displacements = np.random.normal(loc=1.0, scale=0.2, size=n_knots)

    # Create the smooth warping curve using a cubic spline
    spline = CubicSpline(knot_indices, knot_displacements)
    warped_time_axis = spline(np.arange(n_timesteps))

    # Resample the original series at the new, warped time indices
    # We use linear interpolation to find the values at the new indices
    original_time_axis = np.arange(n_timesteps)
    x_resampled = np.interp(original_time_axis, warped_time_axis, x.flatten())

    return x_resampled.reshape(-1, 1)


def magnitude_warp(x, n_knots=4, sigma=0.2):
    """
    Warps the magnitude of a series using a smooth random curve.

    This creates a smooth curve (a cubic spline) and multiplies the time
    series by it.

    Args:
        x (np.ndarray): Input time series of shape (n_timesteps, 1).
        n_knots (int): The number of knots for the spline.
        sigma (float): Std deviation of the random values at the knots.

    Returns:
        np.ndarray: The magnitude-warped series.
    """
    n_timesteps = x.shape[0]

    # Generate random knot points and the random values at those points
    knot_indices = np.linspace(0, n_timesteps - 1, n_knots)
    knot_values = np.random.normal(loc=1.0, scale=sigma, size=n_knots)

    # Create the smooth warping curve (spline) that will multiply the series
    spline = CubicSpline(knot_indices, knot_values)
    warp_curve = spline(np.arange(n_timesteps))

    return x * warp_curve.reshape(-1, 1)


def permute(x, n_segments=5):
    """
    Splits the time series into segments and randomly shuffles them.

    Args:
        x (np.ndarray): Input time series of shape (n_timesteps, 1).
        n_segments (int): The number of segments to split the series into.

    Returns:
        np.ndarray: The permuted time series.
    """
    n_timesteps = x.shape[0]
    segments = np.array_split(x, n_segments)
    np.random.shuffle(segments)
    return np.vstack(segments)


def augment_timeseries(
        x,
        do_jitter=False,
        do_scale=False,
        do_flip=False,
        do_crop=False,
        do_time_warp=False,
        do_magnitude_warp=False,
        do_permute=False,
        # --- Parameters for each augmentation ---
        jitter_sigma=0.03,
        scale_factor=1.1,
        crop_ratio=0.9,
        warp_knots=4,
        permute_segments=5
):
    """
    Applies a sequence of augmentations to a time series.

    This is the main 'head' function that calls individual augmentation
    methods based on the boolean flags provided. By default, all are off.

    Args:
        x (np.ndarray): Input time series of shape (n_timesteps, 1).
        do_... (bool): Flags to turn each augmentation on or off.
        ..._param: Parameters for the corresponding augmentation function.

    Returns:
        np.ndarray: The augmented time series.
    """
    augmented_x = x.copy()
    augmented_series = []

    # The order of augmentations can matter. A common order is to apply
    # structural changes first, then magnitude changes, and finally noise.
    if do_permute:
        augmented_x = permute(augmented_x, n_segments=permute_segments)
        augmented_series.append(augmented_x)
    if do_time_warp:
        augmented_x = time_warp(augmented_x, n_knots=warp_knots)
        augmented_series.append(augmented_x)
    if do_crop:
        augmented_x = crop(augmented_x, crop_ratio=crop_ratio)
        augmented_series.append(augmented_x)
    if do_magnitude_warp:
        augmented_x = magnitude_warp(augmented_x, n_knots=warp_knots)
        augmented_series.append(augmented_x)
    if do_scale:
        augmented_x = scale(augmented_x, factor=scale_factor)
        augmented_series.append(augmented_x)
    if do_flip:
        augmented_x = flip(augmented_x)
        augmented_series.append(augmented_x)
    # Jitter is typically applied last to simulate sensor noise.
    if do_jitter:
        augmented_x = jitter(augmented_x, sigma=jitter_sigma)
        augmented_series.append(augmented_x)

    return augmented_series


# --- Example Usage and Visualization ---
if __name__ == '__main__':
    # Create a sample timeseries. In your use case, you would loop through
    # your 2000 samples and apply augmentations.
    # A sine wave with a linear trend provides a good visual example.
    timesteps = 200
    t = np.linspace(0, 10, timesteps)
    original_series = (np.sin(2 * t) + t * 0.1).reshape(timesteps, 1)

    # --- Demonstrate the main 'head' function ---

    # 1. No augmentation (all flags are False by default)
    no_aug = augment_timeseries(original_series)
    print(f"Are original and non-augmented series the same? {np.allclose(original_series, no_aug)}")

    # 2. Call a single augmentation
    jitter_only = augment_timeseries(original_series, do_jitter=True, jitter_sigma=0.05)

    # 3. Call a combination of augmentations
    combined_aug = augment_timeseries(
        original_series,
        do_scale=True,
        do_time_warp=True,
        do_jitter=True,
        scale_factor=1.5,
        warp_knots=5,
        jitter_sigma=0.02
    )

    # 4. Create a showcase of each augmentation type individually
    augmentations = {
        "Original": original_series,
        "Jitter": augment_timeseries(original_series, do_jitter=True, jitter_sigma=0.05),
        "Scale": augment_timeseries(original_series, do_scale=True, scale_factor=1.5),
        "Flip": augment_timeseries(original_series, do_flip=True),
        "Crop": augment_timeseries(original_series, do_crop=True, crop_ratio=0.8),
        "Time Warp": augment_timeseries(original_series, do_time_warp=True),
        "Magnitude Warp": augment_timeseries(original_series, do_magnitude_warp=True),
        "Permute": augment_timeseries(original_series, do_permute=True, permute_segments=10),
        "Combined (Scale+Warp+Jitter)": combined_aug
    }

    # --- Plot the results for visualization ---
    n_plots = len(augmentations)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, n_plots * 2.5), sharex=True)
    fig.suptitle("Time Series Augmentation Showcase", fontsize=16)

    for i, (title, series) in enumerate(augmentations.items()):
        ax = axes[i]
        ax.plot(original_series, color='gray', linestyle='--', label='Original' if i > 0 else None)
        ax.plot(series, color='blue', label='Augmented')
        ax.set_title(title, fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Timesteps")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()
