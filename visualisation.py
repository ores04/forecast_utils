from matplotlib import pyplot as plt



def visualize_forcast(forcast, ground_trouth=None, base_assest=None, title='Forecast Visualization', label_forcast='Forecast', label_ground_truth='Ground Truth', label_base_asset='Base Asset'):

    """Visualizes the forecasted values against the ground truth and base asset prices."""
    plt.figure(figsize=(14, 7))

    # Plot forecasted values
    plt.plot(forcast, label=label_forcast, color='blue', linewidth=2)

    # Plot ground truth if provided
    if ground_trouth is not None:
        plt.plot(ground_trouth, label=label_ground_truth, color='orange', linestyle='--', linewidth=2)

    # Plot base asset prices if provided
    if base_assest is not None:
        plt.plot(base_assest, label=label_base_asset, color='green', linestyle=':', linewidth=1, alpha=0.5)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()