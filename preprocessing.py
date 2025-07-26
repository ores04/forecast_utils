import os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pywt

from forecast_utils import base_fn, visualisation


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical data for a given ticker."""
    # check if the data is already downloaded
    if f"{ticker}_data.csv" in os.listdir():
        print(f"Data for {ticker} already downloaded.")
        dataframe = pd.read_csv(f"{ticker}_data.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
        # remove first two rows as they are not needed
        dataframe = dataframe.iloc[2:]
        # interpret all but the first column as float
        for col in dataframe.columns[1:]:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='raise')
        dataframe['Close'] = pd.to_numeric(dataframe['Close'], errors='raise')
        return dataframe


    data = yf.download(ticker, start=start, end=end)
    # save the data to a CSV file
    data.to_csv(f"{ticker}_data.csv")

    return data

def calculate_log_realized_volatility_6_to_6(
    data: pd.DataFrame,
    price_column: str,
    start_hour: int = 18) -> pd.Series:
    # --- Input Validation ---
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex.")
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in the DataFrame.")

    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()

    # 1. Calculate 15-minute log_returns
    df['log_return'] = np.log(df[price_column].pct_change())

    # 2. Define the custom trading day by shifting the index
    time_shift = pd.Timedelta(hours=start_hour)
    df['trading_day'] = (df.index - time_shift).date # This means that the trading day is always glued to the start date of the trading day, e.g. 6am to 6am

    # 3. Group by the custom trading day, calculate sum of squared returns, and then sqrt
    # The .sum() calculates the total variance for the 6-to-6 period.
    daily_variance = df.groupby('trading_day')['log_return'].apply(lambda x: (x**2).sum())

    # Take the square root to get the final volatility
    daily_volatility = np.sqrt(daily_variance)
    daily_volatility.name = "realized_log_volatility"

    return daily_volatility


def calculate_final_realized_volatility_6_to_6(
    data: pd.DataFrame,
    price_column: str,
    start_hour: int = 6
) -> pd.Series:
    """
    Calculates the final realized volatility for a custom daily window (e.g., 6am to 6am).

    This function directly computes the final daily value without returning
    intermediate cumulative steps.

    Args:
        data (pd.DataFrame): DataFrame with a DatetimeIndex and price data.
        price_column (str): The name of the column containing the price data.
        start_hour (int): The hour at which the trading day begins.

    Returns:
        pd.Series: A Series indexed by date, containing the final realized
                   volatility for each custom trading day.
    """
    # --- Input Validation ---
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex.")
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in the DataFrame.")

    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()

    # 1. Calculate 15-minute returns
    df['return'] = df[price_column].pct_change()

    # 2. Define the custom trading day by shifting the index
    time_shift = pd.Timedelta(hours=start_hour)
    df['trading_day'] = (df.index - time_shift).date # This means that the trading day is always glued to the start date of the trading day, e.g. 6am to 6am

    # 3. Group by the custom trading day, calculate sum of squared returns, and then sqrt
    # The .sum() calculates the total variance for the 6-to-6 period.
    daily_variance = df.groupby('trading_day')['return'].apply(lambda x: (x**2).sum())

    # Take the square root to get the final volatility
    daily_volatility = np.sqrt(daily_variance)
    daily_volatility.name = "realized_volatility"

    return daily_volatility


def preprocess_15_min_data_for_daily_predict(data, use_log_as_target=False) -> pd.DataFrame:
    data = data.copy()
    daily_realied_vola = calculate_final_realized_volatility_6_to_6(data=data, price_column="Close", start_hour=16)
    daily_realized_log_vola = calculate_log_realized_volatility_6_to_6(data=data, price_column="Close", start_hour=16)
    # remove the last value as it is not complete
    daily_realied_vola = daily_realied_vola[:-1]
    daily_realized_log_vola = daily_realized_log_vola[:-1]
    # resamle so that we have a daily frequency from 6pm to 6pm


    aggregation_rules = {
        'Close': 'last',
    }
    valid_rules = {col: rule for col, rule in aggregation_rules.items() if col in data.columns}
    # shift the whole dataframe 18 hours back to align with the 4pm to 4pm trading day UTC
    data.index = data.index - pd.Timedelta(hours=16)
    aggregated_dataframe = data.resample('D').agg(valid_rules)
    realized_vola = pd.DataFrame(daily_realied_vola)
    # make sure the index is a datetime index
    realized_vola.index = pd.to_datetime(realized_vola.index, utc=True)
    realized_vola.tz_convert("UTC")

    realized_log_vola = pd.DataFrame(daily_realized_log_vola)
    realized_log_vola.index = pd.to_datetime(realized_log_vola.index, utc=True)
    # check if aggregated_dataframe has a tz aware index
    aggregated_dataframe.index = pd.to_datetime(aggregated_dataframe.index, utc=True)

    aggregated_dataframe = aggregated_dataframe.join(realized_vola, how='inner',)
    aggregated_dataframe = aggregated_dataframe.join(realized_log_vola, how='inner',)
    aggregated_dataframe['close_return'] = aggregated_dataframe['Close'].pct_change() # the first value will be assumed to be the same as the second
    aggregated_dataframe['close_return'] = aggregated_dataframe['close_return'].fillna(aggregated_dataframe.iloc[1]['close_return'])  # fill NaN with the first value
    aggregated_dataframe['target'] = aggregated_dataframe['realized_volatility']
    if use_log_as_target:
        aggregated_dataframe['target'] = aggregated_dataframe['realized_log_volatility']


    return aggregated_dataframe

def preprocess_15_min_data(data: pd.DataFrame, target_column: str, feature_columns: list, window: int) -> pd.DataFrame:
    """ This function will preprocess data given in a 15 minute interval such that it is ready for training. In this case the window which is asked in the other functions is not needed as
    we proxy the volatility with the standart realized volatility."""

    # shift the data to align with the 6pm to 6pm trading day
    data = data.copy()
    data.index = data.index - pd.Timedelta(hours=16)  # Shift to align with 6pm to 6pm MESZ trading day

    standart_realized_volatility = calculate_standard_realized_volatility(data, base_column='Close', use_log=True)
    aggregation_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    # Filter for columns that exist in the dataframe before aggregating
    valid_rules = {col: rule for col, rule in aggregation_rules.items() if col in data.columns}
    aggregated_dataframe = data.resample('D').agg(valid_rules)
    aggregated_dataframe = aggregated_dataframe.dropna()

    # add the standart realized volatility to the aggregated dataframe
    aggregated_dataframe['Standard_Realized_Volatility'] = standart_realized_volatility

    dataframe = preprocess_data(aggregated_dataframe, target_column, feature_columns + ['Standard_Realized_Volatility'], window=window)
    return dataframe





def calculate_standard_realized_volatility(data: pd.DataFrame, base_column, start_time=16, use_log=False) -> pd.Series:
    """ This function will calculate the standard realized volatility based on the base column. """
    assert base_column in data.columns, f"Data must contain '{base_column}' column"
    assert isinstance(data.index, pd.DatetimeIndex), "Data index must be a DatetimeIndex"

    # caclulate the 15 minute log_returns
    return_series= data[base_column].pct_change().copy()
    if use_log:
        return_series = np.log(1 + return_series)

    return_series = return_series.fillna(0)
    return_series = return_series ** 2  # Square the returns to get the variance
    return_series = return_series.resample('D').sum()  # Resample to daily returns
    return_series = np.sqrt(return_series)  # Take the square root to get the standard deviation
    return return_series


def preprocess_data(data: pd.DataFrame, target_column: str, feature_columns: list, window: int = 30, base_for_target="Standard_Realized_Volatility") -> pd.DataFrame:
    """ This function will preprocess the data so that it is ready for training. """
    assert window > 0, "Window must be greater than 0."
    assert data is not None, "Data must not be None."

    data_temp = calculate_log_returns(data, base_column='Close')
    data_temp = calculate_volatility(data_temp, base_column='Log_Returns', window=window)
    close_column = 'Close'
    base_column = f'Log_Returns_{window}_Volatility'

    data_temp = calculate_returns(data_temp, base_column=close_column)
    data_temp = calculate_volatility(data_temp, base_column=close_column + '_Returns', window=window)
    data_temp = calculate_window_variance(data_temp, volatility_column=base_column)

    variance_column = f'{base_column}_Variance'

    data_temp = add_target_column(data_temp, target_column=target_column, base_column=base_for_target)
    # for now we will not scale the data as we want to use the raw values for the GARCH model
#    data_temp = scale_data(data_temp, columns=feature_columns + [target_column] + [base_column] + [variance_column])

    data_temp = data_temp.dropna()
    #data_temp = data_temp.reset_index(drop=True)
    data_temp = data_temp[feature_columns + [target_column, base_column, 'Log_Returns', f'{base_column}_Variance', 'Close_Returns', f'Close_Returns_{window}_Volatility']]
    return data_temp

def calculate_returns(data: pd.DataFrame, base_column: str, use_log=False) -> pd.DataFrame:
    """ This function will calculate the returns based on the base column. """
    assert base_column in data.columns, f"Data must contain '{base_column}' column"

    data[f'{base_column}_Returns'] = data[base_column].pct_change()
    if use_log:
        data[f'{base_column}_Returns'] = np.log(1 + data[f'{base_column}_Returns'])
    return data

def calculate_log_returns(data: pd.DataFrame, base_column: str) -> pd.DataFrame:
    """ This function will calculate the log returns based on the base column. """
    assert base_column in data.columns, f"Data must contain '{base_column}' column"

    returns = data[base_column].pct_change()
    data[f'Log_Returns'] = np.log(1 + returns)
    return data

def calculate_window_variance(data: pd.DataFrame, volatility_column: str) -> pd.DataFrame:
    """ This function will calculate the rolling variance of the volatility column. """
    assert volatility_column in data.columns, f"Data must contain '{volatility_column}' column"

    data[f'{volatility_column}_Variance'] = np.square(data[volatility_column])
    return data

def add_target_column(data: pd.DataFrame, target_column: str, base_column:str) -> pd.DataFrame:
    """ This function will preprocess the data by adding a target column based on the base column. """
    assert base_column in data.columns, f"Data must contain '{base_column}' column"
    assert target_column not in data.columns, f"Data must contain '{target_column}' column"
    print("Using base column:", base_column, "to create target column:", target_column)
    # Add target column
    data[target_column] = data[base_column].shift(-1)
    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data

def scale_data(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """ This function will scale the data using MinMaxScaler. """

    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def calculate_volatility(data: pd.DataFrame, base_column: str,  window: int = 30) -> pd.DataFrame:
    """ This function will add a column with the volatitlity variance of the base Column."""
    assert base_column in data.columns, f"Data must contain '{base_column}' column"

    data[f'{base_column}_{window}_Volatility'] = data[base_column].rolling(window).std()
    return data

def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """ This function will drop the specified columns from the data. """
    assert isinstance(columns, list), "Columns must be a list"
    data = data.drop(columns=columns, errors='ignore')
    return data

def reduce_high_frequency_components(volatility_series) -> pd.DataFrame:
    """ This function will apply a wavelet transform to reduce high frequency components in the data."""
    wavelet = 'sym2'
    # The level of decomposition depends on the signal length and desired smoothing.
    level = 2
    coeffs = pywt.swt(volatility_series, wavelet, level=level)

    # 3. Threshold the detail coefficients to remove noise
    # A universal threshold is a common choice.
    sigma = np.median(np.abs(coeffs[-1][-1] - np.median(coeffs[-1][-1]))) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(volatility_series)))

    thresholded_coeffs = []
    for ca, cd in coeffs:
        cd_thresh = pywt.threshold(cd, threshold, mode='soft')
        thresholded_coeffs.append((ca, cd_thresh))

    # 4. Reconstruct the smoothed time series using the inverse SWT
    smoothed_series = pywt.iswt(thresholded_coeffs, wavelet)
    return smoothed_series


if __name__ == "__main__":
    # Example usage
    current_path = base_fn.get_current_path()  # workaround for the jupyter notebook
    btc = pd.read_csv(current_path + "/data/BTC_USD-15min.csv", parse_dates=['Open time'], index_col='Open time')
    data = preprocess_15_min_data(btc.copy(), target_column="Target",
                                                feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                                                window=5)
    data_preprocessed = drop_columns(data.copy(),
                                                   ['Open', 'High', 'Low', 'Log_Returns_5_Volatility', 'Log_Returns',
                                                    'Log_Returns_5_Volatility_Variance', 'Close'])

    standard_vola_copied = data_preprocessed['Standard_Realized_Volatility'].copy()
    wavelet_transformed = reduce_high_frequency_components(standard_vola_copied[4:])
    wavelet_transformed_shortened = reduce_high_frequency_components(wavelet_transformed[-128:])
    print(data_preprocessed['Standard_Realized_Volatility'].shape)

    visualisation.plot_timeseries(wavelet_transformed[-128:],
                                  standard_vola_copied[-128:].reset_index(drop=True),)