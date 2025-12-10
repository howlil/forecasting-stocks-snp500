"""
Utility functions untuk aplikasi FinScope.
Berisi helper functions untuk save/load models, metrics calculation, dan disclaimer.
"""

import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def ensure_models_dir():
    """Membuat folder models/ jika belum ada."""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    return models_dir


def save_model(model, ticker: str):
    """
    Menyimpan model Prophet ke disk menggunakan pickle.
    
    Parameters:
    -----------
    model : Prophet model
        Model Prophet yang sudah di-fit
    ticker : str
        Ticker symbol untuk nama file
    
    Returns:
    --------
    str : Path file yang disimpan
    """
    models_dir = ensure_models_dir()
    filepath = models_dir / f"prophet_model_{ticker}.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    return str(filepath)


def load_model(ticker: str):
    """
    Memuat model Prophet dari disk.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol untuk nama file
    
    Returns:
    --------
    Prophet model atau None jika file tidak ditemukan
    """
    models_dir = ensure_models_dir()
    filepath = models_dir / f"prophet_model_{ticker}.pkl"
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model


def calculate_rmse(y_true, y_pred):
    """
    Menghitung Root Mean Squared Error (RMSE).
    
    Parameters:
    -----------
    y_true : array-like
        Nilai aktual
    y_pred : array-like
        Nilai prediksi
    
    Returns:
    --------
    float : RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """
    Menghitung Mean Absolute Error (MAE).
    
    Parameters:
    -----------
    y_true : array-like
        Nilai aktual
    y_pred : array-like
        Nilai prediksi
    
    Returns:
    --------
    float : MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_metrics(y_true, y_pred):
    """
    Menghitung semua metrics sekaligus.
    
    Parameters:
    -----------
    y_true : array-like
        Nilai aktual
    y_pred : array-like
        Nilai prediksi
    
    Returns:
    --------
    dict : Dictionary berisi RMSE, MAE, dan MAPE
    """
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }




def check_null_percentage(df: pd.DataFrame, threshold: float = 20.0):
    """
    Mengecek persentase null values dalam dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe yang akan dicek
    threshold : float
        Threshold persentase null (default: 20%)
    
    Returns:
    --------
    dict : Dictionary berisi info null percentage per kolom dan warning status
    """
    # Comprehensive None and empty check
    if df is None:
        return {
            'null_percentages': {},
            'max_null': 0,
            'has_warning': False,
            'warning_message': ''
        }
    
    # Check if it's a DataFrame
    if not isinstance(df, pd.DataFrame):
        return {
            'null_percentages': {},
            'max_null': 0,
            'has_warning': False,
            'warning_message': ''
        }
    
    # Check if empty
    if df.empty or len(df) == 0:
        return {
            'null_percentages': {},
            'max_null': 0,
            'has_warning': False,
            'warning_message': ''
        }
    
    try:
        null_percentages = (df.isnull().sum() / len(df)) * 100
        max_null = null_percentages.max()
        
        return {
            'null_percentages': null_percentages.to_dict(),
            'max_null': max_null,
            'has_warning': max_null > threshold,
            'warning_message': f"⚠️ Peringatan: Beberapa kolom memiliki >{threshold}% null values. Hasil analisis mungkin tidak akurat."
        }
    except Exception as e:
        # Return safe default on any error
        return {
            'null_percentages': {},
            'max_null': 0,
            'has_warning': False,
            'warning_message': ''
        }


def downsample_data(df: pd.DataFrame, max_rows: int = 1000000):
    """
    Downsample data jika terlalu besar untuk performa yang lebih baik.
    Menggunakan stratified sampling untuk menjaga distribusi data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe yang akan di-downsample
    max_rows : int
        Maksimum jumlah rows (default: 1M)
    
    Returns:
    --------
    pd.DataFrame : Dataframe yang sudah di-downsample (jika perlu)
    """
    if df is None or df.empty:
        return df
    
    if len(df) <= max_rows:
        return df
    
    # Calculate step size
    step = max(1, len(df) // max_rows)
    
    # If we have Ticker column, try to maintain distribution per ticker
    if 'Ticker' in df.columns and len(df['Ticker'].unique()) > 1:
        # Sample proportionally from each ticker
        sampled_dfs = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker]
            ticker_max = max(1, int(max_rows * (len(ticker_df) / len(df))))
            if len(ticker_df) > ticker_max:
                ticker_step = max(1, len(ticker_df) // ticker_max)
                sampled_dfs.append(ticker_df.iloc[::ticker_step].copy())
            else:
                sampled_dfs.append(ticker_df.copy())
        
        result = pd.concat(sampled_dfs, ignore_index=True)
        
        # If still too large, take uniform sample
        if len(result) > max_rows:
            step = len(result) // max_rows
            result = result.iloc[::step].copy()
        
        return result
    else:
        # Simple uniform sampling
        return df.iloc[::step].copy()

