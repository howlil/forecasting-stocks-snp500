"""
Modeling module untuk aplikasi FinScope.
Menggunakan Prophet untuk time series forecasting dengan regressors.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from joblib import Parallel, delayed
import streamlit as st
from utils import save_model, load_model, calculate_metrics
from datetime import datetime, timedelta


def prepare_prophet_data(df: pd.DataFrame, ticker: str = None):
    """
    Mempersiapkan data untuk Prophet (format: ds, y, regressors).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom Date dan Close
    ticker : str
        Ticker symbol untuk filter data (optional)
    
    Returns:
    --------
    pd.DataFrame : Dataframe dengan format Prophet (ds, y, regressors)
    """
    if ticker and 'Ticker' in df.columns:
        df_ticker = df[df['Ticker'] == ticker].copy()
    else:
        df_ticker = df.copy()
    
    # Pastikan Date adalah datetime
    df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
    df_ticker = df_ticker.sort_values('Date')
    
    # Format untuk Prophet: ds (date) dan y (target)
    prophet_df = pd.DataFrame({
        'ds': df_ticker['Date'],
        'y': df_ticker['Close']
    })
    
    # Tambahkan regressors jika ada
    regressor_cols = ['ROE', 'Debt_Equity', 'EBIT_Margin']
    for col in regressor_cols:
        if col in df_ticker.columns:
            # Fill null dengan forward fill
            prophet_df[col] = df_ticker[col].ffill().bfill()
    
    return prophet_df


def train_prophet_model(prophet_df: pd.DataFrame, ticker: str, 
                        add_regressors: bool = True, 
                        changepoint_prior_scale: float = 0.05,
                        seasonality_prior_scale: float = 10.0):
    """
    Train Prophet model untuk satu ticker.
    
    Parameters:
    -----------
    prophet_df : pd.DataFrame
        Dataframe dengan format Prophet (ds, y, regressors)
    ticker : str
        Ticker symbol
    add_regressors : bool
        Apakah akan menambahkan regressors (ROE, Debt_Equity)
    changepoint_prior_scale : float
        Prior scale untuk changepoints (default: 0.05 untuk lebih fleksibel)
    seasonality_prior_scale : float
        Prior scale untuk seasonality (default: 10.0)
    
    Returns:
    --------
    Prophet : Trained Prophet model
    """
    # Initialize Prophet dengan parameter
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95  # 95% confidence interval
    )
    
    # Tambahkan changepoints untuk events penting (contoh: 2008 crash)
    # Prophet akan otomatis detect changepoints, tapi kita bisa tambahkan manual
    changepoints = [
        pd.Timestamp('2008-09-15'),  # Lehman Brothers collapse
        pd.Timestamp('2020-03-15'),  # COVID-19 market crash
    ]
    
    # Add regressors jika ada dan diminta
    if add_regressors:
        if 'ROE' in prophet_df.columns:
            model.add_regressor('ROE')
        if 'Debt_Equity' in prophet_df.columns:
            model.add_regressor('Debt_Equity')
        if 'EBIT_Margin' in prophet_df.columns:
            model.add_regressor('EBIT_Margin')
    
    # Fit model
    model.fit(prophet_df)
    
    return model


def forecast_prophet(model: Prophet, periods: int = 90, 
                    future_regressors: pd.DataFrame = None):
    """
    Membuat forecast menggunakan Prophet model.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    periods : int
        Jumlah hari ke depan untuk forecast (default: 90)
    future_regressors : pd.DataFrame
        Dataframe dengan regressor values untuk periode forecast
    
    Returns:
    --------
    pd.DataFrame : Forecast dataframe dengan kolom ds, yhat, yhat_lower, yhat_upper
    """
    # Buat future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Tambahkan regressors untuk periode future jika ada
    if future_regressors is not None:
        regressor_cols = ['ROE', 'Debt_Equity', 'EBIT_Margin']
        for col in regressor_cols:
            if col in future_regressors.columns and col in model.regressors:
                # Merge dengan future dataframe berdasarkan tanggal terdekat
                future = future.merge(
                    future_regressors[['ds', col]], 
                    on='ds', 
                    how='left'
                )
                # Forward fill untuk missing values
                future[col] = future[col].ffill().bfill()
    
    # Predict
    forecast = model.predict(future)
    
    return forecast


def train_single_ticker(args):
    """
    Helper function untuk parallel training per ticker.
    
    Parameters:
    -----------
    args : tuple
        (df, ticker, split_date, add_regressors)
    
    Returns:
    --------
    dict : Dictionary berisi model, metrics, dan info ticker
    """
    df, ticker, split_date, add_regressors = args
    
    try:
        # Prepare data
        prophet_df = prepare_prophet_data(df, ticker=ticker)
        
        if len(prophet_df) < 30:  # Minimum data points
            return {
                'ticker': ticker,
                'success': False,
                'error': 'Insufficient data points'
            }
        
        # Split data berdasarkan tanggal
        train_df = prophet_df[prophet_df['ds'] < split_date].copy()
        test_df = prophet_df[prophet_df['ds'] >= split_date].copy()
        
        if len(train_df) < 30 or len(test_df) == 0:
            return {
                'ticker': ticker,
                'success': False,
                'error': 'Insufficient train/test data'
            }
        
        # Train model
        model = train_prophet_model(train_df, ticker, add_regressors=add_regressors)
        
        # Forecast untuk test period
        test_forecast = model.predict(test_df[['ds']])
        
        # Calculate metrics
        y_true = test_df['y'].values
        y_pred = test_forecast['yhat'].values[:len(y_true)]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Save model
        save_model(model, ticker)
        
        return {
            'ticker': ticker,
            'success': True,
            'model': model,
            'metrics': metrics,
            'train_size': len(train_df),
            'test_size': len(test_df)
        }
    
    except Exception as e:
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }


def train_models_parallel(df: pd.DataFrame, split_date: str = '2021-01-01',
                          add_regressors: bool = True, n_jobs: int = -1):
    """
    Train Prophet models untuk semua ticker secara parallel.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan data keuangan
    split_date : str
        Tanggal split untuk train/test (default: '2021-01-01')
    add_regressors : bool
        Apakah akan menambahkan regressors
    n_jobs : int
        Jumlah parallel jobs (-1 untuk semua cores)
    
    Returns:
    --------
    dict : Dictionary berisi hasil training per ticker
    """
    import gc
    import platform
    
    if 'Ticker' not in df.columns:
        # Single time series tanpa ticker
        tickers = ['ALL']
        df['Ticker'] = 'ALL'
    else:
        tickers = df['Ticker'].unique().tolist()
    
    # Limit jumlah ticker untuk training (prevent resource exhaustion)
    max_tickers = 10  # Limit maksimal 10 tickers sekaligus
    if len(tickers) > max_tickers:
        tickers = tickers[:max_tickers]
        import streamlit as st
        st.warning(f"⚠️ Terlalu banyak tickers. Hanya training {max_tickers} tickers pertama.")
    
    split_date = pd.Timestamp(split_date)
    
    # Prepare arguments untuk parallel processing
    args_list = [
        (df, ticker, split_date, add_regressors)
        for ticker in tickers
    ]
    
    # Progress tracking dengan error handling
    progress_bar = None
    status_text = None
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
    except Exception:
        pass
    
    # Parallel training dengan error handling
    results = []
    try:
        # Use backend='threading' untuk Windows (lebih stabil)
        backend = 'threading' if platform.system() == 'Windows' else 'loky'
        
        # Limit n_jobs untuk Windows
        if platform.system() == 'Windows' and n_jobs > 2:
            n_jobs = min(2, len(tickers))
        
        if status_text:
            status_text.text(f"Training {len(tickers)} model(s)...")
        
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(train_single_ticker)(args) for args in args_list
        )
        
        if status_text:
            status_text.text("Training completed!")
        
    except Exception as e:
        # Fallback to sequential jika parallel gagal
        if status_text:
            status_text.text(f"Parallel training failed, using sequential: {str(e)}")
        
        results = []
        for i, args in enumerate(args_list):
            try:
                result = train_single_ticker(args)
                results.append(result)
                if progress_bar:
                    progress_bar.progress((i + 1) / len(args_list))
                if status_text:
                    status_text.text(f"Training {i+1}/{len(args_list)}...")
            except Exception as e2:
                df, ticker, _, _ = args
                results.append({
                    'ticker': ticker,
                    'success': False,
                    'error': str(e2)
                })
    
    finally:
        # Cleanup progress bars
        try:
            if progress_bar:
                progress_bar.empty()
            if status_text:
                status_text.empty()
        except Exception:
            pass
        
        # Force garbage collection
        gc.collect()
    
    # Convert results ke dictionary
    results_dict = {}
    for result in results:
        if result and 'ticker' in result:
            ticker = result['ticker']
            results_dict[ticker] = result
    
    return results_dict


def forecast_future(df: pd.DataFrame, ticker: str, periods: int = 90,
                   model: Prophet = None, add_regressors: bool = True):
    """
    Membuat forecast untuk periode future.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan data historis
    ticker : str
        Ticker symbol
    periods : int
        Jumlah hari ke depan untuk forecast
    model : Prophet
        Model yang sudah di-train (jika None, akan load dari disk)
    add_regressors : bool
        Apakah model menggunakan regressors
    
    Returns:
    --------
    pd.DataFrame : Forecast dataframe
    Prophet : Model yang digunakan
    """
    # Load model jika tidak diberikan
    if model is None:
        model = load_model(ticker)
        if model is None:
            st.error(f"Model untuk {ticker} tidak ditemukan. Silakan train model terlebih dahulu.")
            return None, None
    
    # Prepare data untuk mendapatkan regressor values terakhir
    prophet_df = prepare_prophet_data(df, ticker=ticker)
    
    # Buat future regressors (gunakan nilai terakhir atau rata-rata)
    future_regressors = None
    if add_regressors and len(prophet_df) > 0:
        regressor_cols = ['ROE', 'Debt_Equity', 'EBIT_Margin']
        available_regressors = [col for col in regressor_cols if col in prophet_df.columns]
        
        if available_regressors:
            # Ambil nilai terakhir untuk regressors
            last_values = prophet_df[available_regressors].iloc[-1:].copy()
            
            # Buat dataframe untuk periode future
            future_dates = pd.date_range(
                start=prophet_df['ds'].max() + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            future_regressors = pd.DataFrame({'ds': future_dates})
            
            # Replicate nilai terakhir untuk semua periode future
            for col in available_regressors:
                future_regressors[col] = last_values[col].values[0]
    
    # Forecast
    forecast = forecast_prophet(model, periods=periods, future_regressors=future_regressors)
    
    return forecast, model

