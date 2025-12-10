"""
Modeling module untuk aplikasi FinScope.
Menggunakan Prophet untuk time series forecasting dengan regressors.
UPGRADED: Log transformation, Technical indicators, Hyperparameter tuning, Outlier handling.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from joblib import Parallel, delayed
import streamlit as st
from utils import save_model, load_model, calculate_metrics
from datetime import datetime, timedelta
import itertools
import warnings
warnings.filterwarnings('ignore')


def remove_outliers(df: pd.DataFrame, return_threshold: float = 0.10):
    """
    Menghapus atau clip outliers berdasarkan return harian.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom Date dan Close
    return_threshold : float
        Threshold untuk return harian (default: 0.10 = 10%)
    
    Returns:
    --------
    pd.DataFrame : Dataframe dengan outliers yang sudah di-handle
    """
    df_clean = df.copy()
    
    if 'Close' in df_clean.columns and 'Date' in df_clean.columns:
        df_clean = df_clean.sort_values('Date')
        
        # Calculate daily return
        df_clean['Daily_Return'] = df_clean['Close'].pct_change()
        
        # Clip extreme returns (> 10% or < -10%)
        df_clean['Daily_Return'] = df_clean['Daily_Return'].clip(
            lower=-return_threshold,
            upper=return_threshold
        )
        
        # Reconstruct Close price from clipped returns
        df_clean['Close'] = df_clean['Close'].iloc[0] * (1 + df_clean['Daily_Return']).cumprod()
        
        # Drop temporary column
        df_clean = df_clean.drop(columns=['Daily_Return'], errors='ignore')
    
    return df_clean


def calculate_technical_indicators(df: pd.DataFrame):
    """
    Menghitung indikator teknikal: RSI_14, SMA_20, SMA_50.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom Date dan Close
    
    Returns:
    --------
    pd.DataFrame : Dataframe dengan technical indicators
    """
    df_tech = df.copy()
    
    if 'Close' not in df_tech.columns:
        return df_tech
    
    df_tech = df_tech.sort_values('Date')
    
    # SMA_20 (Simple Moving Average 20 days)
    df_tech['SMA_20'] = df_tech['Close'].rolling(window=20, min_periods=1).mean()
    
    # SMA_50 (Simple Moving Average 50 days)
    df_tech['SMA_50'] = df_tech['Close'].rolling(window=50, min_periods=1).mean()
    
    # RSI_14 (Relative Strength Index 14 days)
    delta = df_tech['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
    df_tech['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Fill NaN values dengan forward fill
    df_tech['RSI_14'] = df_tech['RSI_14'].fillna(50)  # RSI default = 50 (neutral)
    df_tech['SMA_20'] = df_tech['SMA_20'].ffill().bfill()
    df_tech['SMA_50'] = df_tech['SMA_50'].ffill().bfill()
    
    return df_tech


def prepare_prophet_data(df: pd.DataFrame, ticker: str = None, 
                         use_log_transform: bool = True,
                         add_technical_indicators: bool = True):
    """
    Mempersiapkan data untuk Prophet (format: ds, y, regressors).
    UPGRADED: Log transformation, Technical indicators, Outlier handling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom Date dan Close
    ticker : str
        Ticker symbol untuk filter data (optional)
    use_log_transform : bool
        Apakah akan menggunakan log transformation (default: True)
    add_technical_indicators : bool
        Apakah akan menambahkan technical indicators (default: True)
    
    Returns:
    --------
    pd.DataFrame : Dataframe dengan format Prophet (ds, y, regressors)
    dict : Metadata tentang transformasi (untuk inverse transform)
    """
    if ticker and 'Ticker' in df.columns:
        df_ticker = df[df['Ticker'] == ticker].copy()
    else:
        df_ticker = df.copy()
    
    # Pastikan Date adalah datetime
    df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
    df_ticker = df_ticker.sort_values('Date')
    
    # Remove outliers sebelum transformasi
    df_ticker = remove_outliers(df_ticker, return_threshold=0.10)
    
    # Calculate technical indicators
    if add_technical_indicators:
        df_ticker = calculate_technical_indicators(df_ticker)
    
    # Format untuk Prophet: ds (date) dan y (target)
    prophet_df = pd.DataFrame({
        'ds': df_ticker['Date'],
        'y': df_ticker['Close']
    })
    
    # Log transformation untuk y (target)
    metadata = {'use_log_transform': use_log_transform}
    if use_log_transform:
        # Store original min untuk validasi
        metadata['original_min'] = prophet_df['y'].min()
        prophet_df['y'] = np.log1p(prophet_df['y'])
        metadata['log_min'] = prophet_df['y'].min()
    
    # Tambahkan fundamental regressors jika ada
    fundamental_regressors = ['ROE', 'Debt_Equity', 'Debt_Equity_Ratio', 'EBIT_Margin']
    for col in fundamental_regressors:
        if col in df_ticker.columns:
            # Fill null dengan forward fill
            prophet_df[col] = df_ticker[col].ffill().bfill()
    
    # Tambahkan technical regressors jika ada
    if add_technical_indicators:
        technical_regressors = ['RSI_14', 'SMA_20', 'SMA_50']
        for col in technical_regressors:
            if col in df_ticker.columns:
                prophet_df[col] = df_ticker[col].ffill().bfill()
    
    return prophet_df, metadata


def optimize_prophet_params(prophet_df: pd.DataFrame, 
                           initial_periods: int = 365,
                           period: int = 30,
                           horizon: int = 30):
    """
    Hyperparameter tuning menggunakan Grid Search dengan cross-validation.
    
    Parameters:
    -----------
    prophet_df : pd.DataFrame
        Dataframe dengan format Prophet (ds, y, regressors)
    initial_periods : int
        Initial training period untuk cross-validation (default: 365 days)
    period : int
        Period antara validasi (default: 30 days)
    horizon : int
        Horizon untuk validasi (default: 30 days)
    
    Returns:
    --------
    dict : Best parameters dengan RMSE terendah
    """
    # Parameter grid untuk grid search
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    # Generate all combinations
    all_params = [dict(zip(param_grid.keys(), v)) 
                   for v in itertools.product(*param_grid.values())]
    
    # Store results
    results = []
    
    # Minimum data untuk cross-validation
    min_data_points = initial_periods + horizon
    if len(prophet_df) < min_data_points:
        # Jika data tidak cukup, gunakan default parameters
        return {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive'
        }
    
    # Limit jumlah kombinasi untuk performa (max 20 kombinasi)
    max_combinations = 20
    if len(all_params) > max_combinations:
        # Sample secara random atau pilih yang paling promising
        import random
        all_params = random.sample(all_params, max_combinations)
    
    # Try each parameter combination
    for params in all_params:
        try:
            # Get regressor columns
            regressor_cols = [col for col in prophet_df.columns 
                            if col not in ['ds', 'y']]
            
            # Initialize Prophet dengan parameter
            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )
            
            # Add regressors
            for col in regressor_cols:
                model.add_regressor(col)
            
            # Fit model
            model.fit(prophet_df)
            
            # Cross-validation
            try:
                df_cv = cross_validation(
                    model,
                    initial=f'{initial_periods} days',
                    period=f'{period} days',
                    horizon=f'{horizon} days',
                    disable_tqdm=True
                )
                
                # Calculate RMSE
                df_perf = performance_metrics(df_cv)
                rmse = df_perf['rmse'].mean()
                
                results.append({
                    'params': params,
                    'rmse': rmse
                })
            except Exception as e:
                # Skip jika cross-validation gagal
                continue
                
        except Exception as e:
            # Skip jika training gagal
            continue
    
    # Pilih parameter dengan RMSE terendah
    if results:
        best_result = min(results, key=lambda x: x['rmse'])
        return best_result['params']
    else:
        # Fallback ke default jika semua gagal
        return {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive'
        }


def train_prophet_model(prophet_df: pd.DataFrame, ticker: str, 
                        add_regressors: bool = True,
                        use_hyperparameter_tuning: bool = True,
                        changepoint_prior_scale: float = 0.05,
                        seasonality_prior_scale: float = 10.0,
                        seasonality_mode: str = 'additive'):
    """
    Train Prophet model untuk satu ticker.
    UPGRADED: Hyperparameter tuning, Technical indicators support.
    
    Parameters:
    -----------
    prophet_df : pd.DataFrame
        Dataframe dengan format Prophet (ds, y, regressors)
    ticker : str
        Ticker symbol
    add_regressors : bool
        Apakah akan menambahkan regressors
    use_hyperparameter_tuning : bool
        Apakah akan menggunakan hyperparameter tuning (default: True)
    changepoint_prior_scale : float
        Prior scale untuk changepoints (default: 0.05)
    seasonality_prior_scale : float
        Prior scale untuk seasonality (default: 10.0)
    seasonality_mode : str
        Mode seasonality ('additive' atau 'multiplicative')
    
    Returns:
    --------
    Prophet : Trained Prophet model
    dict : Metadata tentang transformasi
    """
    # Hyperparameter tuning jika diminta
    if use_hyperparameter_tuning and len(prophet_df) >= 400:  # Minimum data untuk CV
        try:
            best_params = optimize_prophet_params(prophet_df)
            changepoint_prior_scale = best_params['changepoint_prior_scale']
            seasonality_prior_scale = best_params['seasonality_prior_scale']
            seasonality_mode = best_params['seasonality_mode']
        except Exception as e:
            # Fallback ke default jika tuning gagal
            pass
    
    # Initialize Prophet dengan parameter
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95  # 95% confidence interval
    )
    
    # Get regressor columns (fundamental + technical)
    regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
    
    # Add regressors jika ada dan diminta
    if add_regressors:
        for col in regressor_cols:
            model.add_regressor(col)
    
    # Fit model
    model.fit(prophet_df)
    
    # Store metadata
    metadata = {
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale,
        'seasonality_mode': seasonality_mode,
        'regressors': regressor_cols
    }
    
    # Attach metadata to model (custom attribute)
    model.metadata = metadata
    
    return model, metadata


def forecast_prophet(model: Prophet, periods: int = 90, 
                    future_regressors: pd.DataFrame = None,
                    metadata: dict = None):
    """
    Membuat forecast menggunakan Prophet model.
    UPGRADED: Log transformation inverse, Technical regressors fill forward.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    periods : int
        Jumlah hari ke depan untuk forecast (default: 90)
    future_regressors : pd.DataFrame
        Dataframe dengan regressor values untuk periode forecast
    metadata : dict
        Metadata tentang transformasi (untuk inverse log transform)
    
    Returns:
    --------
    pd.DataFrame : Forecast dataframe dengan kolom ds, yhat, yhat_lower, yhat_upper
    """
    # Buat future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Tambahkan regressors untuk periode future jika ada
    if future_regressors is not None:
        regressor_cols = [col for col in future_regressors.columns if col != 'ds']
        for col in regressor_cols:
            if col in model.regressors:
                # Merge dengan future dataframe
                future = future.merge(
                    future_regressors[['ds', col]], 
                    on='ds', 
                    how='left'
                )
                # Fill forward dari nilai terakhir atau rata-rata bergerak 3 hari terakhir
                if len(future_regressors) >= 3:
                    # Gunakan rata-rata bergerak 3 hari terakhir
                    last_3_avg = future_regressors[col].tail(3).mean()
                    future[col] = future[col].fillna(last_3_avg)
                else:
                    # Forward fill dari nilai terakhir
                    future[col] = future[col].ffill().bfill()
    
    # Predict
    forecast = model.predict(future)
    
    # Inverse log transformation jika digunakan
    if metadata and metadata.get('use_log_transform', False):
        forecast['yhat'] = np.expm1(forecast['yhat'])
        forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
        forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
    
    return forecast


def train_single_ticker(args):
    """
    Helper function untuk parallel training per ticker.
    UPGRADED: Log transformation, Technical indicators, Hyperparameter tuning, Outlier handling.
    
    Parameters:
    -----------
    args : tuple
        (df, ticker, split_date, add_regressors, use_hyperparameter_tuning)
    
    Returns:
    --------
    dict : Dictionary berisi model, metrics, dan info ticker
    """
    df, ticker, split_date, add_regressors, use_hyperparameter_tuning = args
    
    try:
        # Prepare data dengan log transformation dan technical indicators
        prophet_df, metadata = prepare_prophet_data(
            df, 
            ticker=ticker,
            use_log_transform=True,
            add_technical_indicators=True
        )
        
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
        
        # Train model dengan hyperparameter tuning
        model, model_metadata = train_prophet_model(
            train_df, 
            ticker, 
            add_regressors=add_regressors,
            use_hyperparameter_tuning=use_hyperparameter_tuning
        )
        
        # Forecast untuk test period
        # Buat future dataframe untuk test period
        test_future = test_df[['ds']].copy()
        
        # Tambahkan regressors untuk test period jika ada
        if add_regressors:
            regressor_cols = [col for col in test_df.columns if col not in ['ds', 'y']]
            if regressor_cols:
                for col in regressor_cols:
                    test_future[col] = test_df[col].values
        
        # Predict untuk test period
        test_forecast = model.predict(test_future)
        
        # Inverse log transformation jika digunakan
        if metadata.get('use_log_transform', False):
            test_forecast['yhat'] = np.expm1(test_forecast['yhat'])
            test_forecast['yhat_lower'] = np.expm1(test_forecast['yhat_lower'])
            test_forecast['yhat_upper'] = np.expm1(test_forecast['yhat_upper'])
        
        # Calculate metrics (inverse log transform untuk y_true jika perlu)
        y_true = test_df['y'].values
        if metadata.get('use_log_transform', False):
            y_true = np.expm1(y_true)  # Inverse log transform
        
        y_pred = test_forecast['yhat'].values[:len(y_true)]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Save model dengan metadata
        save_model(model, ticker)
        
        # Save metadata separately (optional, bisa juga disimpan di model)
        return {
            'ticker': ticker,
            'success': True,
            'model': model,
            'metadata': metadata,
            'model_metadata': model_metadata,
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
                          add_regressors: bool = True, 
                          use_hyperparameter_tuning: bool = True,
                          n_jobs: int = -1,
                          tickers_to_train: list = None):
    """
    Train Prophet models untuk semua ticker secara parallel.
    UPGRADED: Hyperparameter tuning support, support untuk filter ticker tertentu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan data keuangan
    split_date : str
        Tanggal split untuk train/test (default: '2021-01-01')
    add_regressors : bool
        Apakah akan menambahkan regressors
    use_hyperparameter_tuning : bool
        Apakah akan menggunakan hyperparameter tuning (default: True)
    n_jobs : int
        Jumlah parallel jobs (-1 untuk semua cores)
    tickers_to_train : list
        List ticker yang ingin di-train (jika None, akan train semua ticker di df)
    
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
        # Jika tickers_to_train diberikan, gunakan itu (user sudah filter)
        if tickers_to_train is not None and len(tickers_to_train) > 0:
            tickers = tickers_to_train
        else:
            # Jika tidak ada filter, ambil semua ticker dari df
            tickers = df['Ticker'].unique().tolist()
    
    # Hanya limit jika user TIDAK filter ticker tertentu (tickers_to_train is None)
    # Jika user sudah filter, train semua ticker yang dipilih tanpa limit
    if tickers_to_train is None:
        max_tickers = 10  # Limit maksimal 10 tickers sekaligus jika tidak ada filter
        if len(tickers) > max_tickers:
            tickers = tickers[:max_tickers]
            try:
                st.warning(f"⚠️ Terlalu banyak tickers. Hanya training {max_tickers} tickers pertama.")
            except:
                pass
    
    split_date = pd.Timestamp(split_date)
    
    # Prepare arguments untuk parallel processing
    args_list = [
        (df, ticker, split_date, add_regressors, use_hyperparameter_tuning)
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
            status_text.text(f"Training {len(tickers)} model(s) dengan hyperparameter tuning...")
        
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
                df, ticker, _, _, _ = args
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
    UPGRADED: Log transformation inverse, Technical regressors dengan fill forward.
    
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
            try:
                st.error(f"Model untuk {ticker} tidak ditemukan. Silakan train model terlebih dahulu.")
            except:
                pass
            return None, None
    
    # Prepare data untuk mendapatkan regressor values terakhir
    prophet_df, metadata = prepare_prophet_data(
        df, 
        ticker=ticker,
        use_log_transform=True,
        add_technical_indicators=True
    )
    
    # Buat future regressors (gunakan nilai terakhir atau rata-rata bergerak 3 hari terakhir)
    future_regressors = None
    if add_regressors and len(prophet_df) > 0:
        # Get all regressor columns (fundamental + technical)
        regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
        
        if regressor_cols:
            # Ambil 3 hari terakhir untuk rata-rata bergerak
            last_3_days = prophet_df[['ds'] + regressor_cols].tail(3).copy()
            
            # Buat dataframe untuk periode future
            future_dates = pd.date_range(
                start=prophet_df['ds'].max() + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            future_regressors = pd.DataFrame({'ds': future_dates})
            
            # Fill regressors dengan rata-rata bergerak 3 hari terakhir atau nilai terakhir
            for col in regressor_cols:
                if len(last_3_days) >= 3:
                    # Gunakan rata-rata bergerak 3 hari terakhir
                    avg_value = last_3_days[col].mean()
                else:
                    # Gunakan nilai terakhir
                    avg_value = last_3_days[col].iloc[-1]
                
                future_regressors[col] = avg_value
    
    # Forecast dengan inverse log transformation
    forecast = forecast_prophet(
        model, 
        periods=periods, 
        future_regressors=future_regressors,
        metadata=metadata
    )
    
    return forecast, model
