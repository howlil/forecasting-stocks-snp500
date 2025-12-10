"""
ETL (Extract, Transform, Load) module untuk aplikasi FinScope.
Menangani loading CSV dengan chunking, cleaning, dan feature engineering.
"""

import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


def load_csv_chunked(file_path: str = None, file_content: bytes = None, chunk_size: int = 100000):
    """
    Load CSV file dengan chunking untuk handle jutaan rows.
    
    Parameters:
    -----------
    file_path : str
        Path ke file CSV (optional jika file_content diberikan)
    file_content : bytes
        Content file dalam bentuk bytes (optional jika file_path diberikan)
    chunk_size : int
        Ukuran chunk untuk membaca file (default: 100k rows)
    
    Returns:
    --------
    pd.DataFrame : Dataframe gabungan dari semua chunks
    """
    chunks = []
    try:
        import io
        
        # Gunakan file_content jika ada, jika tidak gunakan file_path
        if file_content is not None:
            # Baca dari bytes langsung
            file_obj = io.BytesIO(file_content)
            source = file_obj
        elif file_path is not None:
            # Check if file exists menggunakan pathlib
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                st.error(f"File tidak ditemukan: {file_path}")
                return pd.DataFrame()
            source = str(file_path)
        else:
            st.error("Harus memberikan file_path atau file_content")
            return pd.DataFrame()
        
        # Try parsing Date column, jika gagal akan di-handle
        try:
            for chunk in pd.read_csv(source, chunksize=chunk_size, parse_dates=['Date']):
                chunks.append(chunk)
        except (KeyError, ValueError):
            # Jika Date tidak ada atau tidak bisa di-parse, baca tanpa parse_dates
            if file_content is not None:
                file_obj = io.BytesIO(file_content)
                source = file_obj
            for chunk in pd.read_csv(source, chunksize=chunk_size):
                if 'Date' in chunk.columns:
                    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
                chunks.append(chunk)
        except Exception as e:
            st.error(f"Error membaca CSV: {str(e)}")
            return pd.DataFrame()
        
        if not chunks:
            st.warning("Tidak ada data yang berhasil dibaca dari file.")
            return pd.DataFrame()
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Pastikan kolom Date ada dan dalam format datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        return df
    except Exception as e:
        import traceback
        st.error(f"Error loading CSV: {str(e)}")
        return pd.DataFrame()


def clean_nulls(df: pd.DataFrame, method: str = 'ffill'):
    """
    Membersihkan null values dengan forward fill atau backward fill per ticker/quarter.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe yang akan dibersihkan
    method : str
        Metode cleaning: 'ffill' (forward fill) atau 'bfill' (backward fill)
    
    Returns:
    --------
    pd.DataFrame : Dataframe yang sudah dibersihkan
    """
    if df.empty:
        return df
    
    # Convert to standard pandas types to avoid pyarrow issues
    df = df.copy()
    
    # Convert any pyarrow/extension arrays to standard numpy arrays
    for col in df.columns:
        dtype_str = str(df[col].dtype).lower()
        if ('arrow' in dtype_str or 
            'extension' in dtype_str or 
            'nullable' in dtype_str or
            hasattr(df[col].dtype, 'name') and ('Int' in str(df[col].dtype) or 'Float' in str(df[col].dtype))):
            # Convert to standard type
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(str)
    
    # Pastikan Date adalah datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Sort by Date dan Ticker
    if 'Ticker' in df.columns:
        df = df.sort_values(['Ticker', 'Date'])
        
        # Group by Ticker dan Quarter untuk cleaning yang lebih granular
        df['Quarter'] = df['Date'].dt.to_period('Q')
        
        # Forward fill per ticker dan quarter
        if method == 'ffill':
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df.groupby(['Ticker', 'Quarter'])[col].ffill()
                    df[col] = df.groupby('Ticker')[col].ffill()
        elif method == 'bfill':
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df.groupby(['Ticker', 'Quarter'])[col].bfill()
                    df[col] = df.groupby('Ticker')[col].bfill()
        
        # Drop kolom Quarter temporary
        df = df.drop('Quarter', axis=1)
    else:
        # Jika tidak ada Ticker, lakukan fillna biasa
        if method == 'ffill':
            df = df.ffill()
        elif method == 'bfill':
            df = df.bfill()
    
    return df


def calculate_daily_returns(df: pd.DataFrame):
    """
    Menghitung daily returns dari harga Close.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom 'Close'
    
    Returns:
    --------
    pd.DataFrame : Dataframe dengan kolom tambahan 'Daily_Return'
    """
    df = df.copy()
    
    if 'Close' not in df.columns:
        return df
    
    if 'Ticker' in df.columns:
        df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    else:
        df['Daily_Return'] = df['Close'].pct_change()
    
    return df


def calculate_volatility(df: pd.DataFrame, window: int = 30):
    """
    Menghitung rolling volatility (standard deviation) dari daily returns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom 'Daily_Return'
    window : int
        Window size untuk rolling calculation (default: 30 hari)
    
    Returns:
    --------
    pd.DataFrame : Dataframe dengan kolom tambahan 'Volatility_30d'
    """
    df = df.copy()
    
    if 'Daily_Return' not in df.columns:
        df = calculate_daily_returns(df)
    
    # Ensure Daily_Return is numeric
    if not pd.api.types.is_numeric_dtype(df['Daily_Return']):
        df['Daily_Return'] = pd.to_numeric(df['Daily_Return'], errors='coerce')
    else:
        df['Daily_Return'] = df['Daily_Return'].astype('float64')
    
    if 'Ticker' in df.columns:
        volatility = df.groupby('Ticker')['Daily_Return'].rolling(
            window=window, min_periods=1
        ).std().reset_index(0, drop=True)
        df[f'Volatility_{window}d'] = volatility.astype('float64')
    else:
        df[f'Volatility_{window}d'] = df['Daily_Return'].rolling(
            window=window, min_periods=1
        ).std().astype('float64')
    
    return df


def scale_close_price(df: pd.DataFrame):
    """
    Scale kolom Close menggunakan MinMaxScaler per ticker.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom 'Close'
    
    Returns:
    --------
    pd.DataFrame : Dataframe dengan kolom tambahan 'Close_Scaled'
    tuple : (scaler_dict, inverse_transform function)
    """
    df = df.copy()
    
    if 'Close' not in df.columns:
        return df, None
    
    # Ensure Close is numeric and convert to standard float64 to avoid pyarrow issues
    if not pd.api.types.is_numeric_dtype(df['Close']):
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Convert to numpy array explicitly to avoid extension array issues
    df['Close'] = np.array(df['Close'].values, dtype='float64')
    
    scalers = {}
    
    if 'Ticker' in df.columns:
        df['Close_Scaled'] = np.nan
        
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            # Convert to numpy array explicitly to avoid pyarrow issues
            ticker_data = np.array(df.loc[ticker_mask, 'Close'].values, dtype='float64')
            ticker_data = ticker_data.reshape(-1, 1)
            
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(ticker_data)
            df.loc[ticker_mask, 'Close_Scaled'] = scaled_values.flatten()
            scalers[ticker] = scaler
    else:
        # Convert to numpy array explicitly
        close_data = np.array(df['Close'].values, dtype='float64').reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(close_data)
        df['Close_Scaled'] = scaled_values.flatten()
        scalers['default'] = scaler
    
    # Ensure Close_Scaled is float64
    df['Close_Scaled'] = np.array(df['Close_Scaled'].values, dtype='float64')
    
    def inverse_transform(ticker, scaled_values):
        """Helper function untuk inverse transform."""
        scaled_values = np.array(scaled_values, dtype='float64')
        if ticker in scalers:
            return scalers[ticker].inverse_transform(scaled_values.reshape(-1, 1)).flatten()
        elif 'default' in scalers:
            return scalers['default'].inverse_transform(scaled_values.reshape(-1, 1)).flatten()
        return scaled_values
    
    return df, (scalers, inverse_transform)


def process_etl_from_df(df: pd.DataFrame, clean_method: str = 'ffill', 
                        calculate_features: bool = True, scale_close: bool = True,
                        progress_bar=None, status_text=None, metrics_container=None):
    """
    Process ETL langsung dari dataframe (untuk menghindari MemoryError).
    Menggunakan memory-efficient processing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe yang sudah di-load
    clean_method : str
        Metode cleaning: 'ffill' atau 'bfill'
    calculate_features : bool
        Apakah akan menghitung derived features (returns, volatility)
    scale_close : bool
        Apakah akan scale kolom Close
    
    Returns:
    --------
    pd.DataFrame : Dataframe yang sudah diproses
    dict : Dictionary berisi metadata (scalers, stats, dll)
    """
    import gc
    
    metadata = {}
    
    # Use provided progress bar and status text, or create new ones
    use_provided_progress = progress_bar is not None
    use_provided_status = status_text is not None
    
    if progress_bar is None:
        try:
            progress_bar = st.progress(0)
        except Exception:
            progress_bar = None
    
    if status_text is None:
        try:
            status_text = st.empty()
        except Exception:
            status_text = None
    
    # Step 1: Data sudah di-load
    try:
        if status_text is not None:
            status_text.text("Data loaded, starting processing...")
        if progress_bar is not None:
            progress_bar.progress(5)
        time.sleep(0.05)  # Jeda untuk memastikan update terkirim
    except Exception:
        pass
    
    if df.empty:
        st.error("Dataframe kosong!")
        return df, metadata
    
    metadata['original_rows'] = len(df)
    metadata['original_columns'] = list(df.columns)
    
    # Step 2: Clean nulls (process in chunks if large)
    try:
        if status_text is not None:
            status_text.text("🧹 Cleaning null values...")
        if progress_bar is not None:
            progress_bar.progress(20)
        time.sleep(0.05)  # Jeda untuk memastikan update terkirim
    except Exception:
        pass
    
    # Convert extension arrays before processing to avoid pyarrow issues
    total_cols = len(df.columns)
    for idx, col in enumerate(df.columns):
        dtype_str = str(df[col].dtype).lower()
        if ('arrow' in dtype_str or 'extension' in dtype_str or 'nullable' in dtype_str):
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Update progress for column conversion dengan jeda
        if (idx + 1) % 5 == 0 and status_text is not None:
            try:
                status_text.text(f"🧹 Cleaning null values... ({idx+1}/{total_cols} columns)")
                time.sleep(0.01)  # Jeda kecil untuk mencegah WebSocket timeout
            except Exception:
                pass
    
    if len(df) > 500000:
        # Process in chunks for very large dataframes
        chunk_size = 200000
        cleaned_chunks = []
        total_chunks = (len(df) // chunk_size) + 1
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunk = clean_nulls(chunk, method=clean_method)
            cleaned_chunks.append(chunk)
            
            # Update progress dengan jeda untuk mencegah WebSocket timeout
            chunk_num = (i // chunk_size) + 1
            if status_text is not None:
                try:
                    status_text.text(f"🧹 Cleaning null values... Chunk {chunk_num}/{total_chunks}")
                except Exception:
                    pass
            if progress_bar is not None:
                try:
                    progress = 20 + int((chunk_num / total_chunks) * 15)
                    progress_bar.progress(progress)
                except Exception:
                    pass
            
            # Beri kesempatan Streamlit untuk mengirim update ke browser
            if chunk_num % 5 == 0:  # Setiap 5 chunks
                time.sleep(0.05)  # Jeda kecil untuk mencegah WebSocket timeout
            
            gc.collect()
        
        df = pd.concat(cleaned_chunks, ignore_index=True)
        del cleaned_chunks
        gc.collect()
    else:
        df = clean_nulls(df, method=clean_method)
    
    try:
        if progress_bar is not None:
            progress_bar.progress(40)
        time.sleep(0.05)  # Jeda untuk memastikan update terkirim
    except Exception:
        pass
    
    # Check null percentage
    from utils import check_null_percentage
    null_info = check_null_percentage(df)
    metadata['null_info'] = null_info
    
    if null_info['has_warning']:
        st.warning(null_info['warning_message'])
    
    # Step 3: Calculate derived features (process in chunks if large)
    if calculate_features:
        try:
            if status_text is not None:
                status_text.text("📊 Calculating derived features (returns, volatility)...")
            if progress_bar is not None:
                progress_bar.progress(50)
            time.sleep(0.05)  # Jeda untuk memastikan update terkirim
        except Exception:
            pass
        
        if len(df) > 500000:
            # Process in chunks
            chunk_size = 200000
            feature_chunks = []
            total_chunks = (len(df) // chunk_size) + 1
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size].copy()
                chunk = calculate_daily_returns(chunk)
                chunk = calculate_volatility(chunk, window=30)
                feature_chunks.append(chunk)
                
                # Update progress dengan jeda untuk mencegah WebSocket timeout
                chunk_num = (i // chunk_size) + 1
                if status_text is not None:
                    try:
                        status_text.text(f"📊 Calculating features... Chunk {chunk_num}/{total_chunks}")
                    except Exception:
                        pass
                if progress_bar is not None:
                    try:
                        progress = 50 + int((chunk_num / total_chunks) * 15)
                        progress_bar.progress(progress)
                    except Exception:
                        pass
                
                # Beri kesempatan Streamlit untuk mengirim update ke browser
                if chunk_num % 5 == 0:  # Setiap 5 chunks
                    time.sleep(0.05)  # Jeda kecil untuk mencegah WebSocket timeout
                
                gc.collect()
            
            df = pd.concat(feature_chunks, ignore_index=True)
            del feature_chunks
            gc.collect()
        else:
            df = calculate_daily_returns(df)
            df = calculate_volatility(df, window=30)
        
        try:
            if progress_bar is not None:
                progress_bar.progress(70)
            time.sleep(0.05)  # Jeda untuk memastikan update terkirim
        except Exception:
            pass
    
    # Step 4: Scale Close price (skip for very large datasets to save memory)
    scalers_info = None
    if scale_close and len(df) <= 2000000:  # Only scale if not too large
        try:
            if status_text is not None:
                status_text.text("Scaling Close price...")
            if progress_bar is not None:
                progress_bar.progress(85)
            time.sleep(0.05)  # Jeda untuk memastikan update terkirim
        except Exception:
            pass
        df, scalers_info = scale_close_price(df)
        metadata['scalers'] = scalers_info
    else:
        if scale_close:
            st.info("⚠️ Scaling di-skip untuk dataset besar (>2M rows) untuk menghemat memory.")
        try:
            if progress_bar is not None:
                progress_bar.progress(85)
            time.sleep(0.05)  # Jeda untuk memastikan update terkirim
        except Exception:
            pass
    
    # Step 5: Final stats
    try:
        if status_text is not None:
            status_text.text("✅ Finalizing...")
        if progress_bar is not None:
            progress_bar.progress(95)
        time.sleep(0.05)  # Jeda untuk memastikan update terkirim
    except Exception:
        pass
    
    metadata['final_rows'] = len(df)
    metadata['final_columns'] = list(df.columns)
    
    # Update metrics container if provided
    if metrics_container is not None:
        try:
            with metrics_container.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    if 'Ticker' in df.columns:
                        st.metric("Tickers", len(df['Ticker'].unique()))
        except Exception:
            pass
    
    try:
        if 'Date' in df.columns:
            metadata['date_range'] = {
                'min': str(df['Date'].min()),
                'max': str(df['Date'].max())
            }
        else:
            metadata['date_range'] = {'min': None, 'max': None}
    except Exception:
        metadata['date_range'] = {'min': None, 'max': None}
    
    if 'Ticker' in df.columns:
        try:
            metadata['tickers'] = sorted(df['Ticker'].unique().tolist())
            metadata['num_tickers'] = len(metadata['tickers'])
        except Exception:
            metadata['tickers'] = []
            metadata['num_tickers'] = 0
    
    try:
        if progress_bar is not None:
            progress_bar.progress(100)
            time.sleep(0.1)  # Jeda sebelum cleanup
            progress_bar.empty()
        if status_text is not None:
            status_text.empty()
    except Exception:
        pass
    
    # Final cleanup
    gc.collect()
    
    return df, metadata


def process_etl(file_path: str = None, file_content: bytes = None, clean_method: str = 'ffill', 
                calculate_features: bool = True, scale_close: bool = True):
    """
    Fungsi utama ETL yang menggabungkan semua proses.
    
    Parameters:
    -----------
    file_path : str
        Path ke file CSV (optional jika file_content diberikan)
    file_content : bytes
        Content file dalam bentuk bytes (optional jika file_path diberikan)
    clean_method : str
        Metode cleaning: 'ffill' atau 'bfill'
    calculate_features : bool
        Apakah akan menghitung derived features (returns, volatility)
    scale_close : bool
        Apakah akan scale kolom Close
    
    Returns:
    --------
    pd.DataFrame : Dataframe yang sudah diproses
    dict : Dictionary berisi metadata (scalers, stats, dll)
    """
    metadata = {}
    
    # Progress bar with error handling
    progress_bar = None
    status_text = None
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
    except Exception:
        # If progress bar creation fails, continue without it
        pass
    
    # Step 1: Load CSV
    try:
        if status_text is not None:
            status_text.text("Loading CSV file...")
        if progress_bar is not None:
            progress_bar.progress(10)
    except Exception:
        pass
    df = load_csv_chunked(file_path=file_path, file_content=file_content)
    
    if df.empty:
        st.error("Dataframe kosong setelah loading!")
        return df, metadata
    
    metadata['original_rows'] = len(df)
    metadata['original_columns'] = list(df.columns)
    
    # Step 2: Clean nulls
    try:
        if status_text is not None:
            status_text.text("Cleaning null values...")
        if progress_bar is not None:
            progress_bar.progress(30)
    except Exception:
        pass
    df = clean_nulls(df, method=clean_method)
    
    # Check null percentage
    from utils import check_null_percentage
    null_info = check_null_percentage(df)
    metadata['null_info'] = null_info
    
    if null_info['has_warning']:
        st.warning(null_info['warning_message'])
    
    # Step 3: Calculate derived features
    if calculate_features:
        try:
            if status_text is not None:
                status_text.text("Calculating derived features (returns, volatility)...")
            if progress_bar is not None:
                progress_bar.progress(50)
        except Exception:
            pass
        df = calculate_daily_returns(df)
        df = calculate_volatility(df, window=30)
    
    # Step 4: Scale Close price
    scalers_info = None
    if scale_close:
        try:
            if status_text is not None:
                status_text.text("Scaling Close price...")
            if progress_bar is not None:
                progress_bar.progress(80)
        except Exception:
            pass
        df, scalers_info = scale_close_price(df)
        metadata['scalers'] = scalers_info
    
    # Step 5: Final stats
    try:
        if status_text is not None:
            status_text.text("Finalizing...")
        if progress_bar is not None:
            progress_bar.progress(100)
    except Exception:
        pass
    
    metadata['final_rows'] = len(df)
    metadata['final_columns'] = list(df.columns)
    metadata['date_range'] = {
        'min': str(df['Date'].min()) if 'Date' in df.columns else None,
        'max': str(df['Date'].max()) if 'Date' in df.columns else None
    }
    
    if 'Ticker' in df.columns:
        metadata['tickers'] = sorted(df['Ticker'].unique().tolist())
        metadata['num_tickers'] = len(metadata['tickers'])
    
    try:
        if progress_bar is not None:
            progress_bar.empty()
        if status_text is not None:
            status_text.empty()
    except Exception:
        pass
    
    return df, metadata

