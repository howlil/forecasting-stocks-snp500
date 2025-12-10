"""
Main Streamlit application untuk FinScope.
Aplikasi web untuk visualisasi dan forecasting data keuangan S&P 500.
"""

import os
import sys

# Set max upload size to 200MB
# This modifies sys.argv so Streamlit will read it during initialization
if len(sys.argv) > 1:
    # Check if maxUploadSize is already set
    has_max_upload = any('maxUploadSize' in arg for arg in sys.argv)
    if not has_max_upload:
        sys.argv.insert(1, '--server.maxUploadSize=200')
else:
    # If no arguments, add it
    sys.argv.append('--server.maxUploadSize=200')

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import platform
import gc

# Import modules
from etl import process_etl, process_etl_from_df
from modeling import train_models_parallel, forecast_future, prepare_prophet_data
from viz import (
    plot_candlestick_with_forecast,
    plot_prophet_decomposition,
    plot_correlation_heatmap,
    plot_roe_waterfall,
    plot_scenario_funnel,
    plot_radar_multi_ratio,
    plot_3d_surface_forecast,
    plot_3d_scatter_decomposition,
    plot_3d_funnel_scenarios,
    plot_3d_sensitivity_mesh,
    plot_3d_economic_terrain,
    plot_3d_leverage_vortex,
    plot_3d_cycle_globe,
    plot_3d_profit_prism,
    plot_3d_network_galaxy,
    plot_3d_forecast_terrain,
    plot_3d_ci_ribbon,
    plot_3d_component_orbit,
    plot_3d_waterfall_dupont,
    plot_3d_sensitivity_vortex,
    plot_fundamental_equalizer,
    plot_trend_decomposition_stack,
    plot_anomaly_traffic_light,
    plot_motion_quadrant,
    plot_scenario_simulator,
    plot_seasonal_heatmap_calendar,
    plot_financial_health_radar,
    plot_volume_pressure,
    plot_valuation_band,
    plot_neon_cyber_forecast,
    plot_forecast_bridge,
    plot_seasonal_compass,
    plot_risk_reward_motion,
    # New 3D visualizations
    plot_neon_time_tunnel,
    plot_decomposition_glass_stack,
    plot_seasonal_helix,
    plot_market_universe,
    plot_what_if_terrain,
    # New forecasting visualizations
    plot_fan_chart,
    plot_seasonal_heatmap_matrix,
    plot_regime_change
)
from utils import check_null_percentage, load_model
from model_evaluation import (
    evaluate_model_performance,
    calculate_comprehensive_metrics,
    plot_residual_analysis,
    plot_forecast_accuracy_by_horizon,
    generate_model_evaluation_report
)

# Safe wrapper functions untuk mencegah WebSocketClosedError
def safe_streamlit_call(func, *args, **kwargs):
    """Wrapper untuk semua fungsi Streamlit yang bisa menyebabkan WebSocket error."""
    try:
        return func(*args, **kwargs)
    except (Exception, RuntimeError, AttributeError) as e:
        # Ignore WebSocket errors dan error lainnya yang tidak critical
        error_type = type(e).__name__
        if 'WebSocket' not in error_type and 'StreamClosed' not in str(e):
            # Log non-websocket errors untuk debugging (optional)
            pass
        return None

def safe_error(message):
    """Safe wrapper untuk st.error."""
    return safe_streamlit_call(st.error, message)

def safe_warning(message):
    """Safe wrapper untuk st.warning."""
    return safe_streamlit_call(st.warning, message)

def safe_info(message):
    """Safe wrapper untuk st.info."""
    return safe_streamlit_call(st.info, message)

def safe_success(message):
    """Safe wrapper untuk st.success."""
    return safe_streamlit_call(st.success, message)

def safe_markdown(content, unsafe_allow_html=False):
    """Safe wrapper untuk st.markdown."""
    return safe_streamlit_call(st.markdown, content, unsafe_allow_html=unsafe_allow_html)

def safe_dataframe(data, width=None):
    """Safe wrapper untuk st.dataframe."""
    if width:
        return safe_streamlit_call(st.dataframe, data, width=width)
    return safe_streamlit_call(st.dataframe, data)

def safe_plotly_chart(fig, width=None):
    """Safe wrapper untuk st.plotly_chart."""
    if width:
        return safe_streamlit_call(st.plotly_chart, fig, width=width)
    return safe_streamlit_call(st.plotly_chart, fig)

# Page configuration
st.set_page_config(
    page_title="FinScope - Financial Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme sidebar
st.markdown("""
<style>
    /* Dark theme sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #1e1e1e !important;
    }
    
    /* Section headers */
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
        font-size: 16px !important;
    }
    
    [data-testid="stSidebar"] h3:first-child {
        margin-top: 0 !important;
    }
    
    /* Radio button labels */
    [data-testid="stSidebar"] .stRadio label {
        color: #e0e0e0 !important;
        font-size: 14px !important;
        padding: 8px 4px !important;
        border-radius: 4px !important;
        transition: background-color 0.2s !important;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #2a2a2a !important;
    }
    
    /* Radio button outer circle - selected (red border) */
    [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label > div:first-child {
        border: 2px solid #ff4444 !important;
    }
    
    /* Radio button inner circle - selected (red fill) */
    [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label > div:first-child > div {
        background-color: #ff4444 !important;
        width: 10px !important;
        height: 10px !important;
        border-radius: 50% !important;
    }
    
    /* Radio button outer circle - unselected */
    [data-testid="stSidebar"] .stRadio input[type="radio"] + label > div:first-child {
        border: 2px solid #666666 !important;
    }
    
    /* Radio button container */
    [data-testid="stSidebar"] .stRadio {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* All text in sidebar */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div:not([class*="st"]),
    [data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'eda_ticker_selected' not in st.session_state:
    st.session_state.eda_ticker_selected = None
if 'eda_date_range_selected' not in st.session_state:
    st.session_state.eda_date_range_selected = None

# Sidebar: Navigation
with st.sidebar:
    # Initialize page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    
    # Check if data is available
    has_data = (st.session_state.df_processed is not None and 
               hasattr(st.session_state.df_processed, 'empty') and 
               not st.session_state.df_processed.empty)
    
    # Navigation Section
    st.markdown("### Navigation")
    
    # Navigation options - semua halaman selalu ditampilkan
    nav_options = [
        "🏠 Home",
        "📊 ETL Results",
        "🔍 Exploratory Data Analysis",
        "🤖 Forecasting",
        "📈 Model Evaluation"
    ]
    
    # Tentukan index halaman saat ini
    current_index = 0
    if st.session_state.current_page in nav_options:
        current_index = nav_options.index(st.session_state.current_page)
    
    # Radio button for navigation - semua opsi selalu terlihat
    selected_page = st.radio(
        "Navigation",
        options=nav_options,
        index=current_index,
        label_visibility="collapsed",
        key="nav_radio",
        disabled=False  # Semua opsi bisa dipilih, tapi akan ada warning di halaman jika tidak ada data
    )
    
    # Update current page if changed
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    # Global Filters untuk EDA page (di sidebar)
    if selected_page == "🔍 Exploratory Data Analysis":
        if (st.session_state.df_processed is not None and 
            hasattr(st.session_state.df_processed, 'columns') and 
            not st.session_state.df_processed.empty):
            
            st.markdown("---")
            st.markdown("### 🔍 Global Filters")
            
            if 'Ticker' in st.session_state.df_processed.columns:
                tickers = sorted(st.session_state.df_processed['Ticker'].unique().tolist())
                selected_ticker_eda = st.selectbox(
                    "Pilih Ticker",
                    ['All'] + tickers,
                    help="Filter data berdasarkan ticker symbol (berlaku untuk semua visualisasi)",
                    key="sidebar_eda_ticker"
                )
                if selected_ticker_eda == 'All':
                    selected_ticker_eda = None
                # Simpan ke session state
                st.session_state.eda_ticker_selected = selected_ticker_eda
            else:
                st.session_state.eda_ticker_selected = None
            
            if 'Date' in st.session_state.df_processed.columns:
                try:
                    min_date = st.session_state.df_processed['Date'].min().date()
                    max_date = st.session_state.df_processed['Date'].max().date()
                    date_range_eda = st.date_input(
                        "Pilih Range Tanggal",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        help="Filter data berdasarkan range tanggal (berlaku untuk semua visualisasi)",
                        key="sidebar_eda_date_range"
                    )
                    # Simpan ke session state
                    st.session_state.eda_date_range_selected = date_range_eda
                except Exception:
                    st.session_state.eda_date_range_selected = None
            else:
                st.session_state.eda_date_range_selected = None
        else:
            st.session_state.eda_ticker_selected = None
            st.session_state.eda_date_range_selected = None
    else:
        # Initialize untuk halaman lain
        if 'eda_ticker_selected' not in st.session_state:
            st.session_state.eda_ticker_selected = None
        if 'eda_date_range_selected' not in st.session_state:
            st.session_state.eda_date_range_selected = None
    
    # Initialize default values untuk forecasting options
    if 'forecast_periods' not in st.session_state:
        st.session_state.forecast_periods = 90
    if 'add_regressors' not in st.session_state:
        st.session_state.add_regressors = True
    if 'split_date_str' not in st.session_state:
        st.session_state.split_date_str = '2021-01-01'
    
    # Initialize filters
    selected_ticker = None
    date_range = None

# Set current page - harus setelah semua button dibuat (di luar sidebar)
page = st.session_state.current_page

# Check if data exists
has_data = (st.session_state.df_processed is not None and 
           hasattr(st.session_state.df_processed, 'empty') and 
           not st.session_state.df_processed.empty)

# Main content based on selected page
# Allow Home and ETL Results pages even without data, but require data for other pages
if page not in ["🏠 Home", "📊 ETL Results"] and not has_data:
    safe_warning("⚠️ Upload dan proses file terlebih dahulu untuk mengakses halaman ini.")
    safe_info("👈 Silakan upload file CSV di halaman ETL Results untuk memulai analisis.")
    st.stop()

# Main Content Area
if page == "🏠 Home":
    st.header("📈 FinScope - Financial Forecasting Dashboard")
    st.caption("Platform Analisis Data untuk Prediksi Harga Saham S&P 500")
    
    st.markdown("---")
    
    # Project Group Section
    st.subheader("👥 Project Group - Data Acquisition B")
    
    # Team Members
    st.markdown("**Anggota Tim:**")
    st.markdown("""
    - **Mhd Ulil Abshar** - 2211521003
    - **Mashia Zavira Septyana** - 2311522028
    - **Zhahra Idhya Astwoti** - 2311523006
    """)
    
    # Supervisor
    st.markdown("**Dosen Pengampu:**")
    st.markdown("""
    **Rahmatika Pratama Santi, M.T.**  
    NIDN: 199308152022032017
    """)
    
    st.markdown("---")
    
    # About Website Section
    st.subheader("💻 Tentang Website")
    
    st.markdown("""
    Website ini merupakan platform analisis data yang fokus pada pengolahan dan visualisasi data 
    historis saham perusahaan S&P 500. Kami menyajikan berbagai analisis dan visualisasi interaktif 
    untuk membantu investor dan analis memahami tren pasar saham serta memprediksi pergerakan harga 
    di masa depan.
    """)
    
    st.markdown("**Objek Pengembangan:**")
    st.markdown("""
    Objek pengembangan dalam aplikasi FinScope adalah data historis saham perusahaan S&P 500 yang 
    diambil dari dataset berjudul **"SP_500_Stocks_Data-ratios_news_price_10_yrs"**. 
    Dataset ini menyediakan informasi mengenai harga saham harian, volume perdagangan, serta berbagai 
    rasio keuangan fundamental selama sepuluh tahun terakhir. Dataset terdiri dari satu file utama yang 
    memuat kombinasi data harga, waktu, dan rasio finansial.
    """)
    
    st.markdown("**Fitur-fitur utama website:**")
    st.markdown("""
    - **ETL Pipeline**: Proses ekstraksi, transformasi, dan loading data saham dengan kemampuan handle dataset besar (jutaan rows)
    - **Exploratory Data Analysis**: Analisis data eksploratif untuk memahami pola, tren, dan karakteristik data saham
    - **Time Series Forecasting**: Prediksi harga saham di masa depan menggunakan model Prophet dengan regressors fundamental
    - **Visualisasi Interaktif**: Dashboard dengan berbagai visualisasi interaktif untuk analisis mendalam data saham
    - **What-If Analysis**: Simulasi skenario untuk melihat dampak perubahan variabel fundamental terhadap prediksi harga
    """)
    


elif page == "📊 ETL Results":
    st.header("ETL Results")
    
    # File Upload Section
    st.subheader("📁 Upload File")
    st.caption("Max 200MB")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="File harus memiliki kolom: Date, Ticker, Open, High, Low, Close, Volume, ROE, Debt_Equity, EBIT_Margin"
    )
    
    # Check file size before processing
    # Downsampling AKTIF secara default untuk file > 50MB untuk optimasi memory
    enable_downsample = False
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        # Validasi ukuran file - max 200MB
        if file_size_mb > 200:
            safe_error(f"⚠️ File terlalu besar ({file_size_mb:.2f} MB). Maksimal 200MB.")
            uploaded_file = None
        elif file_size_mb > 100:
            safe_warning(f"⚠️ File besar ({file_size_mb:.2f} MB). Proses mungkin memakan waktu lama.")
            enable_downsample = True  # Aktif untuk file > 100MB
        elif file_size_mb > 50:
            safe_info(f"ℹ️ File sedang ({file_size_mb:.2f} MB). Downsampling aktif untuk performa optimal.")
            enable_downsample = True  # Aktif untuk file > 50MB
        else:
            enable_downsample = False  # Tidak perlu downsampling untuk file kecil
    
    # Process ETL jika file baru di-upload
    if uploaded_file is not None:
        # Process ETL jika file baru di-upload
        if st.session_state.df_processed is None or 'file_name' not in st.session_state.metadata or \
           st.session_state.metadata['file_name'] != uploaded_file.name:
            
            with st.spinner("Memproses data... Mohon tunggu."):
                try:
                    # Validate file first
                    if uploaded_file.size == 0:
                        safe_error("❌ File kosong! Silakan upload file yang berisi data.")
                        uploaded_file = None
                        st.stop()
                    
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    
                    # For large files (>50MB), use optimized chunked reading with downsampling
                    if file_size_mb > 50:
                        import io
                        import gc
                        from utils import downsample_data
                        
                        # Chunking untuk optimasi memory - minimal 1 juta rows
                        # Tidak ada downsampling yang membatasi, hanya chunking untuk efisiensi
                        target_rows = None  # Tidak ada limit - ambil semua data dengan chunking
                        
                        # Chunk size disesuaikan dengan ukuran file untuk optimasi memory
                        # Chunking memungkinkan file besar diproses tanpa MemoryError
                        if file_size_mb > 100:
                            chunk_size = 50000  # 50k rows per chunk untuk file besar
                        elif file_size_mb > 50:
                            chunk_size = 50000  # 50k rows per chunk untuk file sedang
                        else:
                            chunk_size = 50000  # 50k rows per chunk
                        
                        max_chunks = None  # Tidak ada limit chunks - ambil semua data
                        
                        uploaded_file.seek(0)
                        
                        # Progress tracking
                        progress_bar = None
                        status_text = None
                        
                        try:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            chunk_list = []
                            total_rows = 0
                            
                            # Start reading file
                            if status_text is not None:
                                try:
                                    status_text.text(f"📖 Membaca file dengan chunking (minimal 1M rows, bisa lebih)...")
                                except Exception:
                                    pass
                            
                            # Try to read with Date parsing (without pyarrow backend)
                            # Handle various encoding and format issues
                            try:
                                # First, try to read a small sample to check columns
                                uploaded_file.seek(0)
                                sample_df = pd.read_csv(uploaded_file, nrows=5, low_memory=True)
                                uploaded_file.seek(0)
                                
                                # Check if Date column exists
                                has_date_col = 'Date' in sample_df.columns
                                
                                if has_date_col:
                                    # Try with Date parsing
                                    try:
                                        chunk_reader = pd.read_csv(
                                            uploaded_file, 
                                            chunksize=chunk_size, 
                                            parse_dates=['Date'], 
                                            low_memory=True,
                                            encoding='utf-8',
                                            on_bad_lines='skip'  # Skip bad lines
                                        )
                                    except (UnicodeDecodeError, ValueError):
                                        # Try with different encoding
                                        uploaded_file.seek(0)
                                        chunk_reader = pd.read_csv(
                                            uploaded_file, 
                                            chunksize=chunk_size, 
                                            parse_dates=['Date'], 
                                            low_memory=True,
                                            encoding='latin-1',
                                            on_bad_lines='skip'
                                        )
                                else:
                                    # No Date column, read without parsing
                                    chunk_reader = pd.read_csv(
                                        uploaded_file, 
                                        chunksize=chunk_size, 
                                        low_memory=True,
                                        encoding='utf-8',
                                        on_bad_lines='skip'
                                    )
                            except Exception as e:
                                # Fallback: read without Date parsing and any encoding issues
                                uploaded_file.seek(0)
                                try:
                                    chunk_reader = pd.read_csv(
                                        uploaded_file, 
                                        chunksize=chunk_size, 
                                        low_memory=True,
                                        encoding='latin-1',
                                        on_bad_lines='skip'
                                    )
                                except:
                                    # Last resort: read with minimal options
                                    uploaded_file.seek(0)
                                    chunk_reader = pd.read_csv(
                                        uploaded_file, 
                                        chunksize=chunk_size, 
                                        low_memory=False,
                                        on_bad_lines='skip',
                                        encoding='latin-1'
                                    )
                            
                            for i, chunk in enumerate(chunk_reader):
                                # Convert any extension arrays to standard pandas types to avoid pyarrow issues
                                # Use inplace operations where possible to save memory
                                for col in chunk.columns:
                                    # Check if column uses extension array (pyarrow, nullable, etc.)
                                    dtype_str = str(chunk[col].dtype).lower()
                                    if ('arrow' in dtype_str or 
                                        'extension' in dtype_str or 
                                        'nullable' in dtype_str or
                                        hasattr(chunk[col].dtype, 'name') and 'Int' in str(chunk[col].dtype)):
                                        # Convert to standard type (inplace where possible)
                                        if pd.api.types.is_numeric_dtype(chunk[col]):
                                            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float64')
                                        elif pd.api.types.is_datetime64_any_dtype(chunk[col]):
                                            chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
                                        else:
                                            chunk[col] = chunk[col].astype(str)
                                
                                # Parse Date if not already parsed
                                if 'Date' in chunk.columns:
                                    if chunk['Date'].dtype == 'object' or 'datetime' not in str(chunk['Date'].dtype).lower():
                                        chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
                                
                                # Ambil semua chunk tanpa batasan - chunking hanya untuk optimasi memory
                                chunk_list.append(chunk)
                                total_rows += len(chunk)
                                
                                # Update progress dengan jeda untuk mencegah WebSocket timeout
                                if progress_bar is not None and i % 10 == 0:  # Update setiap 10 chunks
                                    try:
                                        # Progress berdasarkan jumlah chunks yang sudah dibaca
                                        # Estimasi: setiap 20 chunks = ~1M rows (dengan chunk_size 50k)
                                        progress = min(95, 10 + int((i / 20) * 85))  # Progress 10-95% berdasarkan chunks
                                        progress_bar.progress(progress)
                                        time.sleep(0.01)  # Jeda kecil untuk mencegah WebSocket timeout
                                    except Exception:
                                        pass  # Ignore WebSocket errors during progress update
                                
                                # Aggressive memory cleanup every 3 chunks untuk file besar
                                if len(chunk_list) % 3 == 0:
                                    gc.collect()
                                    time.sleep(0.01)  # Jeda kecil untuk mencegah WebSocket timeout
                            
                            if progress_bar is not None:
                                try:
                                    progress_bar.progress(95)
                                except Exception:
                                    pass
                            if status_text is not None:
                                try:
                                    status_text.text(f"📊 Menggabungkan {len(chunk_list)} chunks ({total_rows:,} rows)...")
                                except Exception:
                                    pass
                            
                            # Combine chunks in very small batches to minimize memory peak
                            if len(chunk_list) > 1:
                                # For very large files, use smaller batch size
                                batch_size = 3 if file_size_mb > 100 else 5
                                combined_batches = []
                                
                                for i in range(0, len(chunk_list), batch_size):
                                    batch = chunk_list[i:i+batch_size]
                                    if len(batch) > 1:
                                        # Use copy=False where possible to save memory
                                        combined_batches.append(pd.concat(batch, ignore_index=True, copy=False))
                                    else:
                                        combined_batches.append(batch[0])
                                    
                                    # Aggressive cleanup
                                    del batch
                                    if i % (batch_size * 3) == 0:  # GC every 3 batches
                                        gc.collect()
                                
                                # Final combine with copy=False
                                if len(combined_batches) > 1:
                                    df_combined = pd.concat(combined_batches, ignore_index=True, copy=False)
                                else:
                                    df_combined = combined_batches[0]
                                
                                del combined_batches, chunk_list
                                gc.collect()
                            else:
                                df_combined = chunk_list[0] if chunk_list else pd.DataFrame()
                                del chunk_list
                                gc.collect()
                            
                            if progress_bar is not None:
                                try:
                                    progress_bar.progress(98)
                                except Exception:
                                    pass
                            if status_text is not None:
                                try:
                                    status_text.text(f"✅ Data loaded: {len(df_combined):,} rows")
                                except Exception:
                                    pass
                            
                            # Tidak ada downsampling - semua data diambil dengan chunking
                            safe_info(f"📊 Data berhasil dimuat: {len(df_combined):,} rows (menggunakan chunking untuk optimasi memory)")
                            
                            # Process ETL from dataframe with memory optimization
                            if status_text is not None:
                                try:
                                    status_text.text("⚙️ Memproses ETL...")
                                except Exception:
                                    pass
                            
                            # Create ETL progress container
                            etl_container = st.container()
                            with etl_container:
                                st.subheader("📊 Proses ETL")
                                etl_progress_bar = st.progress(0)
                                etl_status = st.empty()
                                etl_metrics = st.empty()
                            
                            # Skip scaling for very large files to save memory
                            should_scale = file_size_mb <= 100  # Only scale if file < 100MB
                            
                            # Process ETL dengan visualisasi yang lebih baik
                            st.session_state.df_processed, st.session_state.metadata = process_etl_from_df(
                                df_combined,
                                clean_method='ffill',
                                calculate_features=True,
                                scale_close=should_scale,
                                progress_bar=etl_progress_bar,
                                status_text=etl_status,
                                metrics_container=etl_metrics
                            )
                            st.session_state.metadata['file_name'] = uploaded_file.name
                            
                            # Clear memory
                            del df_combined
                            gc.collect()
                            
                            if progress_bar is not None:
                                try:
                                    progress_bar.progress(100)
                                    progress_bar.empty()
                                except Exception:
                                    pass
                            if status_text is not None:
                                try:
                                    status_text.empty()
                                except Exception:
                                    pass
                            
                            safe_success(f"✅ Data berhasil diproses: {len(st.session_state.df_processed):,} rows")
                            
                            # Check null percentage
                            try:
                                null_info = check_null_percentage(st.session_state.df_processed)
                                if null_info and null_info.get('has_warning', False):
                                    safe_warning(null_info['warning_message'])
                            except (AttributeError, TypeError, Exception):
                                pass
                            
                        except MemoryError as e:
                            if progress_bar is not None:
                                try:
                                    progress_bar.empty()
                                except Exception:
                                    pass
                            if status_text is not None:
                                try:
                                    status_text.empty()
                                except Exception:
                                    pass
                            safe_error("❌ Memory Error: File terlalu besar untuk diproses.")
                            safe_info("💡 Downsampling sudah aktif secara otomatis untuk file > 50MB.")
                            safe_info("💡 Solusi: Gunakan file yang lebih kecil (< 150MB) atau split file menjadi beberapa bagian.")
                            st.session_state.df_processed = None
                            st.session_state.metadata = {}
                        except Exception as e:
                            if progress_bar is not None:
                                try:
                                    progress_bar.empty()
                                except Exception:
                                    pass
                            if status_text is not None:
                                try:
                                    status_text.empty()
                                except Exception:
                                    pass
                            safe_error(f"❌ Error memproses file: {str(e)}")
                            st.session_state.df_processed = None
                            st.session_state.metadata = {}
                        finally:
                            # Ensure cleanup even if something goes wrong
                            if progress_bar is not None:
                                try:
                                    progress_bar.empty()
                                except Exception:
                                    pass
                            if status_text is not None:
                                try:
                                    status_text.empty()
                                except Exception:
                                    pass
                    
                    # Handle smaller files (<50MB)
                    if file_size_mb <= 50:
                        # For smaller files, use temporary file approach
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            # Write in chunks to avoid loading entire file to memory
                            uploaded_file.seek(0)
                            chunk_size_bytes = 1024 * 1024  # 1MB chunks
                            while True:
                                chunk = uploaded_file.read(chunk_size_bytes)
                                if not chunk:
                                    break
                                tmp_file.write(chunk)
                            tmp_path = tmp_file.name
                        
                        try:
                            # Skip scaling for large files to save memory
                            file_size_mb_small = uploaded_file.size / (1024 * 1024)
                            should_scale_small = file_size_mb_small <= 100  # Only scale if file < 100MB
                            
                            # Try with default encoding first
                            try:
                                st.session_state.df_processed, st.session_state.metadata = process_etl(
                                    tmp_path,
                                    clean_method='ffill',
                                    calculate_features=True,
                                    scale_close=should_scale_small
                                )
                            except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as e:
                                # If encoding error, try with different encoding
                                safe_warning(f"⚠️ Masalah encoding terdeteksi. Mencoba encoding alternatif...")
                                # Re-read file with different encoding
                                try:
                                    # Read with latin-1 encoding and save to new temp file
                                    df_temp = pd.read_csv(
                                        tmp_path, 
                                        encoding='latin-1', 
                                        low_memory=True, 
                                        on_bad_lines='skip',
                                        engine='python',
                                        sep=None
                                    )
                                    tmp_path2 = tmp_path + '_latin1.csv'
                                    df_temp.to_csv(tmp_path2, index=False, encoding='utf-8')
                                    st.session_state.df_processed, st.session_state.metadata = process_etl(
                                        tmp_path2,
                                        clean_method='ffill',
                                        calculate_features=True,
                                        scale_close=should_scale_small
                                    )
                                    if os.path.exists(tmp_path2):
                                        os.unlink(tmp_path2)
                                except Exception as e2:
                                    safe_error(f"❌ Error memproses file: {str(e2)}")
                                    raise e2
                            
                            st.session_state.metadata['file_name'] = uploaded_file.name
                            
                            # Check null percentage
                            try:
                                null_info = check_null_percentage(st.session_state.df_processed)
                                if null_info and null_info.get('has_warning', False):
                                    safe_warning(null_info['warning_message'])
                            except (AttributeError, TypeError, Exception):
                                pass
                        finally:
                            # Clean up temporary file
                            if os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                except MemoryError:
                    safe_error("❌ Memory Error: File terlalu besar untuk diproses. Silakan aktifkan downsampling atau gunakan file yang lebih kecil.")
                    safe_info("💡 Tips: Aktifkan checkbox 'Aktifkan Downsampling' di atas untuk memproses file besar.")
                    st.session_state.df_processed = None
                    st.session_state.metadata = {}
                except Exception as e:
                    safe_error(f"❌ Error memproses file: {str(e)}")
                    st.session_state.df_processed = None
                    st.session_state.metadata = {}
            
    st.markdown("---")
    
    if (st.session_state.df_processed is not None and 
        hasattr(st.session_state.df_processed, 'columns') and 
        not st.session_state.df_processed.empty):
        
        # Filters untuk ETL Results page
        with st.expander("🔍 Filters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if 'Ticker' in st.session_state.df_processed.columns:
                    tickers = sorted(st.session_state.df_processed['Ticker'].unique().tolist())
                    selected_ticker_etl = st.selectbox(
                        "Pilih Ticker",
                        ['All'] + tickers,
                        help="Filter data berdasarkan ticker symbol",
                        key="etl_ticker"
                    )
                    if selected_ticker_etl == 'All':
                        selected_ticker_etl = None
                else:
                    selected_ticker_etl = None
            
            with col2:
                if 'Date' in st.session_state.df_processed.columns:
                    try:
                        min_date = st.session_state.df_processed['Date'].min().date()
                        max_date = st.session_state.df_processed['Date'].max().date()
                        date_range_etl = st.date_input(
                            "Pilih Range Tanggal",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            help="Filter data berdasarkan range tanggal",
                            key="etl_date_range"
                        )
                    except Exception:
                        date_range_etl = None
                else:
                    date_range_etl = None
        
        # ========== ETL RESULTS DISPLAY SECTION ==========
        st.subheader("📊 Hasil Preprocessing ETL")
        
        # Summary Overview
        st.markdown("### 📈 Overview Data")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            original_rows = st.session_state.metadata.get('original_rows', len(st.session_state.df_processed))
            final_rows = st.session_state.metadata.get('final_rows', len(st.session_state.df_processed))
            st.metric("Total Rows", f"{final_rows:,}", delta=f"{final_rows - original_rows:,}" if original_rows != final_rows else None)
        
        with col2:
            num_cols = len(st.session_state.df_processed.columns)
            original_cols = len(st.session_state.metadata.get('original_columns', []))
            st.metric("Total Columns", num_cols, delta=f"{num_cols - original_cols}" if original_cols != num_cols else None)
        
        with col3:
            num_tickers = st.session_state.metadata.get('num_tickers', 0)
            if 'Ticker' in st.session_state.df_processed.columns:
                num_tickers = len(st.session_state.df_processed['Ticker'].unique())
            st.metric("Jumlah Ticker", num_tickers)
        
        with col4:
            date_range_info = st.session_state.metadata.get('date_range', {})
            if date_range_info and date_range_info.get('min') and date_range_info.get('max'):
                try:
                    from datetime import datetime
                    min_date = datetime.strptime(date_range_info['min'], '%Y-%m-%d %H:%M:%S').date() if isinstance(date_range_info['min'], str) else date_range_info['min']
                    max_date = datetime.strptime(date_range_info['max'], '%Y-%m-%d %H:%M:%S').date() if isinstance(date_range_info['max'], str) else date_range_info['max']
                    if isinstance(min_date, str):
                        min_date = datetime.strptime(min_date.split()[0], '%Y-%m-%d').date()
                    if isinstance(max_date, str):
                        max_date = datetime.strptime(max_date.split()[0], '%Y-%m-%d').date()
                    days_span = (max_date - min_date).days
                    st.metric("Rentang Data", f"{days_span:,} hari", help=f"{min_date} s/d {max_date}")
                except:
                    st.metric("Rentang Data", "N/A")
            else:
                st.metric("Rentang Data", "N/A")
        
        st.markdown("---")
        
        # Data Quality Report
        st.markdown("### 🔍 Data Quality Report")
        
        # Null Information
        null_info = st.session_state.metadata.get('null_info', {})
        if null_info and null_info.get('null_percentages'):
            null_percentages = null_info['null_percentages']
            
            # Create visualization for null percentages
            if null_percentages:
                null_df = pd.DataFrame({
                    'Column': list(null_percentages.keys()),
                    'Null Percentage': list(null_percentages.values())
                }).sort_values('Null Percentage', ascending=False)
                
                # Filter only columns with null values
                null_df = null_df[null_df['Null Percentage'] > 0]
                
                if not null_df.empty:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Bar chart for null percentages
                        fig_null = px.bar(
                            null_df.head(20),  # Top 20 columns with nulls
                            x='Column',
                            y='Null Percentage',
                            title='Persentase Missing Values per Kolom',
                            labels={'Null Percentage': 'Persentase (%)', 'Column': 'Kolom'},
                            color='Null Percentage',
                            color_continuous_scale='Reds'
                        )
                        fig_null.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_null, width='stretch')
                    
                    with col2:
                        st.markdown("**📋 Summary:**")
                        max_null = null_info.get('max_null', 0)
                        st.metric("Max Null %", f"{max_null:.2f}%")
                        st.metric("Kolom dengan Null", len(null_df))
                        
                        if null_info.get('has_warning', False):
                            st.warning("⚠️ Beberapa kolom memiliki >20% null values")
                        else:
                            st.success("✅ Data quality baik")
                else:
                    st.success("✅ Tidak ada missing values dalam data!")
            else:
                st.info("ℹ️ Informasi null tidak tersedia")
        else:
            # Calculate null info if not in metadata
            try:
                null_counts = st.session_state.df_processed.isnull().sum()
                null_percentages = (null_counts / len(st.session_state.df_processed)) * 100
                null_df = pd.DataFrame({
                    'Column': null_percentages.index,
                    'Null Percentage': null_percentages.values
                }).sort_values('Null Percentage', ascending=False)
                null_df = null_df[null_df['Null Percentage'] > 0]
                
                if not null_df.empty:
                    fig_null = px.bar(
                        null_df.head(20),
                        x='Column',
                        y='Null Percentage',
                        title='Persentase Missing Values per Kolom',
                        labels={'Null Percentage': 'Persentase (%)', 'Column': 'Kolom'},
                        color='Null Percentage',
                        color_continuous_scale='Reds'
                    )
                    fig_null.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_null, width='stretch')
                else:
                    st.success("✅ Tidak ada missing values dalam data!")
            except Exception as e:
                st.info("ℹ️ Tidak dapat menghitung null percentages")
        
        # Duplicates check
        st.markdown("#### 🔄 Duplicate Records")
        try:
            duplicates = st.session_state.df_processed.duplicated().sum()
            if duplicates > 0:
                st.warning(f"⚠️ Ditemukan {duplicates:,} baris duplikat ({duplicates/len(st.session_state.df_processed)*100:.2f}%)")
            else:
                st.success("✅ Tidak ada baris duplikat")
        except:
            st.info("ℹ️ Tidak dapat memeriksa duplikat")
        
        st.markdown("---")
        
        # Data Types Overview
        st.markdown("### 📋 Data Types & Structure")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Tipe Data per Kolom")
            dtype_df = pd.DataFrame({
                'Column': st.session_state.df_processed.columns,
                'Data Type': [str(dtype) for dtype in st.session_state.df_processed.dtypes],
                'Non-Null Count': [st.session_state.df_processed[col].notna().sum() for col in st.session_state.df_processed.columns]
            })
            dtype_df['Null Count'] = len(st.session_state.df_processed) - dtype_df['Non-Null Count']
            st.dataframe(dtype_df, width='stretch', height=300)
        
        with col2:
            st.markdown("#### Summary Statistics")
            numeric_cols = st.session_state.df_processed.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                try:
                    summary_stats = st.session_state.df_processed[numeric_cols[:10]].describe()  # First 10 numeric columns
                    st.dataframe(summary_stats, width='stretch', height=300)
                except:
                    st.info("ℹ️ Tidak dapat menghitung summary statistics")
            else:
                st.info("ℹ️ Tidak ada kolom numerik")
        
        st.markdown("---")
        
        # Features Created
        st.markdown("### 🎯 Features yang Dibuat")
        original_cols = set(st.session_state.metadata.get('original_columns', []))
        final_cols = set(st.session_state.df_processed.columns)
        new_features = final_cols - original_cols
        
        if new_features:
            st.success(f"✅ {len(new_features)} feature baru berhasil dibuat:")
            feature_list = list(new_features)
            
            # Categorize features
            return_features = [f for f in feature_list if 'return' in f.lower() or 'Return' in f]
            vol_features = [f for f in feature_list if 'volatility' in f.lower() or 'Volatility' in f or 'vol' in f.lower()]
            scaled_features = [f for f in feature_list if 'scaled' in f.lower() or 'Scaled' in f]
            other_features = [f for f in feature_list if f not in return_features + vol_features + scaled_features]
            
            cols = st.columns(2)
            with cols[0]:
                if return_features:
                    st.markdown("**📈 Return Features:**")
                    for feat in return_features:
                        st.markdown(f"- `{feat}`")
                if vol_features:
                    st.markdown("**📊 Volatility Features:**")
                    for feat in vol_features:
                        st.markdown(f"- `{feat}`")
            
            with cols[1]:
                if scaled_features:
                    st.markdown("**⚖️ Scaled Features:**")
                    for feat in scaled_features:
                        st.markdown(f"- `{feat}`")
                if other_features:
                    st.markdown("**🔧 Other Features:**")
                    for feat in other_features[:10]:  # Limit to 10
                        st.markdown(f"- `{feat}`")
                    if len(other_features) > 10:
                        st.caption(f"... dan {len(other_features) - 10} feature lainnya")
        else:
            st.info("ℹ️ Tidak ada feature baru yang dibuat")
        
        st.markdown("---")
        
        # Sample Data Preview
        st.markdown("### 👀 Preview Data")
        st.caption("Menampilkan 10 baris pertama dan 10 baris terakhir dari data yang sudah diproses")
        
        preview_cols = st.columns(2)
        with preview_cols[0]:
            st.markdown("**📌 10 Baris Pertama:**")
            st.dataframe(st.session_state.df_processed.head(10), width='stretch', height=300)
        
        with preview_cols[1]:
            st.markdown("**📌 10 Baris Terakhir:**")
            st.dataframe(st.session_state.df_processed.tail(10), width='stretch', height=300)
        
        st.markdown("---")
        
        # Apply filters (no need to copy - filtering creates new DataFrame)
        df_filtered_etl = st.session_state.df_processed
        if selected_ticker_etl and 'Ticker' in df_filtered_etl.columns:
            df_filtered_etl = df_filtered_etl[df_filtered_etl['Ticker'] == selected_ticker_etl]
        if date_range_etl and len(date_range_etl) == 2 and 'Date' in df_filtered_etl.columns:
            start_date, end_date = date_range_etl
            df_filtered_etl = df_filtered_etl[
                (df_filtered_etl['Date'].dt.date >= start_date) &
                (df_filtered_etl['Date'].dt.date <= end_date)
            ]
        
        # Filtered Data Summary (if filters applied)
        if selected_ticker_etl or (date_range_etl and len(date_range_etl) == 2):
            st.markdown("### 🔍 Data Setelah Filter")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
                st.metric("Filtered Rows", f"{len(df_filtered_etl):,}")
        with col2:
            st.metric("Total Columns", len(df_filtered_etl.columns))
        with col3:
            if 'Ticker' in df_filtered_etl.columns:
                    st.metric("Tickers", len(df_filtered_etl['Ticker'].unique()))
            else:
                    st.metric("Tickers", "N/A")
        with col4:
            if 'Date' in df_filtered_etl.columns:
                date_min = df_filtered_etl['Date'].min()
                date_max = df_filtered_etl['Date'].max()
                st.metric("Date Range", f"{date_min.date() if hasattr(date_min, 'date') else date_min} to {date_max.date() if hasattr(date_max, 'date') else date_max}")
            else:
                st.metric("Date Range", "N/A")
        
        # Additional filtered data preview (if filters applied)
        if selected_ticker_etl or (date_range_etl and len(date_range_etl) == 2):
            st.markdown("#### 📋 Preview Data Setelah Filter")
            st.dataframe(df_filtered_etl.head(20), width='stretch', height=300)
    
elif page == "🔍 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    # Apply global filters dari sidebar (menggunakan session state)
    if (st.session_state.df_processed is not None and 
        hasattr(st.session_state.df_processed, 'columns') and 
        not st.session_state.df_processed.empty):
        
        # Ambil filter dari session state (di-set di sidebar)
        selected_ticker_eda = st.session_state.get('eda_ticker_selected', None)
        date_range_eda = st.session_state.get('eda_date_range_selected', None)
        
        # Apply global filters
        df_filtered_eda = st.session_state.df_processed
        if selected_ticker_eda and 'Ticker' in df_filtered_eda.columns:
            df_filtered_eda = df_filtered_eda[df_filtered_eda['Ticker'] == selected_ticker_eda]
        if date_range_eda and len(date_range_eda) == 2 and 'Date' in df_filtered_eda.columns:
            start_date, end_date = date_range_eda
            df_filtered_eda = df_filtered_eda[
                (df_filtered_eda['Date'].dt.date >= start_date) &
                (df_filtered_eda['Date'].dt.date <= end_date)
            ]
    else:
        df_filtered_eda = pd.DataFrame()
        selected_ticker_eda = None
        date_range_eda = None
    
    if len(df_filtered_eda) > 0:
            # Optimasi: Untuk dataset besar, sample data untuk visualisasi
            max_rows_for_viz = 100000  # Maksimal 100k rows untuk visualisasi
            if len(df_filtered_eda) > max_rows_for_viz:
                safe_info(f"⚠️ Dataset besar ({len(df_filtered_eda):,} rows). Menggunakan sample {max_rows_for_viz:,} rows untuk visualisasi.")
                df_viz = df_filtered_eda.sample(n=max_rows_for_viz, random_state=42).copy()
            else:
                df_viz = df_filtered_eda.copy()
            
            # Pastikan Date column ada dan di-set sebagai index untuk time series
            if 'Date' in df_viz.columns:
                df_viz = df_viz.copy()
                df_viz['Date'] = pd.to_datetime(df_viz['Date'])
                df_viz = df_viz.sort_values('Date')
                df_ts = df_viz.set_index('Date')
            else:
                df_ts = None
            
            # 1. Line Chart: Historical Price & Volume Trends dengan Volatility overlay
            st.subheader("1. Historical Price & Volume Trends (Line Chart)")
            
            # Filter khusus untuk Line Chart
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_open = st.checkbox("Tampilkan Open Price", value=True, key="line_show_open")
                    show_volatility = st.checkbox("Tampilkan Volatility", value=True, key="line_show_volatility")
                with col2:
                    show_volume = st.checkbox("Tampilkan Volume", value=True, key="line_show_volume")
                    resample_option = st.selectbox(
                        "Resample Frequency",
                        ["Auto", "Daily", "Weekly", "Monthly"],
                        index=0,
                        key="line_resample"
                    )
                with col3:
                    chart_height = st.slider("Chart Height", 300, 800, 500, 50, key="line_height")
            
            try:
                if df_ts is not None and 'Close' in df_ts.columns:
                    # Determine resample frequency based on user selection
                    if resample_option == "Auto":
                        # Auto resample untuk dataset besar (>5 tahun)
                        if len(df_ts) > 1300:  # ~5 tahun daily data
                            resample_freq = 'W'
                            safe_info("📊 Data di-resample ke weekly untuk performa yang lebih baik.")
                        else:
                            resample_freq = None
                    elif resample_option == "Daily":
                        resample_freq = None
                    elif resample_option == "Weekly":
                        resample_freq = 'W'
                    elif resample_option == "Monthly":
                        resample_freq = 'ME'
                    
                    chart_cols = ['Close']
                    if show_open and 'Open' in df_ts.columns:
                        chart_cols.append('Open')
                    
                    if resample_freq:
                        df_chart = df_ts[chart_cols].resample(resample_freq).last()
                        if show_volume and 'Volume' in df_ts.columns:
                            df_chart['Volume'] = df_ts['Volume'].resample(resample_freq).sum()
                        if show_volatility and 'Volatility_30d' in df_ts.columns:
                            df_chart['Volatility_30d'] = df_ts['Volatility_30d'].resample(resample_freq).mean()
                    else:
                        df_chart = df_ts[chart_cols].copy()
                        if show_volume and 'Volume' in df_ts.columns:
                            df_chart['Volume'] = df_ts['Volume']
                        if show_volatility and 'Volatility_30d' in df_ts.columns:
                            df_chart['Volatility_30d'] = df_ts['Volatility_30d']
                    
                    # Create figure with secondary y-axis
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Price traces (primary y-axis)
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], 
                                            name='Close Price', line=dict(color='#1f77b4', width=2)),
                                 secondary_y=False)
                    if show_open and 'Open' in df_chart.columns:
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Open'], 
                                                name='Open Price', line=dict(color='#2ca02c', width=1, dash='dot')),
                                     secondary_y=False)
                    
                    # Volatility overlay (primary y-axis, as area)
                    if show_volatility and 'Volatility_30d' in df_chart.columns:
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Volatility_30d'], 
                                                name='Volatility (30d)', line=dict(color='red', width=1, dash='dash'),
                                                fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'),
                                     secondary_y=False)
                    
                    # Volume trace (secondary y-axis)
                    if show_volume and 'Volume' in df_chart.columns:
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Volume'], 
                                                name='Volume', line=dict(color='rgba(255,165,0,0.5)', width=1),
                                                fill='tozeroy', fillcolor='rgba(255,165,0,0.1)'),
                                     secondary_y=True)
                    
                    # Update axes
                    fig.update_xaxes(title_text="Date")
                    fig.update_yaxes(title_text="Price", secondary_y=False)
                    if show_volume:
                        fig.update_yaxes(title_text="Volume", secondary_y=True)
                    
                    fig.update_layout(
                        title="Historical Price & Volume Trends dengan Volatility Overlay",
                        hovermode='x unified',
                        height=chart_height
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    safe_warning("Data tidak memiliki kolom Date atau Close untuk line chart.")
            except Exception as e:
                safe_error(f"Error membuat line chart: {str(e)}")
            
            # 2. Fundamental Divergence Chart (Dual-Axis)
            st.subheader("2. Fundamental Divergence: Harga vs Kinerja Perusahaan")
            
            # Filter khusus untuk Fundamental Divergence
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    divergence_height = st.slider("Chart Height", 300, 800, 500, 50, key="divergence_height")
                    fundamental_metric = st.selectbox(
                        "Fundamental Metric",
                        ["ROE", "Net_Profit_Margin", "EBIT_Margin"],
                        index=0,
                        help="Pilih metrik fundamental untuk dibandingkan dengan harga",
                        key="divergence_metric"
                    )
                with col2:
                    show_divergence_zones = st.checkbox("Tampilkan Zona Divergence", value=True, key="divergence_zones")
                    resample_divergence = st.selectbox(
                        "Resample Frequency",
                        ["None", "Weekly", "Monthly"],
                        index=1,
                        key="divergence_resample"
                    )
            
            try:
                if df_ts is not None and 'Close' in df_ts.columns and fundamental_metric in df_viz.columns:
                    df_div = df_viz.copy()
                    df_div['Date'] = pd.to_datetime(df_div['Date'])
                    df_div = df_div.sort_values('Date')
                    
                    # Apply resample jika dipilih
                    if resample_divergence == "Weekly":
                        df_div = df_div.set_index('Date').resample('W').agg({
                            'Close': 'last',
                            fundamental_metric: 'mean'
                        }).reset_index()
                    elif resample_divergence == "Monthly":
                        df_div = df_div.set_index('Date').resample('ME').agg({
                            'Close': 'last',
                            fundamental_metric: 'mean'
                        }).reset_index()
                    
                    # Create figure with secondary y-axis
                    fig_div = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Price line (primary y-axis)
                    fig_div.add_trace(
                        go.Scatter(
                            x=df_div['Date'],
                            y=df_div['Close'],
                            name='Harga Saham (Close)',
                            line=dict(color='#1f77b4', width=2)
                        ),
                        secondary_y=False
                    )
                    
                    # Fundamental metric line (secondary y-axis)
                    fig_div.add_trace(
                        go.Scatter(
                            x=df_div['Date'],
                            y=df_div[fundamental_metric],
                            name=f'{fundamental_metric}',
                            line=dict(color='#ff7f0e', width=2, dash='dash')
                        ),
                        secondary_y=True
                    )
                    
                    # Highlight divergence zones
                    if show_divergence_zones and len(df_div) > 1:
                        df_div['Price_Change'] = df_div['Close'].pct_change()
                        df_div['Fundamental_Change'] = df_div[fundamental_metric].pct_change()
                        df_div['Divergence'] = df_div.apply(
                            lambda row: 'Convergence' if (row['Price_Change'] >= 0 and row['Fundamental_Change'] >= 0) or 
                                        (row['Price_Change'] < 0 and row['Fundamental_Change'] < 0)
                            else 'Divergence', axis=1
                        )
                        
                        # Add divergence annotations
                        for i in range(1, len(df_div)):
                            if df_div.iloc[i]['Divergence'] == 'Divergence':
                                price_chg = df_div.iloc[i]['Price_Change']
                                fund_chg = df_div.iloc[i]['Fundamental_Change']
                                if price_chg > 0 and fund_chg < 0:
                                    # Bubble warning
                                    fig_div.add_annotation(
                                        x=df_div.iloc[i]['Date'],
                                        y=df_div.iloc[i]['Close'],
                                        text="⚠️ Bubble",
                                        showarrow=True,
                                        arrowhead=2,
                                        bgcolor="red",
                                        font=dict(color="white", size=10)
                                    )
                                elif price_chg < 0 and fund_chg > 0:
                                    # Opportunity
                                    fig_div.add_annotation(
                                        x=df_div.iloc[i]['Date'],
                                        y=df_div.iloc[i]['Close'],
                                        text="💎 Opportunity",
                                        showarrow=True,
                                        arrowhead=2,
                                        bgcolor="green",
                                        font=dict(color="white", size=10)
                                    )
                    
                    fig_div.update_xaxes(title_text="Date")
                    fig_div.update_yaxes(title_text="Harga Saham (USD)", secondary_y=False)
                    fig_div.update_yaxes(title_text=f"{fundamental_metric} (%)", secondary_y=True)
                    fig_div.update_layout(
                        title=f"Fundamental Divergence: Harga vs {fundamental_metric}",
                        hovermode='x unified',
                        height=divergence_height
                    )
                    
                    st.plotly_chart(fig_div, width='stretch', key=f'divergence_{selected_ticker_eda}')
                    
                    # Key Insight
                    if len(df_div) > 1:
                        recent_price_chg = df_div['Close'].pct_change().tail(10).mean()
                        recent_fund_chg = df_div[fundamental_metric].pct_change().tail(10).mean()
                        if recent_price_chg > 0 and recent_fund_chg > 0:
                            safe_success("✅ **Key Insight**: Convergence terdeteksi! Harga naik dan fundamental naik = Kenaikan sehat (Fundamental driven).")
                        elif recent_price_chg > 0 and recent_fund_chg < 0:
                            safe_error("❌ **Key Insight**: Divergence (Bahaya)! Harga naik TAPI fundamental turun = Saham 'Gorengan' atau Bubble (Harga naik tanpa dukungan kinerja).")
                        elif recent_price_chg < 0 and recent_fund_chg > 0:
                            safe_info("💡 **Key Insight**: Opportunity! Harga turun TAPI fundamental naik = Saham Undervalued (Salah harga, peluang beli).")
                else:
                    safe_warning(f"Tidak dapat membuat fundamental divergence chart. Pastikan data memiliki kolom Date, Close, dan {fundamental_metric}.")
            except Exception as e:
                safe_error(f"Error membuat fundamental divergence chart: {str(e)}")
            
            # 3. Heatmap: Correlation Matrix Ratios (fokus ke ratios utama)
            st.subheader("3. Correlation Matrix (Heatmap)")
            
            # Filter khusus untuk Correlation Heatmap
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    corr_threshold = st.slider(
                        "Minimum Correlation Threshold",
                        0.0, 1.0, 0.5, 0.1,
                        help="Hanya tampilkan korelasi di atas threshold ini",
                        key="corr_threshold"
                    )
                    corr_height = st.slider("Chart Height", 400, 1000, 600, 50, key="corr_height")
                with col2:
                    corr_colorscale = st.selectbox(
                        "Color Scale",
                        ["RdYlGn", "RdBu", "Viridis", "Plasma", "Coolwarm"],
                        index=0,
                        key="corr_colorscale"
                    )
                    show_corr_values = st.checkbox("Tampilkan Nilai Korelasi", value=True, key="corr_show_values")
            
            try:
                # Focus on key ratios: Close, ROE, EBIT_Margin, Debt_Equity, Volume, Volatility
                key_cols = ['Close', 'ROE', 'EBIT_Margin', 'Debt_Equity_Ratio', 'Debt_Equity', 
                           'Volume', 'Volatility_30d', 'Current_Ratio', 'Asset_Turnover', 'Gross_Margin']
                available_key_cols = [col for col in key_cols if col in df_viz.columns]
                
                if len(available_key_cols) >= 2:
                    # Create correlation matrix for key columns only
                    df_corr_subset = df_viz[available_key_cols].select_dtypes(include=[np.number])
                    if len(df_corr_subset.columns) >= 2:
                        corr_matrix = df_corr_subset.corr()
                        
                        # Filter correlation matrix berdasarkan threshold
                        corr_matrix_filtered = corr_matrix.copy()
                        if corr_threshold > 0:
                            # Set nilai di bawah threshold menjadi NaN untuk visualisasi
                            corr_matrix_filtered = corr_matrix_filtered.where(
                                (corr_matrix_filtered.abs() >= corr_threshold) | 
                                (corr_matrix_filtered == 1.0)
                            )
                        
                        # Create heatmap
                        text_values = None
                        if show_corr_values:
                            text_values = corr_matrix.round(2).values
                        
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_matrix_filtered.values,
                            x=corr_matrix_filtered.columns,
                            y=corr_matrix_filtered.columns,
                            colorscale=corr_colorscale,
                            zmid=0,
                            text=text_values,
                            texttemplate='%{text}' if show_corr_values else '',
                            textfont={"size": 10},
                            colorbar=dict(title="Correlation")
                        ))
                        
                        fig_corr.update_layout(
                            title="Correlation Matrix - Key Financial Ratios",
                            height=corr_height,
                            xaxis_title="",
                            yaxis_title=""
                        )
                        st.plotly_chart(fig_corr, width='stretch')
                        
                        # Show high correlations berdasarkan threshold yang dipilih
                        high_corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                val = corr_matrix.iloc[i, j]
                                if abs(val) > corr_threshold:
                                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
                        
                        if high_corr_pairs:
                            st.info(f"💡 **High Correlations (|r| > {corr_threshold})**: {len(high_corr_pairs)} pairs found. "
                                   f"Focus on correlations > {corr_threshold} for meaningful relationships.")
                    else:
                        safe_warning("Tidak cukup kolom numerik untuk correlation heatmap.")
                else:
                    # Fallback to original function
                    fig_corr = plot_correlation_heatmap(df_viz, ticker=selected_ticker_eda)
                    if fig_corr:
                        st.plotly_chart(fig_corr, width='stretch')
                    else:
                        safe_warning("Tidak dapat membuat correlation heatmap. Pastikan ada minimal 2 kolom numerik.")
            except Exception as e:
                safe_error(f"Error membuat correlation heatmap: {str(e)}")
            
            # 4. Volatility Band (Bollinger Band Style)
            st.subheader("4. Volatility Band: Bollinger Band Style Analysis")
            
            # Filter khusus untuk Volatility Band
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    volatility_height = st.slider("Chart Height", 300, 800, 500, 50, key="volatility_height")
                    volatility_multiplier = st.slider(
                        "Volatility Multiplier",
                        1.0, 3.0, 2.0, 0.5,
                        help="Faktor untuk menghitung upper/lower band (default: 2.0)",
                        key="volatility_mult"
                    )
                with col2:
                    show_volatility_ma = st.checkbox("Tampilkan Moving Average", value=True, key="volatility_ma")
                    ma_period_vol = st.number_input("MA Period", min_value=5, max_value=200, value=20, key="volatility_ma_period")
            
            try:
                if df_ts is not None and 'Close' in df_ts.columns and 'Volatility_30d' in df_viz.columns:
                    df_vol = df_viz.copy()
                    df_vol['Date'] = pd.to_datetime(df_vol['Date'])
                    df_vol = df_vol.sort_values('Date')
                    
                    # Calculate bands
                    df_vol['Upper_Band'] = df_vol['Close'] + (volatility_multiplier * df_vol['Volatility_30d'] * df_vol['Close'])
                    df_vol['Lower_Band'] = df_vol['Close'] - (volatility_multiplier * df_vol['Volatility_30d'] * df_vol['Close'])
                    
                    fig_vol = go.Figure()
                    
                    # Upper band
                    fig_vol.add_trace(go.Scatter(
                        x=df_vol['Date'],
                        y=df_vol['Upper_Band'],
                        name='Upper Band',
                        line=dict(color='rgba(255,0,0,0.3)', width=1),
                        showlegend=True
                    ))
                    
                    # Lower band
                    fig_vol.add_trace(go.Scatter(
                        x=df_vol['Date'],
                        y=df_vol['Lower_Band'],
                        name='Lower Band',
                        line=dict(color='rgba(255,0,0,0.3)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        showlegend=True
                    ))
                    
                    # Close price
                    fig_vol.add_trace(go.Scatter(
                        x=df_vol['Date'],
                        y=df_vol['Close'],
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Moving average
                    if show_volatility_ma:
                        df_vol['MA'] = df_vol['Close'].rolling(window=ma_period_vol).mean()
                        fig_vol.add_trace(go.Scatter(
                            x=df_vol['Date'],
                            y=df_vol['MA'],
                            name=f'MA{ma_period_vol}',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                    
                    fig_vol.update_layout(
                        title="Volatility Band (Bollinger Band Style)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        height=volatility_height
                    )
                    
                    st.plotly_chart(fig_vol, width='stretch', key=f'volatility_band_{selected_ticker_eda}')
                    
                    # Key Insight
                    if len(df_vol) > 0:
                        recent_vol = df_vol['Volatility_30d'].tail(30).mean()
                        band_width = (df_vol['Upper_Band'].iloc[-1] - df_vol['Lower_Band'].iloc[-1]) / df_vol['Close'].iloc[-1] * 100
                        if band_width > 20:
                            safe_warning(f"⚠️ **Key Insight**: Pita volatilitas lebar ({band_width:.1f}%) menunjukkan pasar sedang 'Panik' atau tidak pasti. Hati-hati dengan volatilitas tinggi.")
                        else:
                            safe_info(f"✅ **Key Insight**: Pita volatilitas sempit ({band_width:.1f}%) menunjukkan pasar sedang 'Tenang'. Jika harga menembus Upper Band, biasanya indikasi Overbought (terlalu mahal sesaat).")
                else:
                    safe_warning("Tidak dapat membuat volatility band. Pastikan data memiliki kolom Date, Close, dan Volatility_30d.")
            except Exception as e:
                safe_error(f"Error membuat volatility band: {str(e)}")
            
            # 5. Seasonal Heatmap Calendar (Bonus)
            st.subheader("5. Seasonal Return Heatmap: Best Time to Buy?")
            
            # Filter khusus untuk Seasonal Heatmap
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    seasonal_height = st.slider("Chart Height", 400, 1000, 600, 50, key="seasonal_height")
                    seasonal_colorscale = st.selectbox(
                        "Color Scale",
                        ["RdYlGn", "RdYlBu", "Viridis", "Plasma"],
                        index=0,
                        key="seasonal_colorscale"
                    )
                with col2:
                    show_seasonal_values = st.checkbox("Tampilkan Nilai Return", value=True, key="seasonal_show_values")
                    min_year_filter = st.number_input(
                        "Minimum Year",
                        min_value=2000,
                        max_value=2030,
                        value=2000,
                        help="Filter tahun minimum untuk ditampilkan",
                        key="seasonal_min_year"
                    )
            
            try:
                fig_heatmap = plot_seasonal_heatmap_calendar(df_viz, ticker=selected_ticker_eda)
                if fig_heatmap is not None:
                    st.plotly_chart(fig_heatmap, width='stretch', key=f'seasonal_heatmap_{selected_ticker_eda}')
                else:
                    safe_warning("Tidak dapat membuat seasonal heatmap. Pastikan data memiliki kolom Date dan Close.")
            except Exception as e:
                safe_error(f"Error membuat seasonal heatmap: {str(e)}")
            
            # 6. Financial Health Radar (Spider Chart)
            st.subheader("6. Financial Health Radar: 5 Pilar Kesehatan Perusahaan")
            
            # Filter khusus untuk Financial Health Radar
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    radar_height = st.slider("Chart Height", 400, 1000, 600, 50, key="radar_height")
                    show_radar_fill = st.checkbox("Tampilkan Fill Area", value=True, key="radar_show_fill")
                with col2:
                    radar_normalize = st.checkbox("Normalize Values", value=True, help="Normalize untuk perbandingan yang lebih adil", key="radar_normalize")
            
            try:
                # 5 Pilar: ROE, Asset Turnover, Current Ratio, Debt/Equity (inverted), Net Profit Margin
                radar_metrics = {
                    'ROE': 'ROE',
                    'Asset_Turnover': 'Asset_Turnover',
                    'Current_Ratio': 'Current_Ratio',
                    'Debt_Equity_Ratio': 'Debt_Equity_Ratio',  # Will be inverted
                    'Net_Profit_Margin': 'Net_Profit_Margin'
                }
                
                available_metrics = {k: v for k, v in radar_metrics.items() if v in df_viz.columns}
                
                if len(available_metrics) >= 3:
                    # Calculate average per ticker
                    if 'Ticker' in df_viz.columns:
                        df_radar = df_viz.groupby('Ticker').agg({
                            'ROE': 'mean' if 'ROE' in df_viz.columns else 'first',
                            'Asset_Turnover': 'mean' if 'Asset_Turnover' in df_viz.columns else 'first',
                            'Current_Ratio': 'mean' if 'Current_Ratio' in df_viz.columns else 'first',
                            'Debt_Equity_Ratio': 'mean' if 'Debt_Equity_Ratio' in df_viz.columns else 'first',
                            'Net_Profit_Margin': 'mean' if 'Net_Profit_Margin' in df_viz.columns else 'first'
                        }).reset_index()
                    else:
                        df_radar = pd.DataFrame({
                            'Ticker': ['ALL'],
                            'ROE': [df_viz['ROE'].mean()] if 'ROE' in df_viz.columns else [0],
                            'Asset_Turnover': [df_viz['Asset_Turnover'].mean()] if 'Asset_Turnover' in df_viz.columns else [0],
                            'Current_Ratio': [df_viz['Current_Ratio'].mean()] if 'Current_Ratio' in df_viz.columns else [0],
                            'Debt_Equity_Ratio': [df_viz['Debt_Equity_Ratio'].mean()] if 'Debt_Equity_Ratio' in df_viz.columns else [0],
                            'Net_Profit_Margin': [df_viz['Net_Profit_Margin'].mean()] if 'Net_Profit_Margin' in df_viz.columns else [0]
                        })
                    
                    # Invert Debt/Equity (makin kecil makin bagus)
                    if 'Debt_Equity_Ratio' in df_radar.columns:
                        max_debt = df_radar['Debt_Equity_Ratio'].max()
                        df_radar['Debt_Equity_Inverted'] = max_debt - df_radar['Debt_Equity_Ratio'] + 1
                    
                    # Normalize if requested
                    if radar_normalize:
                        for col in ['ROE', 'Asset_Turnover', 'Current_Ratio', 'Debt_Equity_Inverted', 'Net_Profit_Margin']:
                            if col in df_radar.columns:
                                max_val = df_radar[col].max()
                                min_val = df_radar[col].min()
                                if max_val > min_val:
                                    df_radar[col] = (df_radar[col] - min_val) / (max_val - min_val) * 100
                    
                    # Create radar chart
                    categories = ['ROE', 'Asset Turnover', 'Current Ratio', 'Debt/Equity (Inverted)', 'Net Profit Margin']
                    
                    fig_radar = go.Figure()
                    
                    for idx, row in df_radar.iterrows():
                        values = [
                            row.get('ROE', 0),
                            row.get('Asset_Turnover', 0),
                            row.get('Current_Ratio', 0),
                            row.get('Debt_Equity_Inverted', 0),
                            row.get('Net_Profit_Margin', 0)
                        ]
                        # Close the radar chart
                        values.append(values[0])
                        categories_closed = categories + [categories[0]]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories_closed,
                            fill='toself' if show_radar_fill else None,
                            name=row['Ticker']
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100 if radar_normalize else None]
                            )
                        ),
                        showlegend=True,
                        title="Financial Health Radar: 5 Pilar Kesehatan Perusahaan",
                        height=radar_height
                    )
                    
                    st.plotly_chart(fig_radar, width='stretch', key=f'health_radar_{selected_ticker_eda}')
                    
                    # Key Insight
                    safe_info("💡 **Key Insight**: Bentuk Segilima Penuh = Perusahaan Sempurna. Bentuk Gepeng/Penyok di satu sisi = Ada masalah fatal (misal: Profit tinggi tapi Hutang menumpuk).")
                else:
                    safe_warning("Tidak dapat membuat financial health radar. Pastikan data memiliki minimal 3 dari 5 kolom: ROE, Asset_Turnover, Current_Ratio, Debt_Equity_Ratio, Net_Profit_Margin.")
            except Exception as e:
                safe_error(f"Error membuat financial health radar: {str(e)}")
            
            # 7. Risk-Reward Quadrant (Scatter Plot)
            if 'Ticker' in df_viz.columns and df_viz['Ticker'].nunique() > 1:
                st.subheader("7. Risk-Reward Quadrant: Pilih Saham Terbaik")
                
                # Filter khusus untuk Risk-Reward Quadrant
                with st.expander("⚙️ Filter Visualisasi", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        quadrant_height = st.slider("Chart Height", 400, 1000, 600, 50, key="quadrant_height")
                        risk_metric = st.selectbox(
                            "Risk Metric",
                            ["Volatility_30d", "Debt_Equity_Ratio"],
                            index=0,
                            key="quadrant_risk"
                        )
                    with col2:
                        reward_metric = st.selectbox(
                            "Reward Metric",
                            ["Daily_Return", "ROE"],
                            index=0,
                            key="quadrant_reward"
                        )
                        use_volume_size = st.checkbox("Gunakan Volume sebagai Ukuran", value=True, key="quadrant_volume")
                
                try:
                    if risk_metric in df_viz.columns and reward_metric in df_viz.columns:
                        # Calculate average per ticker
                        df_quad = df_viz.groupby('Ticker').agg({
                            risk_metric: 'mean',
                            reward_metric: 'mean',
                            'Volume': 'mean' if 'Volume' in df_viz.columns else 'first'
                        }).reset_index()
                        
                        fig_quad = go.Figure()
                        
                        # Determine quadrant colors
                        risk_median = df_quad[risk_metric].median()
                        reward_median = df_quad[reward_metric].median()
                        
                        df_quad['Quadrant'] = df_quad.apply(
                            lambda row: 'Gems (Low Risk, High Reward)' if row[risk_metric] <= risk_median and row[reward_metric] > reward_median
                            else 'Toxic (High Risk, Low Reward)' if row[risk_metric] > risk_median and row[reward_metric] <= reward_median
                            else 'Aggressive (High Risk, High Reward)' if row[risk_metric] > risk_median and row[reward_metric] > reward_median
                            else 'Conservative (Low Risk, Low Reward)',
                            axis=1
                        )
                        
                        colors = {
                            'Gems (Low Risk, High Reward)': 'green',
                            'Toxic (High Risk, Low Reward)': 'red',
                            'Aggressive (High Risk, High Reward)': 'orange',
                            'Conservative (Low Risk, Low Reward)': 'blue'
                        }
                        
                        for quadrant in df_quad['Quadrant'].unique():
                            df_q = df_quad[df_quad['Quadrant'] == quadrant]
                            sizes = df_q['Volume'].values if use_volume_size and 'Volume' in df_q.columns else [10] * len(df_q)
                            
                            fig_quad.add_trace(go.Scatter(
                                x=df_q[risk_metric],
                                y=df_q[reward_metric],
                                mode='markers+text',
                                text=df_q['Ticker'],
                                textposition='top center',
                                name=quadrant,
                                marker=dict(
                                    size=sizes,
                                    color=colors[quadrant],
                                    opacity=0.6,
                                    line=dict(width=1, color='black')
                                )
                            ))
                        
                        # Add quadrant lines
                        fig_quad.add_hline(y=reward_median, line_dash="dash", line_color="gray", annotation_text="Reward Median")
                        fig_quad.add_vline(x=risk_median, line_dash="dash", line_color="gray", annotation_text="Risk Median")
                        
                        fig_quad.update_layout(
                            title="Risk-Reward Quadrant: Pilih Saham Terbaik",
                            xaxis_title=f"Risk ({risk_metric})",
                            yaxis_title=f"Reward ({reward_metric})",
                            height=quadrant_height,
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig_quad, width='stretch', key=f'risk_reward_{selected_ticker_eda}')
                        
                        # Key Insight
                        gems_count = len(df_quad[df_quad['Quadrant'] == 'Gems (Low Risk, High Reward)'])
                        toxic_count = len(df_quad[df_quad['Quadrant'] == 'Toxic (High Risk, Low Reward)'])
                        safe_info(f"💡 **Key Insight**: {gems_count} saham di Kuadran Gems (Kiri Atas) = Harta karun. {toxic_count} saham di Kuadran Toxic (Kanan Bawah) = Hindari.")
                    else:
                        safe_warning(f"Tidak dapat membuat risk-reward quadrant. Pastikan data memiliki kolom {risk_metric} dan {reward_metric}.")
                except Exception as e:
                    safe_error(f"Error membuat risk-reward quadrant: {str(e)}")
            
            # 8. Volume-Price Pressure Analysis
            st.subheader("8. Volume-Price Pressure: Deteksi Akumulasi/Distribusi")
            
            # Filter khusus untuk Volume-Price Pressure
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    volume_pressure_height = st.slider("Chart Height", 300, 800, 500, 50, key="volume_pressure_height")
                    show_volume_ma = st.checkbox("Tampilkan MA Volume", value=True, key="volume_pressure_ma")
                with col2:
                    volume_ma_period = st.number_input("MA Volume Period", min_value=5, max_value=50, value=20, key="volume_ma_period")
                    resample_volume = st.selectbox(
                        "Resample Frequency",
                        ["None", "Weekly", "Monthly"],
                        index=0,
                        key="volume_pressure_resample"
                    )
            
            try:
                if df_ts is not None and 'Close' in df_ts.columns and 'Volume' in df_viz.columns:
                    df_vol_press = df_viz.copy()
                    df_vol_press['Date'] = pd.to_datetime(df_vol_press['Date'])
                    df_vol_press = df_vol_press.sort_values('Date')
                    
                    # Apply resample jika dipilih
                    if resample_volume == "Weekly":
                        df_vol_press = df_vol_press.set_index('Date').resample('W').agg({
                            'Close': 'last',
                            'Volume': 'sum'
                        }).reset_index()
                    elif resample_volume == "Monthly":
                        df_vol_press = df_vol_press.set_index('Date').resample('ME').agg({
                            'Close': 'last',
                            'Volume': 'sum'
                        }).reset_index()
                    
                    # Calculate price change and volume color
                    df_vol_press['Price_Change'] = df_vol_press['Close'].pct_change()
                    df_vol_press['Volume_Color'] = df_vol_press['Price_Change'].apply(
                        lambda x: '#2ecc71' if x >= 0 else '#e74c3c'
                    )
                    
                    # Calculate MA Volume
                    if show_volume_ma:
                        df_vol_press['MA_Volume'] = df_vol_press['Volume'].rolling(window=volume_ma_period).mean()
                    
                    # Create subplots
                    fig_vol_press = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Price', 'Volume (Green=Up, Red=Down)')
                    )
                    
                    # Price line
                    fig_vol_press.add_trace(
                        go.Scatter(
                            x=df_vol_press['Date'],
                            y=df_vol_press['Close'],
                            name='Close Price',
                            line=dict(color='#1f77b4', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Volume bars
                    fig_vol_press.add_trace(
                        go.Bar(
                            x=df_vol_press['Date'],
                            y=df_vol_press['Volume'],
                            marker_color=df_vol_press['Volume_Color'],
                            name='Volume',
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    # MA Volume
                    if show_volume_ma:
                        fig_vol_press.add_trace(
                            go.Scatter(
                                x=df_vol_press['Date'],
                                y=df_vol_press['MA_Volume'],
                                name=f'MA Volume ({volume_ma_period})',
                                line=dict(color='orange', width=2, dash='dash')
                            ),
                            row=2, col=1
                        )
                    
                    fig_vol_press.update_xaxes(title_text="Date", row=2, col=1)
                    fig_vol_press.update_yaxes(title_text="Price (USD)", row=1, col=1)
                    fig_vol_press.update_yaxes(title_text="Volume", row=2, col=1)
                    fig_vol_press.update_layout(
                        title="Volume-Price Pressure Analysis",
                        hovermode='x unified',
                        height=volume_pressure_height
                    )
                    
                    st.plotly_chart(fig_vol_press, width='stretch', key=f'volume_pressure_{selected_ticker_eda}')
                    
                    # Key Insight
                    if len(df_vol_press) > 1:
                        recent_price_chg = df_vol_press['Price_Change'].tail(5).mean()
                        recent_volume = df_vol_press['Volume'].tail(5).mean()
                        avg_volume = df_vol_press['Volume'].mean()
                        if recent_price_chg > 0.02 and recent_volume > avg_volume * 1.5:
                            safe_success("✅ **Key Insight**: Validasi Tren! Harga naik tinggi + Volume Hijau Raksasa = Tren Kuat (Big Money masuk).")
                        elif recent_price_chg > 0.02 and recent_volume < avg_volume * 0.8:
                            safe_warning("⚠️ **Key Insight**: Jebakan (Fakeout)! Harga naik tinggi + Volume Kecil/Tipis = Kenaikan palsu, rawan bantingan.")
                else:
                    safe_warning("Tidak dapat membuat volume-price pressure chart. Pastikan data memiliki kolom Date, Close, dan Volume.")
            except Exception as e:
                safe_error(f"Error membuat volume-price pressure chart: {str(e)}")
            

elif page == "🤖 Forecasting":
    st.header("Prophet Forecasting")
    
    # Forecasting Options di halaman Forecasting
    if (st.session_state.df_processed is not None and 
        hasattr(st.session_state.df_processed, 'columns') and 
        not st.session_state.df_processed.empty):
        
        with st.expander("⚙️ Forecasting Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                forecast_periods = st.number_input(
                    "Forecast Periods (days)",
                    min_value=1,
                    max_value=365,
                    value=st.session_state.forecast_periods,
                    help="Jumlah hari ke depan untuk forecast",
                    key="forecast_periods_input"
                )
                st.session_state.forecast_periods = forecast_periods
            
            with col2:
                # Regressors tetap aktif, tidak perlu checkbox
                add_regressors = st.session_state.add_regressors
            
            # Auto-detect split date based on data
            if 'Date' in st.session_state.df_processed.columns:
                try:
                    df_dates = pd.to_datetime(st.session_state.df_processed['Date'])
                    min_date = df_dates.min()
                    max_date = df_dates.max()
                    total_days = (max_date - min_date).days
                    
                    # Use 80% of data for training (split at 80% point)
                    split_date = min_date + pd.Timedelta(days=int(total_days * 0.8))
                    
                    # Ensure minimum dates: at least 30 days for test, and split date not too early
                    if split_date > max_date - pd.Timedelta(days=30):
                        split_date = max_date - pd.Timedelta(days=30)
                    
                    if split_date < pd.Timestamp('2020-01-01'):
                        split_date = pd.Timestamp('2020-01-01')
                    
                    split_date_str = split_date.strftime('%Y-%m-%d')
                    st.session_state.split_date_str = split_date_str
                    
                    # Calculate train/test sizes
                    train_size = len(st.session_state.df_processed[df_dates < split_date])
                    test_size = len(st.session_state.df_processed[df_dates >= split_date])
                    
                    st.caption(f"💡 Split: {split_date_str} | Train: {train_size:,} rows | Test: {test_size:,} rows")
                except Exception as e:
                    split_date_str = '2021-01-01'
                    st.session_state.split_date_str = split_date_str
                    st.caption(f"⚠️ Using default split date: {split_date_str}")
            else:
                split_date_str = '2021-01-01'
                st.session_state.split_date_str = split_date_str
                st.caption(f"⚠️ No Date column, using default: {split_date_str}")
        
        # Filters untuk Forecasting page
        with st.expander("🔍 Filters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if 'Ticker' in st.session_state.df_processed.columns:
                    tickers = sorted(st.session_state.df_processed['Ticker'].unique().tolist())
                    selected_ticker_forecast = st.selectbox(
                        "Pilih Ticker",
                        ['All'] + tickers,
                        help="Filter data berdasarkan ticker symbol",
                        key="forecast_ticker"
                    )
                    if selected_ticker_forecast == 'All':
                        selected_ticker_forecast = None
                else:
                    selected_ticker_forecast = None
            
            with col2:
                if 'Date' in st.session_state.df_processed.columns:
                    try:
                        min_date = st.session_state.df_processed['Date'].min().date()
                        max_date = st.session_state.df_processed['Date'].max().date()
                        date_range_forecast = st.date_input(
                            "Pilih Range Tanggal",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            help="Filter data berdasarkan range tanggal",
                            key="forecast_date_range"
                        )
                    except Exception:
                        date_range_forecast = None
                else:
                    date_range_forecast = None
        
        # Apply filters (no need to copy - filtering creates new DataFrame)
        df_filtered_forecast = st.session_state.df_processed
        if selected_ticker_forecast and 'Ticker' in df_filtered_forecast.columns:
            df_filtered_forecast = df_filtered_forecast[df_filtered_forecast['Ticker'] == selected_ticker_forecast]
        if date_range_forecast and len(date_range_forecast) == 2 and 'Date' in df_filtered_forecast.columns:
            start_date, end_date = date_range_forecast
            df_filtered_forecast = df_filtered_forecast[
                (df_filtered_forecast['Date'].dt.date >= start_date) &
                (df_filtered_forecast['Date'].dt.date <= end_date)
            ]
    else:
        df_filtered_forecast = pd.DataFrame()
        selected_ticker_forecast = None
        forecast_periods = st.session_state.forecast_periods
        add_regressors = st.session_state.add_regressors
        split_date_str = st.session_state.split_date_str
    
    if len(df_filtered_forecast) == 0:
        safe_warning("Tidak ada data untuk forecasting. Silakan pilih filter yang berbeda.")
    else:
        # Train models button
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
                if 'Ticker' in df_filtered_forecast.columns:
                    tickers_to_train = df_filtered_forecast['Ticker'].unique().tolist()
                else:
                    tickers_to_train = ['ALL']
                
                with st.spinner(f"Training models untuk {len(tickers_to_train)} ticker(s)..."):
                    try:
                        # Limit parallel jobs untuk Windows (reduce resource usage)
                        # Windows memiliki limitasi resource yang lebih ketat
                        if platform.system() == 'Windows':
                            # Limit to max 2 jobs untuk Windows untuk menghindari resource exhaustion
                            max_jobs = min(2, len(tickers_to_train), os.cpu_count() or 1)
                        else:
                            # Linux/Mac bisa handle lebih banyak
                            max_jobs = min(4, len(tickers_to_train), (os.cpu_count() or 1) // 2)
                        
                        # Cleanup memory sebelum training
                        gc.collect()
                        
                        # Pass filtered tickers untuk training hanya ticker yang dipilih
                        # Gunakan data lengkap (st.session_state.df_processed) tapi hanya train ticker yang dipilih
                        results = train_models_parallel(
                            st.session_state.df_processed,  # Gunakan data lengkap untuk training
                            split_date=st.session_state.split_date_str,
                            add_regressors=st.session_state.add_regressors,
                            use_hyperparameter_tuning=True,  # Enable hyperparameter tuning
                            n_jobs=max_jobs,
                            tickers_to_train=tickers_to_train  # Pass list ticker yang ingin di-train (sudah di-filter)
                        )
                        
                        # Cleanup setelah training
                        gc.collect()
                        
                        st.session_state.trained_models = results
                        
                        # Display results
                        successful_count = len([r for r in results.values() if r.get('success')])
                        failed_count = len([r for r in results.values() if not r.get('success')])
                        
                        if successful_count > 0:
                            safe_success(f"✅ Training selesai untuk {successful_count} ticker(s)!")
                        else:
                            safe_error(f"❌ Training gagal untuk semua {len(tickers_to_train)} ticker(s)!")
                            if failed_count > 0:
                                # Show error details
                                st.warning("**Detail Error:**")
                                for ticker, result in results.items():
                                    if not result.get('success'):
                                        error_msg = result.get('error', 'Unknown error')
                                        st.text(f"• {ticker}: {error_msg}")
                        
                        # Metrics table
                        metrics_data = []
                        for ticker, result in results.items():
                            if result.get('success'):
                                metrics_data.append({
                                    'Ticker': ticker,
                                    'RMSE': result['metrics']['RMSE'],
                                    'MAE': result['metrics']['MAE'],
                                    'MAPE': f"{result['metrics']['MAPE']:.2f}%",
                                    'Train Size': result['train_size'],
                                    'Test Size': result['test_size']
                                })
                        
                        if metrics_data:
                            st.subheader("Model Metrics")
                            st.dataframe(pd.DataFrame(metrics_data), width='stretch')
                    except Exception as e:
                        safe_error(f"Error saat training: {str(e)}")
        
        st.markdown("---")
        
        # Forecast button
        if st.button("🔮 Generate Forecast", use_container_width=True):
                if not st.session_state.trained_models:
                    safe_error("Silakan train models terlebih dahulu!")
                else:
                    ticker_for_forecast = selected_ticker_forecast if selected_ticker_forecast else \
                        (df_filtered_forecast['Ticker'].iloc[0] if 'Ticker' in df_filtered_forecast.columns else None)
                    
                    if ticker_for_forecast:
                        with st.spinner("Generating forecast..."):
                            model = load_model(ticker_for_forecast)
                            if model:
                                forecast, _ = forecast_future(
                                    df_filtered_forecast,
                                    ticker_for_forecast,
                                    periods=st.session_state.forecast_periods,
                                    model=model,
                                    add_regressors=st.session_state.add_regressors
                                )
                                
                                if forecast is not None:
                                    st.session_state.forecasts[ticker_for_forecast] = forecast
                                    safe_success("Forecast berhasil di-generate!")
                                    
                                    # Display forecast summary
                                    st.subheader("Forecast Summary")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Forecast Start", str(forecast['ds'].iloc[-st.session_state.forecast_periods]))
                                    with col2:
                                        st.metric("Forecast End", str(forecast['ds'].iloc[-1]))
                                    with col3:
                                        st.metric("Forecasted Price", f"${forecast['yhat'].iloc[-1]:.2f}")
                            else:
                                safe_error(f"Model untuk {ticker_for_forecast} tidak ditemukan!")
                    else:
                        safe_error("Silakan pilih ticker untuk forecast!")
        
        # Forecasting Visualizations (Practical & Insightful)
        if st.session_state.forecasts:
            st.divider()
            st.header("📊 Forecasting Visualizations")
            
            ticker_viz = selected_ticker_forecast if selected_ticker_forecast else \
                (df_filtered_forecast['Ticker'].iloc[0] if 'Ticker' in df_filtered_forecast.columns and len(df_filtered_forecast) > 0 else None)
            
            if ticker_viz and ticker_viz in st.session_state.forecasts:
                forecast_data = st.session_state.forecasts[ticker_viz]
                model_data = load_model(ticker_viz) if ticker_viz else None
                
                # 1. Forecast Tunnel (Prediksi dengan Rentang Keyakinan)
                st.subheader("1. Forecast Tunnel: Prediksi dengan Rentang Keyakinan")
                try:
                    if 'Date' in df_filtered_forecast.columns and 'Close' in df_filtered_forecast.columns:
                        df_hist = df_filtered_forecast.copy()
                        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
                        df_hist = df_hist.sort_values('Date')
                        
                        # Get last year of historical data
                        last_date = df_hist['Date'].max()
                        one_year_ago = last_date - pd.Timedelta(days=365)
                        df_hist_recent = df_hist[df_hist['Date'] >= one_year_ago]
                        
                        # Prepare forecast data
                        forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
                        
                        fig_tunnel = go.Figure()
                        
                        # Historical data (solid line)
                        fig_tunnel.add_trace(go.Scatter(
                            x=df_hist_recent['Date'],
                            y=df_hist_recent['Close'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        # Forecast line (dashed)
                        if 'yhat' in forecast_data.columns:
                            fig_tunnel.add_trace(go.Scatter(
                                x=forecast_data['ds'],
                                y=forecast_data['yhat'],
                                mode='lines',
                                name='Forecast (Base)',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                        
                        # Confidence intervals (shaded area)
                        if 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
                            fig_tunnel.add_trace(go.Scatter(
                                x=forecast_data['ds'],
                                y=forecast_data['yhat_upper'],
                                mode='lines',
                                name='Upper CI',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            fig_tunnel.add_trace(go.Scatter(
                                x=forecast_data['ds'],
                                y=forecast_data['yhat_lower'],
                                mode='lines',
                                name='Confidence Interval',
                                fill='tonexty',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(width=0),
                                showlegend=True
                            ))
                        
                        fig_tunnel.update_layout(
                            title=f"Forecast Tunnel: {ticker_viz}",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_tunnel, width='stretch', key=f'tunnel_{ticker_viz}')
                        
                        # Key Insight
                        if 'yhat_upper' in forecast_data.columns and 'yhat_lower' in forecast_data.columns:
                            ci_width = ((forecast_data['yhat_upper'].iloc[-1] - forecast_data['yhat_lower'].iloc[-1]) / forecast_data['yhat'].iloc[-1]) * 100
                            if ci_width > 10:
                                safe_warning(f"⚠️ **Key Insight**: Confidence interval lebar ({ci_width:.1f}%) menunjukkan volatilitas tinggi. Harga kemungkinan besar akan bergerak dalam rentang yang luas.")
                            else:
                                safe_info(f"✅ **Key Insight**: Confidence interval sempit ({ci_width:.1f}%) menunjukkan prediksi yang lebih pasti. Model cukup yakin dengan arah harga.")
                    else:
                        safe_warning("Tidak dapat membuat forecast tunnel. Pastikan data memiliki kolom Date dan Close.")
                except Exception as e:
                    safe_error(f"Error membuat forecast tunnel: {str(e)}")
                
                # 2. Backtesting: Reality Check (Aktual vs Prediksi)
                st.subheader("2. Backtesting: Reality Check (Aktual vs Prediksi)")
                try:
                    if 'Date' in df_filtered_forecast.columns and 'Close' in df_filtered_forecast.columns and model_data:
                        # Get last year of data for backtesting
                        df_backtest = df_filtered_forecast.copy()
                        df_backtest['Date'] = pd.to_datetime(df_backtest['Date'])
                        df_backtest = df_backtest.sort_values('Date')
                        
                        # Split: use last year for backtesting
                        split_date = df_backtest['Date'].max() - pd.Timedelta(days=365)
                        df_train = df_backtest[df_backtest['Date'] < split_date]
                        df_test = df_backtest[df_backtest['Date'] >= split_date]
                        
                        if len(df_test) > 0:
                            # Generate backtest forecast
                            from modeling import forecast_future
                            backtest_forecast, _ = forecast_future(
                                df_train,
                                ticker_viz,
                                periods=len(df_test),
                                model=model_data,
                                add_regressors=st.session_state.add_regressors
                            )
                            
                            if backtest_forecast is not None and len(backtest_forecast) > 0:
                                fig_backtest = go.Figure()
                                
                                # Actual price
                                fig_backtest.add_trace(go.Scatter(
                                    x=df_test['Date'],
                                    y=df_test['Close'],
                                    mode='lines',
                                    name='Harga Aktual',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Predicted price
                                if 'yhat' in backtest_forecast.columns:
                                    backtest_forecast['ds'] = pd.to_datetime(backtest_forecast['ds'])
                                    fig_backtest.add_trace(go.Scatter(
                                        x=backtest_forecast['ds'],
                                        y=backtest_forecast['yhat'],
                                        mode='lines',
                                        name='Harga Prediksi Model',
                                        line=dict(color='red', width=2, dash='dash')
                                    ))
                                
                                fig_backtest.update_layout(
                                    title=f"Backtesting: {ticker_viz} (1 Tahun Terakhir)",
                                    xaxis_title="Date",
                                    yaxis_title="Price (USD)",
                                    hovermode='x unified',
                                    height=500
                                )
                                
                                st.plotly_chart(fig_backtest, width='stretch', key=f'backtest_{ticker_viz}')
                                
                                # Calculate accuracy metrics
                                if 'yhat' in backtest_forecast.columns and len(backtest_forecast) == len(df_test):
                                    actual = df_test['Close'].values
                                    predicted = backtest_forecast['yhat'].values
                                    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
                                    
                                    if mape < 5:
                                        safe_success(f"✅ **Key Insight**: Model sangat akurat! MAPE: {mape:.2f}%. Prediksi mengikuti tren aktual dengan sangat baik, sehingga prediksi ke depan dapat dipercaya.")
                                    elif mape < 10:
                                        safe_info(f"ℹ️ **Key Insight**: Model cukup akurat. MAPE: {mape:.2f}%. Prediksi mengikuti tren aktual dengan baik.")
                                    else:
                                        safe_warning(f"⚠️ **Key Insight**: Model memiliki akurasi sedang. MAPE: {mape:.2f}%. Gunakan prediksi dengan hati-hati dan pertimbangkan faktor eksternal.")
                        else:
                            safe_warning("Tidak cukup data untuk backtesting. Perlu minimal 1 tahun data.")
                    else:
                        safe_warning("Tidak dapat membuat backtesting. Pastikan data memiliki kolom Date, Close, dan model tersedia.")
                except Exception as e:
                    safe_error(f"Error membuat backtesting: {str(e)}")
                
                # 3. Analisa Musiman (Seasonality Heatmap)
                st.subheader("3. Analisa Musiman: Seasonal Pattern Analysis")
                try:
                    if 'Date' in df_filtered_forecast.columns and 'Close' in df_filtered_forecast.columns:
                        df_seasonal = df_filtered_forecast.copy()
                        df_seasonal['Date'] = pd.to_datetime(df_seasonal['Date'])
                        df_seasonal = df_seasonal.sort_values('Date')
                        df_seasonal['Year'] = df_seasonal['Date'].dt.year
                        df_seasonal['Month'] = df_seasonal['Date'].dt.month
                        df_seasonal['Return'] = df_seasonal['Close'].pct_change() * 100
                        
                        # Create monthly average returns
                        monthly_returns = df_seasonal.groupby('Month')['Return'].mean()
                        
                        # Create heatmap data
                        years = sorted(df_seasonal['Year'].unique())
                        months = range(1, 13)
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        heatmap_data = []
                        for year in years:
                            year_data = df_seasonal[df_seasonal['Year'] == year]
                            month_returns = []
                            for month in months:
                                month_data = year_data[year_data['Month'] == month]
                                if len(month_data) > 0:
                                    month_returns.append(month_data['Return'].mean())
                                else:
                                    month_returns.append(0)
                            heatmap_data.append(month_returns)
                        
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=heatmap_data,
                            x=month_names,
                            y=[str(y) for y in years],
                            colorscale='RdYlGn',
                            zmid=0,
                            colorbar=dict(title="Return (%)")
                        ))
                        
                        fig_heatmap.update_layout(
                            title=f"Seasonal Return Heatmap: {ticker_viz}",
                            xaxis_title="Month",
                            yaxis_title="Year",
                            height=600
                        )
                        
                        st.plotly_chart(fig_heatmap, width='stretch', key=f'seasonal_heatmap_{ticker_viz}')
                        
                        # Key Insight: Best and worst months
                        best_month = monthly_returns.idxmax()
                        worst_month = monthly_returns.idxmin()
                        best_return = monthly_returns.max()
                        worst_return = monthly_returns.min()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Bulan Terbaik", month_names[best_month-1], f"{best_return:.2f}%")
                        with col2:
                            st.metric("Bulan Terburuk", month_names[worst_month-1], f"{worst_return:.2f}%")
                        
                        safe_info(f"💡 **Key Insight**: Rata-rata return terbaik di bulan {month_names[best_month-1]} ({best_return:.2f}%) dan terburuk di bulan {month_names[worst_month-1]} ({worst_return:.2f}%). Gunakan pola ini untuk market timing.")
                    else:
                        safe_warning("Tidak dapat membuat seasonal heatmap. Pastikan data memiliki kolom Date dan Close.")
                except Exception as e:
                    safe_error(f"Error membuat seasonal heatmap: {str(e)}")
                
                # 4. Skenario Bull vs Bear (Optimis vs Pesimis)
                st.subheader("4. Skenario Bull vs Bear: Optimis vs Pesimis")
                try:
                    if 'yhat' in forecast_data.columns and 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
                        forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
                        
                        fig_scenarios = go.Figure()
                        
                        # Bullish scenario (upper bound)
                        fig_scenarios.add_trace(go.Scatter(
                            x=forecast_data['ds'],
                            y=forecast_data['yhat_upper'],
                            mode='lines',
                            name='Bullish (Optimis)',
                            line=dict(color='green', width=2, dash='dot')
                        ))
                        
                        # Base scenario
                        fig_scenarios.add_trace(go.Scatter(
                            x=forecast_data['ds'],
                            y=forecast_data['yhat'],
                            mode='lines',
                            name='Base (Netral)',
                            line=dict(color='gray', width=2)
                        ))
                        
                        # Bearish scenario (lower bound)
                        fig_scenarios.add_trace(go.Scatter(
                            x=forecast_data['ds'],
                            y=forecast_data['yhat_lower'],
                            mode='lines',
                            name='Bearish (Pesimis)',
                            line=dict(color='red', width=2, dash='dot')
                        ))
                        
                        fig_scenarios.update_layout(
                            title=f"Bull vs Bear Scenarios: {ticker_viz}",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_scenarios, width='stretch', key=f'scenarios_{ticker_viz}')
                        
                        # Key Insight
                        price_change_bull = ((forecast_data['yhat_upper'].iloc[-1] - forecast_data['yhat'].iloc[0]) / forecast_data['yhat'].iloc[0]) * 100
                        price_change_bear = ((forecast_data['yhat_lower'].iloc[-1] - forecast_data['yhat'].iloc[0]) / forecast_data['yhat'].iloc[0]) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Bullish Scenario", f"{forecast_data['yhat_upper'].iloc[-1]:.2f}", f"{price_change_bull:+.1f}%")
                        with col2:
                            st.metric("Base Scenario", f"{forecast_data['yhat'].iloc[-1]:.2f}", "Base")
                        with col3:
                            st.metric("Bearish Scenario", f"{forecast_data['yhat_lower'].iloc[-1]:.2f}", f"{price_change_bear:+.1f}%")
                        
                        safe_info(f"💡 **Key Insight**: Skenario Bull menunjukkan potensi kenaikan {price_change_bull:+.1f}%, sedangkan Bear menunjukkan risiko penurunan {abs(price_change_bear):.1f}%. Gunakan untuk risk management dan stop loss.")
                    else:
                        safe_warning("Tidak dapat membuat skenario Bull vs Bear. Pastikan forecast memiliki yhat, yhat_lower, dan yhat_upper.")
                except Exception as e:
                    safe_error(f"Error membuat skenario Bull vs Bear: {str(e)}")
                
                # 5. Indikator Teknikal (Moving Average Overlay)
                st.subheader("5. Indikator Teknikal: Moving Average Overlay")
                try:
                    if 'Date' in df_filtered_forecast.columns and 'Close' in df_filtered_forecast.columns:
                        df_tech = df_filtered_forecast.copy()
                        df_tech['Date'] = pd.to_datetime(df_tech['Date'])
                        df_tech = df_tech.sort_values('Date')
                        
                        # Get last year + forecast
                        last_date = df_tech['Date'].max()
                        one_year_ago = last_date - pd.Timedelta(days=365)
                        df_tech_recent = df_tech[df_tech['Date'] >= one_year_ago]
                        
                        # Combine historical and forecast
                        forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
                        df_combined = pd.concat([
                            df_tech_recent[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}),
                            forecast_data[['ds', 'yhat']].rename(columns={'yhat': 'y'})
                        ], ignore_index=True)
                        df_combined = df_combined.sort_values('ds')
                        
                        # Calculate MAs
                        df_combined['MA50'] = df_combined['y'].rolling(window=50).mean()
                        df_combined['MA200'] = df_combined['y'].rolling(window=200).mean()
                        
                        fig_ma = go.Figure()
                        
                        # Price line
                        fig_ma.add_trace(go.Scatter(
                            x=df_combined['ds'],
                            y=df_combined['y'],
                            mode='lines',
                            name='Price (Historical + Forecast)',
                            line=dict(color='black', width=2)
                        ))
                        
                        # MA50
                        fig_ma.add_trace(go.Scatter(
                            x=df_combined['ds'],
                            y=df_combined['MA50'],
                            mode='lines',
                            name='MA50',
                            line=dict(color='orange', width=2)
                        ))
                        
                        # MA200
                        fig_ma.add_trace(go.Scatter(
                            x=df_combined['ds'],
                            y=df_combined['MA200'],
                            mode='lines',
                            name='MA200',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Detect Golden Cross / Death Cross
                        if len(df_combined) > 200:
                            last_ma50 = df_combined['MA50'].iloc[-1]
                            last_ma200 = df_combined['MA200'].iloc[-1]
                            prev_ma50 = df_combined['MA50'].iloc[-2] if len(df_combined) > 1 else last_ma50
                            prev_ma200 = df_combined['MA200'].iloc[-2] if len(df_combined) > 1 else last_ma200
                            
                            # Golden Cross: MA50 crosses above MA200
                            if prev_ma50 <= prev_ma200 and last_ma50 > last_ma200:
                                fig_ma.add_annotation(
                                    x=df_combined['ds'].iloc[-1],
                                    y=df_combined['y'].iloc[-1],
                                    text="🟢 GOLDEN CROSS - BUY SIGNAL",
                                    showarrow=True,
                                    arrowhead=2,
                                    bgcolor="green",
                                    font=dict(color="white", size=14)
                                )
                                signal = "BUY"
                            # Death Cross: MA50 crosses below MA200
                            elif prev_ma50 >= prev_ma200 and last_ma50 < last_ma200:
                                fig_ma.add_annotation(
                                    x=df_combined['ds'].iloc[-1],
                                    y=df_combined['y'].iloc[-1],
                                    text="🔴 DEATH CROSS - SELL SIGNAL",
                                    showarrow=True,
                                    arrowhead=2,
                                    bgcolor="red",
                                    font=dict(color="white", size=14)
                                )
                                signal = "SELL"
                            else:
                                signal = "HOLD"
                        else:
                            signal = "N/A"
                        
                        fig_ma.update_layout(
                            title=f"Technical Indicators: {ticker_viz}",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_ma, width='stretch', key=f'ma_{ticker_viz}')
                        
                        # Key Insight
                        if signal == "BUY":
                            safe_success(f"🟢 **Key Insight**: GOLDEN CROSS terdeteksi! MA50 memotong ke atas MA200, sinyal BULLISH. Ini adalah sinyal beli yang kuat dalam technical analysis.")
                        elif signal == "SELL":
                            safe_error(f"🔴 **Key Insight**: DEATH CROSS terdeteksi! MA50 memotong ke bawah MA200, sinyal BEARISH. Pertimbangkan untuk mengurangi posisi atau memasang stop loss.")
                        else:
                            safe_info(f"ℹ️ **Key Insight**: Tidak ada crossover terdeteksi. MA50 dan MA200 masih dalam tren yang sama. Monitor terus untuk sinyal Golden/Death Cross.")
                    else:
                        safe_warning("Tidak dapat membuat technical indicators. Pastikan data memiliki kolom Date dan Close.")
                except Exception as e:
                    safe_error(f"Error membuat technical indicators: {str(e)}")
                
                # 6. Signal Traffic Light (Sinyal Beli/Jual)
                st.subheader("6. Signal Traffic Light: Sinyal Beli/Jual")
                try:
                    if 'yhat' in forecast_data.columns and len(forecast_data) >= 7:
                        # Calculate 1-week forecast change
                        current_price = df_filtered_forecast['Close'].iloc[-1] if 'Close' in df_filtered_forecast.columns else forecast_data['yhat'].iloc[0]
                        week_forecast = forecast_data['yhat'].iloc[6] if len(forecast_data) > 6 else forecast_data['yhat'].iloc[-1]
                        week_change = ((week_forecast - current_price) / current_price) * 100
                        
                        # Calculate probability (based on confidence interval width)
                        if 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
                            ci_width = ((forecast_data['yhat_upper'].iloc[6] - forecast_data['yhat_lower'].iloc[6]) / forecast_data['yhat'].iloc[6]) * 100 if len(forecast_data) > 6 else 10
                            # Narrower CI = higher confidence
                            probability = max(50, 100 - (ci_width * 2))
                        else:
                            probability = 70
                        
                        # Determine signal
                        if week_change > 5:
                            signal_color = "green"
                            signal_text = "🟢 STRONG BUY"
                            signal_desc = f"Prediksi naik {week_change:.1f}% dalam seminggu"
                        elif week_change > 2:
                            signal_color = "lightgreen"
                            signal_text = "🟢 BUY"
                            signal_desc = f"Prediksi naik {week_change:.1f}% dalam seminggu"
                        elif week_change < -5:
                            signal_color = "red"
                            signal_text = "🔴 SELL"
                            signal_desc = f"Prediksi turun {abs(week_change):.1f}% dalam seminggu"
                        elif week_change < -2:
                            signal_color = "orange"
                            signal_text = "🟡 WEAK SELL"
                            signal_desc = f"Prediksi turun {abs(week_change):.1f}% dalam seminggu"
                        else:
                            signal_color = "yellow"
                            signal_text = "🟡 HOLD"
                            signal_desc = "Prediksi datar/sideways"
                        
                        # Display signal box
                        st.markdown(f"""
                        <div style="background-color: {signal_color}; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                            <h2 style="color: {'white' if signal_color in ['red', 'green'] else 'black'}; margin: 0;">
                                {signal_text}
                            </h2>
                            <p style="color: {'white' if signal_color in ['red', 'green'] else 'black'}; font-size: 18px; margin: 10px 0;">
                                {signal_desc}
                            </p>
                            <p style="color: {'white' if signal_color in ['red', 'green'] else 'black'}; font-size: 16px; margin: 5px 0;">
                                Peluang: {probability:.0f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Key Insight
                        if week_change > 5:
                            safe_success(f"✅ **Key Insight**: Sinyal STRONG BUY dengan probabilitas {probability:.0f}%. Prediksi menunjukkan kenaikan signifikan ({week_change:.1f}%) dalam seminggu. Pertimbangkan untuk membuka posisi long.")
                        elif week_change > 2:
                            safe_info(f"ℹ️ **Key Insight**: Sinyal BUY dengan probabilitas {probability:.0f}%. Prediksi menunjukkan kenaikan moderat ({week_change:.1f}%) dalam seminggu.")
                        elif week_change < -5:
                            safe_error(f"❌ **Key Insight**: Sinyal SELL dengan probabilitas {probability:.0f}%. Prediksi menunjukkan penurunan signifikan ({abs(week_change):.1f}%) dalam seminggu. Pertimbangkan untuk mengurangi posisi atau memasang stop loss.")
                        elif week_change < -2:
                            safe_warning(f"⚠️ **Key Insight**: Sinyal WEAK SELL dengan probabilitas {probability:.0f}%. Prediksi menunjukkan penurunan moderat ({abs(week_change):.1f}%) dalam seminggu.")
                        else:
                            safe_info(f"ℹ️ **Key Insight**: Sinyal HOLD. Prediksi menunjukkan pergerakan sideways. Tunggu sinyal yang lebih jelas sebelum mengambil keputusan.")
                    else:
                        safe_warning("Tidak dapat membuat signal traffic light. Pastikan forecast memiliki yhat dan minimal 7 hari ke depan.")
                except Exception as e:
                    safe_error(f"Error membuat signal traffic light: {str(e)}")
            else:
                safe_info("👆 Silakan generate forecast terlebih dahulu untuk melihat visualisasi!")
        else:
            safe_info("👆 Silakan generate forecast terlebih dahulu untuk melihat visualisasi!")

elif page == "📈 Model Evaluation":
    st.header("📈 Model Evaluation & Analysis")
    st.markdown("""
    Halaman ini menyediakan analisis komprehensif dan evaluasi mendalam terhadap model forecasting.
    Termasuk residual analysis, diagnostic tests, dan rekomendasi perbaikan model.
    """)
    
    if (st.session_state.df_processed is not None and 
        hasattr(st.session_state.df_processed, 'columns') and 
        not st.session_state.df_processed.empty):
        
        # Check if models are trained
        if not st.session_state.trained_models:
            safe_warning("⚠️ Silakan train models terlebih dahulu di halaman Forecasting untuk melakukan evaluasi.")
            safe_info("👈 Kembali ke halaman Forecasting dan klik tombol '🚀 Train Models'.")
        else:
            # Select ticker for evaluation
            with st.expander("🔍 Select Model for Evaluation", expanded=True):
                if 'Ticker' in st.session_state.df_processed.columns:
                    tickers = sorted([t for t in st.session_state.trained_models.keys() 
                                    if st.session_state.trained_models[t].get('success', False)])
                    if tickers:
                        selected_ticker_eval = st.selectbox(
                            "Pilih Ticker untuk Evaluasi",
                            tickers,
                            help="Pilih ticker yang sudah di-train untuk melihat evaluasi detail",
                            key="eval_ticker"
                        )
                    else:
                        selected_ticker_eval = None
                        safe_warning("Tidak ada model yang berhasil di-train.")
                else:
                    selected_ticker_eval = 'ALL' if 'ALL' in st.session_state.trained_models else None
            
            if selected_ticker_eval and selected_ticker_eval in st.session_state.trained_models:
                model_result = st.session_state.trained_models[selected_ticker_eval]
                
                if model_result.get('success', False):
                    try:
                        # Prepare data for evaluation
                        from modeling import prepare_prophet_data
                        prophet_df, _ = prepare_prophet_data(
                            st.session_state.df_processed, 
                            ticker=selected_ticker_eval if selected_ticker_eval != 'ALL' else None,
                            use_log_transform=True,
                            add_technical_indicators=True
                        )
                        
                        # Get split date
                        split_date = pd.Timestamp(st.session_state.split_date_str)
                        
                        # Split data
                        train_df = prophet_df[prophet_df['ds'] < split_date].copy()
                        test_df = prophet_df[prophet_df['ds'] >= split_date].copy()
                        
                        if len(train_df) > 0 and len(test_df) > 0:
                            # Load model
                            model = load_model(selected_ticker_eval)
                            
                            if model:
                                # Perform comprehensive evaluation
                                with st.spinner("Menghitung evaluasi model..."):
                                    eval_results = evaluate_model_performance(
                                        model=model,
                                        train_df=train_df,
                                        test_df=test_df
                                    )
                                
                                # Display metrics
                                st.markdown("---")
                                st.subheader("📊 Model Performance Metrics")
                                
                                metrics = eval_results['metrics']
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                                    st.caption("Root Mean Squared Error")
                                
                                with col2:
                                    st.metric("MAE", f"{metrics['MAE']:.4f}")
                                    st.caption("Mean Absolute Error")
                                
                                with col3:
                                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                                    st.caption("Mean Absolute Percentage Error")
                                
                                with col4:
                                    st.metric("R²", f"{metrics['R2']:.4f}")
                                    st.caption("Coefficient of Determination")
                                
                                # Advanced metrics
                                st.markdown("#### Advanced Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    mase = metrics['MASE']
                                    st.metric("MASE", f"{mase:.4f}")
                                    if mase < 1:
                                        st.success("✅ Better than naive")
                                    elif mase == 1:
                                        st.info("⚖️ Equal to naive")
                                    else:
                                        st.error("❌ Worse than naive")
                                    st.caption("Mean Absolute Scaled Error")
                                
                                with col2:
                                    da = metrics['Directional_Accuracy']
                                    st.metric("Directional Accuracy", f"{da:.2f}%")
                                    if da > 60:
                                        st.success("✅ Good")
                                    elif da > 50:
                                        st.info("⚖️ Acceptable")
                                    else:
                                        st.error("❌ Poor")
                                    st.caption("Direction Prediction Accuracy")
                                
                                with col3:
                                    st.metric("Theil's U", f"{metrics['Theil_U']:.4f}")
                                    st.caption("Normalized RMSE")
                                
                                with col4:
                                    st.metric("Mean Error", f"{metrics['Mean_Error']:.4f}")
                                    st.caption("Bias (ideal: 0)")
                                
                                # Residual Analysis
                                st.markdown("---")
                                st.subheader("🔍 Residual Analysis")
                                
                                residuals_info = eval_results['residuals']
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Mean", f"{residuals_info['mean']:.4f}")
                                    st.caption("Ideal: 0")
                                
                                with col2:
                                    st.metric("Std Dev", f"{residuals_info['std']:.4f}")
                                
                                with col3:
                                    st.metric("Min", f"{residuals_info['min']:.4f}")
                                
                                with col4:
                                    st.metric("Max", f"{residuals_info['max']:.4f}")
                                
                                # Diagnostic Tests
                                st.markdown("#### Diagnostic Tests")
                                diagnostics = eval_results['diagnostics']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Normality Test (Jarque-Bera):**")
                                    if diagnostics['is_normal']:
                                        st.success(f"✅ Residuals are normally distributed (p-value: {diagnostics['jb_pvalue']:.4f})")
                                    else:
                                        st.warning(f"⚠️ Residuals are NOT normally distributed (p-value: {diagnostics['jb_pvalue']:.4f})")
                                
                                with col2:
                                    st.markdown("**Autocorrelation Test (Ljung-Box):**")
                                    if diagnostics['has_autocorrelation']:
                                        st.warning(f"⚠️ Residuals have autocorrelation (p-value: {diagnostics['lb_pvalue']:.4f})")
                                    else:
                                        st.success(f"✅ No significant autocorrelation (p-value: {diagnostics['lb_pvalue']:.4f})")
                                
                                # Visualizations
                                st.markdown("---")
                                st.subheader("📈 Visualization")
                                
                                # Residual Analysis Plot
                                try:
                                    forecast_df = eval_results.get('forecast_df')
                                    if forecast_df is not None:
                                        y_true = test_df['y'].values
                                        y_pred = forecast_df['yhat'].values[:len(y_true)]
                                        
                                        fig_residual = plot_residual_analysis(
                                            y_true, 
                                            y_pred,
                                            title=f"Residual Analysis - {selected_ticker_eval}"
                                        )
                                        st.plotly_chart(fig_residual, use_container_width=True)
                                except Exception as e:
                                    safe_warning(f"Tidak dapat membuat residual analysis plot: {str(e)}")
                                
                                # Forecast Accuracy by Horizon
                                try:
                                    fig_horizon = plot_forecast_accuracy_by_horizon(
                                        eval_results,
                                        title=f"Forecast Accuracy by Horizon - {selected_ticker_eval}"
                                    )
                                    st.plotly_chart(fig_horizon, use_container_width=True)
                                except Exception as e:
                                    safe_warning(f"Tidak dapat membuat horizon analysis plot: {str(e)}")
                                
                                
                            else:
                                safe_error(f"Model untuk {selected_ticker_eval} tidak ditemukan.")
                        else:
                            safe_warning("Data train/test tidak mencukupi untuk evaluasi.")
                    except Exception as e:
                        safe_error(f"Error saat evaluasi model: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    safe_error(f"Model untuk {selected_ticker_eval} gagal di-train: {model_result.get('error', 'Unknown error')}")
            else:
                safe_info("👆 Silakan pilih ticker untuk evaluasi.")
    else:
        safe_warning("⚠️ Upload dan proses file terlebih dahulu untuk mengakses halaman ini.")
        safe_info("👈 Silakan upload file CSV di halaman ETL Results untuk memulai analisis.")

else:
    # Welcome screen
    safe_info("👈 Silakan upload file CSV di sidebar untuk memulai analisis.")
    
    st.markdown("""
    ### Fitur FinScope:
    
    1. **ETL Pipeline**: Load dan clean data dengan chunking untuk handle jutaan rows
    2. **Exploratory Data Analysis**: 5 visualisasi utama (Line Chart, Histogram, Scatter, Heatmap, Box Plot)
    3. **Prophet Forecasting**: Time series forecasting dengan regressors
    4. **Advanced Visualizations**: 6+ visualisasi interaktif dengan Plotly
    
    ### Format Data yang Diperlukan:
    - **Date**: Kolom tanggal (format: YYYY-MM-DD)
    - **Ticker**: Symbol ticker (optional, untuk multi-ticker)
    - **Open, High, Low, Close**: Harga OHLC
    - **Volume**: Volume trading
    - **ROE, Debt_Equity, EBIT_Margin**: Financial ratios (optional tapi direkomendasikan)
    """)
