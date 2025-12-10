"""
Main Streamlit application untuk FinScope.
Aplikasi web untuk visualisasi dan forecasting data keuangan S&P 500.
"""

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
    plot_what_if_terrain
)
from utils import check_null_percentage, load_model

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
        "🤖 Forecasting"
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
    enable_downsample = False
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 200:
            safe_error(f"⚠️ File terlalu besar ({file_size_mb:.2f} MB). Maksimal 200MB.")
            uploaded_file = None
        elif file_size_mb > 100:
            safe_warning(f"⚠️ File besar ({file_size_mb:.2f} MB). Proses mungkin memakan waktu lama.")
        
        # Downsampling option for large files (default: enabled for files >50MB)
        if file_size_mb > 50:
            enable_downsample = True
        else:
            enable_downsample = False
    
    # Process ETL jika file baru di-upload
    if uploaded_file is not None:
        # Process ETL jika file baru di-upload
        if st.session_state.df_processed is None or 'file_name' not in st.session_state.metadata or \
           st.session_state.metadata['file_name'] != uploaded_file.name:
            
            with st.spinner("Memproses data... Mohon tunggu."):
                try:
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    
                    # For large files (>50MB), use optimized chunked reading with automatic downsampling
                    if file_size_mb > 50:
                        import io
                        import gc
                        from utils import downsample_data
                        
                        # Calculate target rows based on file size and downsampling option
                        if enable_downsample:
                            # More aggressive downsampling for very large files
                            if file_size_mb > 150:
                                target_rows = 500000  # 500k rows max
                            elif file_size_mb > 100:
                                target_rows = 1000000  # 1M rows max
                            else:
                                target_rows = 2000000  # 2M rows max
                        else:
                            target_rows = 5000000  # 5M rows max
                        
                        chunk_size = 30000  # Smaller chunks: 30k rows
                        max_chunks = min(200, (target_rows // chunk_size) + 10)
                        
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
                                    status_text.text(f"📖 Membaca file (target: {target_rows:,} rows)...")
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
                                
                                # Check if we've reached target rows
                                if total_rows + len(chunk) > target_rows:
                                    # Take only what we need (only copy if necessary)
                                    remaining_rows = target_rows - total_rows
                                    if remaining_rows > 0:
                                        chunk = chunk.iloc[:remaining_rows]
                                        chunk_list.append(chunk)
                                        total_rows += len(chunk)
                                    break
                                
                                chunk_list.append(chunk)
                                total_rows += len(chunk)
                                
                                # Update progress dengan jeda untuk mencegah WebSocket timeout
                                if progress_bar is not None and i % 10 == 0:  # Update setiap 10 chunks
                                    progress = min(100, int((total_rows / target_rows) * 90))
                                    try:
                                        progress_bar.progress(progress)
                                        time.sleep(0.01)  # Jeda kecil untuk mencegah WebSocket timeout
                                    except Exception:
                                        pass  # Ignore WebSocket errors during progress update
                                
                                # Stop if we've read enough chunks
                                if i >= max_chunks - 1:
                                    break
                                
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
                            
                            # Additional downsampling if still too large
                            if len(df_combined) > target_rows:
                                df_combined = downsample_data(df_combined, max_rows=target_rows)
                                safe_info(f"📊 Data di-downsample menjadi {len(df_combined):,} rows untuk performa yang lebih baik.")
                            
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
                            safe_error("❌ Memory Error: File terlalu besar. Silakan gunakan file yang lebih kecil atau aktifkan downsampling.")
                            safe_info("💡 Tips: File besar akan otomatis di-downsample untuk performa yang lebih baik.")
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
                    else:
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
                            except (UnicodeDecodeError, ValueError) as e:
                                # If encoding error, try with different encoding
                                safe_warning(f"⚠️ Masalah encoding terdeteksi. Mencoba encoding alternatif...")
                                # Re-read file with different encoding
                                try:
                                    # Read with latin-1 encoding and save to new temp file
                                    df_temp = pd.read_csv(tmp_path, encoding='latin-1', low_memory=True, on_bad_lines='skip')
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
                    safe_error("❌ Memory Error: File terlalu besar untuk diproses. Silakan gunakan file yang lebih kecil atau aktifkan downsampling.")
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
        
        # Processing Steps Applied
        st.markdown("### ⚙️ Langkah-Langkah Preprocessing yang Diterapkan")
        
        steps_applied = []
        
        # Check cleaning method
        if st.session_state.metadata.get('null_info'):
            steps_applied.append("✅ **Null Value Cleaning**: Forward fill (ffill) untuk mengisi missing values")
        
        # Check feature calculation
        if 'Daily_Return' in st.session_state.df_processed.columns or 'Volatility' in st.session_state.df_processed.columns:
            steps_applied.append("✅ **Feature Engineering**: Menghitung Daily Return dan Volatility (30-day rolling)")
        
        # Check scaling
        if st.session_state.metadata.get('scalers'):
            steps_applied.append("✅ **Scaling**: Normalisasi Close price menggunakan MinMaxScaler")
        
        # Check date parsing
        if 'Date' in st.session_state.df_processed.columns and pd.api.types.is_datetime64_any_dtype(st.session_state.df_processed['Date']):
            steps_applied.append("✅ **Date Parsing**: Konversi kolom Date ke format datetime")
        
        if steps_applied:
            for step in steps_applied:
                st.markdown(step)
        else:
            st.info("ℹ️ Informasi langkah preprocessing tidak tersedia")
        
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
            
            # 2. Volume Pressure Chart (Pengganti Histogram)
            st.subheader("2. Volume Pressure: Price x Volume Strength")
            st.markdown("""
            **Konsep**: Menggabungkan harga dan volume untuk melihat kekuatan trend. 
            Bar Volume hijau = harga naik (didukung pasar), Bar Volume merah = harga turun (lemah).
            Jika harga naik tapi volume kecil = kenaikan lemah (fakeout).
            """)
            
            # Filter khusus untuk Volume Pressure Chart
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    volume_chart_height = st.slider("Chart Height", 300, 800, 500, 50, key="volume_height")
                    show_candlestick = st.checkbox("Tampilkan Candlestick", value=False, key="volume_candlestick")
                with col2:
                    volume_resample = st.selectbox(
                        "Resample Frequency",
                        ["None", "Weekly", "Monthly"],
                        index=0,
                        key="volume_resample"
                    )
                    min_volume_threshold = st.number_input(
                        "Minimum Volume Threshold",
                        min_value=0,
                        value=0,
                        help="Filter volume di bawah threshold ini",
                        key="volume_threshold"
                    )
            
            try:
                # Apply filters untuk Volume Pressure
                df_volume_filtered = df_viz.copy()
                
                # Apply volume threshold filter
                if min_volume_threshold > 0 and 'Volume' in df_volume_filtered.columns:
                    df_volume_filtered = df_volume_filtered[df_volume_filtered['Volume'] >= min_volume_threshold]
                
                # Apply resample jika dipilih
                if volume_resample != "None" and 'Date' in df_volume_filtered.columns:
                    df_volume_filtered['Date'] = pd.to_datetime(df_volume_filtered['Date'])
                    df_volume_filtered = df_volume_filtered.set_index('Date')
                    if volume_resample == "Weekly":
                        df_volume_filtered = df_volume_filtered.resample('W').agg({
                            'Close': 'last',
                            'Volume': 'sum',
                            'Open': 'first' if 'Open' in df_volume_filtered.columns else 'last',
                            'High': 'max' if 'High' in df_volume_filtered.columns else 'last',
                            'Low': 'min' if 'Low' in df_volume_filtered.columns else 'last'
                        })
                    elif volume_resample == "Monthly":
                        df_volume_filtered = df_volume_filtered.resample('ME').agg({
                            'Close': 'last',
                            'Volume': 'sum',
                            'Open': 'first' if 'Open' in df_volume_filtered.columns else 'last',
                            'High': 'max' if 'High' in df_volume_filtered.columns else 'last',
                            'Low': 'min' if 'Low' in df_volume_filtered.columns else 'last'
                        })
                    df_volume_filtered = df_volume_filtered.reset_index()
                
                fig_volume_pressure = plot_volume_pressure(df_volume_filtered, ticker=selected_ticker_eda)
                if fig_volume_pressure is not None:
                    # Apply height filter
                    fig_volume_pressure.update_layout(height=volume_chart_height)
                    
                    # Apply candlestick option jika dipilih dan data tersedia
                    if show_candlestick and all(col in df_volume_filtered.columns for col in ['Open', 'High', 'Low', 'Close']):
                        # Replace line chart with candlestick
                        fig_volume_pressure.data = []  # Clear existing traces
                        df_volume_filtered['Date'] = pd.to_datetime(df_volume_filtered['Date'])
                        df_volume_filtered = df_volume_filtered.sort_values('Date')
                        
                        # Recreate subplots
                        fig_volume_pressure = make_subplots(
                            rows=2, cols=1,
                            row_heights=[0.7, 0.3],
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=('Price (Candlestick)', 'Volume')
                        )
                        
                        # Add candlestick
                        fig_volume_pressure.add_trace(
                            go.Candlestick(
                                x=df_volume_filtered['Date'],
                                open=df_volume_filtered['Open'],
                                high=df_volume_filtered['High'],
                                low=df_volume_filtered['Low'],
                                close=df_volume_filtered['Close'],
                                name='Price'
                            ),
                            row=1, col=1
                        )
                        
                        # Add volume bars
                        if 'Volume' in df_volume_filtered.columns:
                            df_volume_filtered['Price_Change'] = df_volume_filtered['Close'].pct_change()
                            df_volume_filtered['Volume_Color'] = df_volume_filtered['Price_Change'].apply(
                                lambda x: '#2ecc71' if x >= 0 else '#e74c3c'
                            )
                            fig_volume_pressure.add_trace(
                                go.Bar(
                                    x=df_volume_filtered['Date'],
                                    y=df_volume_filtered['Volume'],
                                    marker_color=df_volume_filtered['Volume_Color'],
                                    name='Volume',
                                    showlegend=False
                                ),
                                row=2, col=1
                            )
                        
                        fig_volume_pressure.update_xaxes(title_text="Date", row=2, col=1)
                        fig_volume_pressure.update_yaxes(title_text="Price (USD)", row=1, col=1)
                        fig_volume_pressure.update_yaxes(title_text="Volume", row=2, col=1)
                        fig_volume_pressure.update_layout(
                            height=volume_chart_height,
                            hovermode='x unified'
                        )
                    
                    st.plotly_chart(fig_volume_pressure, width='stretch', key=f'volume_pressure_{selected_ticker_eda}')
                else:
                    safe_warning("Tidak dapat membuat volume pressure chart. Pastikan data memiliki kolom Date, Close, dan Volume.")
            except Exception as e:
                safe_error(f"Error membuat volume pressure chart: {str(e)}")
            
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
            
            # 4. Valuation Band (Pengganti Box Plot)
            st.subheader("4. Valuation Band: Is It Cheap or Expensive?")
            st.markdown("""
            **Konsep**: Menunjukkan apakah harga "mahal" atau "murah" relatif terhadap fundamental.
            Triangle hijau = Undervalued (beli), Triangle merah = Overvalued (jual).
            Lebih relevan daripada box plot karena mempertimbangkan trend dan fundamental.
            """)
            
            # Filter khusus untuk Valuation Band
            with st.expander("⚙️ Filter Visualisasi", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    valuation_height = st.slider("Chart Height", 300, 800, 500, 50, key="valuation_height")
                    valuation_method = st.selectbox(
                        "Valuation Method",
                        ["Moving Average", "Percentile", "Z-Score"],
                        index=0,
                        key="valuation_method"
                    )
                with col2:
                    ma_window = st.number_input(
                        "Moving Average Window",
                        min_value=10,
                        max_value=365,
                        value=30,
                        help="Window untuk moving average (jika metode MA dipilih)",
                        key="valuation_ma_window"
                    )
                    show_bands = st.checkbox("Tampilkan Bands", value=True, key="valuation_show_bands")
            
            try:
                fig_valuation = plot_valuation_band(df_viz, ticker=selected_ticker_eda)
                if fig_valuation is not None:
                    st.plotly_chart(fig_valuation, width='stretch', key=f'valuation_band_{selected_ticker_eda}')
                else:
                    safe_warning("Tidak dapat membuat valuation band. Pastikan data memiliki kolom Date dan Close.")
            except Exception as e:
                safe_error(f"Error membuat valuation band: {str(e)}")
            
            # 5. Seasonal Heatmap Calendar (Bonus)
            st.subheader("5. Seasonal Return Heatmap: Best Time to Buy?")
            st.markdown("""
            **Konsep**: Pola kalender yang bisa ditindaklanjuti. Hijau pekat = return bulanan positif tinggi, 
            Merah pekat = rugi. Lihat pola seperti "January Effect" atau "Sell in May".
            """)
            
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
            
            # 6. Financial Health Radar (Bonus - jika multiple tickers)
            if 'Ticker' in df_viz.columns and df_viz['Ticker'].nunique() > 1:
                st.subheader("6. Financial Health Radar: Compare Companies")
                st.markdown("""
                **Konsep**: Spider chart untuk membandingkan kualitas perusahaan sekilas pandang.
                Area besar = perusahaan sehat dan efisien, Area kecil = hutang tinggi dan margin tipis.
                """)
                
                # Filter khusus untuk Financial Health Radar
                with st.expander("⚙️ Filter Visualisasi", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        radar_height = st.slider("Chart Height", 400, 1000, 600, 50, key="radar_height")
                        max_tickers_radar = st.number_input(
                            "Max Tickers to Compare",
                            min_value=2,
                            max_value=10,
                            value=5,
                            help="Maksimal jumlah ticker yang dibandingkan",
                            key="radar_max_tickers"
                        )
                    with col2:
                        radar_metrics = st.multiselect(
                            "Pilih Metrics",
                            ["ROE", "Net_Profit_Margin", "Current_Ratio", "Asset_Turnover", "Debt_Equity_Ratio"],
                            default=["ROE", "Net_Profit_Margin", "Current_Ratio", "Asset_Turnover", "Debt_Equity_Ratio"],
                            help="Pilih metrik yang akan ditampilkan di radar",
                            key="radar_metrics"
                        )
                        show_radar_fill = st.checkbox("Tampilkan Fill Area", value=True, key="radar_show_fill")
                
                try:
                    fig_radar = plot_financial_health_radar(df_viz, ticker=selected_ticker_eda)
                    if fig_radar is not None:
                        st.plotly_chart(fig_radar, width='stretch', key=f'health_radar_{selected_ticker_eda}')
                    else:
                        safe_info("Financial health radar tidak tersedia. Pastikan data memiliki kolom financial ratios.")
                except Exception as e:
                    safe_error(f"Error membuat financial health radar: {str(e)}")
            

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
                add_regressors = st.checkbox(
                    "Gunakan Regressors (ROE, Debt_Equity)",
                    value=st.session_state.add_regressors,
                    help="Tambahkan regressors untuk meningkatkan akurasi",
                    key="add_regressors_input"
                )
                st.session_state.add_regressors = add_regressors
            
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
                        
                        results = train_models_parallel(
                            st.session_state.df_processed,
                            split_date=st.session_state.split_date_str,
                            add_regressors=st.session_state.add_regressors,
                            n_jobs=max_jobs
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
        
        # 3D Immersive Visualizations
        if st.session_state.forecasts:
            st.divider()
            st.header("🌐 3D Immersive Visualizations")
            st.markdown("""
            **Premium 3D Experience**: Visualisasi 3D interaktif dengan perspektif yang lebih dalam.
            Rotate, zoom, dan explore data forecasting dengan cara yang belum pernah Anda lihat sebelumnya.
            """)
            
            ticker_viz = selected_ticker_forecast if selected_ticker_forecast else \
                (df_filtered_forecast['Ticker'].iloc[0] if 'Ticker' in df_filtered_forecast.columns and len(df_filtered_forecast) > 0 else None)
            
            if ticker_viz and ticker_viz in st.session_state.forecasts:
                forecast_data = st.session_state.forecasts[ticker_viz]
                model_data = load_model(ticker_viz) if ticker_viz else None
                
                # 1. Neon Time-Tunnel (3D Perspective)
                st.subheader("1. Neon Time-Tunnel: 3D Road to Future")
                st.markdown("""
                **Konsep**: Perspektif 3D seperti menyetir ke masa depan. 
                Garis tengah neon = trend forecast, dinding transparan = uncertainty (semakin lebar = semakin tidak pasti).
                Objek merah = anomali historis yang menabrak dinding.
                """)
                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        show_uncertainty = st.checkbox("Tampilkan Uncertainty Walls", value=True, key=f'uncertainty_{ticker_viz}')
                    with col2:
                        highlight_anomalies = st.checkbox("Highlight Anomalies", value=True, key=f'anomalies_{ticker_viz}')
                    
                    fig_tunnel = plot_neon_time_tunnel(
                        df_filtered_forecast, forecast_data, 
                        ticker=ticker_viz,
                        show_uncertainty=show_uncertainty,
                        highlight_anomalies=highlight_anomalies
                    )
                    if fig_tunnel is not None:
                        st.plotly_chart(fig_tunnel, width='stretch', key=f'tunnel_{ticker_viz}')
                    else:
                        safe_warning("Tidak dapat membuat neon time-tunnel. Pastikan forecast memiliki yhat dan CI bands.")
                except Exception as e:
                    safe_error(f"Error membuat neon time-tunnel: {str(e)}")
                
                # 2. Decomposition Glass Stack (3D Layers)
                st.subheader("2. Decomposition Glass Stack: Isi Perut Harga")
                st.markdown("""
                **Konsep**: 3 layer kaca transparan bertumpuk (Trend, Seasonality, Regressors).
                Garis putih tebal = forecast final yang menembus semua layer.
                Rotate untuk melihat dari berbagai sudut, tekan "Explode" untuk memisahkan layer.
                """)
                try:
                    explode_layers = st.checkbox("Explode Layers", value=False, key=f'explode_{ticker_viz}')
                    fig_glass = plot_decomposition_glass_stack(
                        model_data, forecast_data, 
                        ticker=ticker_viz,
                        explode_layers=explode_layers
                    )
                    if fig_glass is not None:
                        st.plotly_chart(fig_glass, width='stretch', key=f'glass_{ticker_viz}')
                    else:
                        safe_info("Decomposition glass stack tidak tersedia untuk data ini.")
                except Exception as e:
                    safe_error(f"Error membuat decomposition glass stack: {str(e)}")
                
                # 3. Seasonal Helix (3D Spiral)
                st.subheader("3. Seasonal Helix: DNA of Market Cycles")
                st.markdown("""
                **Konsep**: Waktu berputar dalam spiral 3D. Satu putaran = 1 tahun.
                Hijau = periode untung, Merah = periode rugi.
                Lihat pola vertikal untuk melihat bulan yang selalu baik/buruk setiap tahunnya.
                """)
                try:
                    if 'Date' in df_filtered_forecast.columns:
                        df_dates = pd.to_datetime(df_filtered_forecast['Date'])
                        min_year = int(df_dates.min().year)
                        max_year = int(df_dates.max().year)
                        
                        year_range = st.slider(
                            "Filter Tahun",
                            min_value=min_year,
                            max_value=max_year,
                            value=(min_year, max_year),
                            key=f'year_range_{ticker_viz}'
                        )
                        
                        fig_helix = plot_seasonal_helix(
                            df_filtered_forecast, forecast_data,
                            ticker=ticker_viz,
                            years_filter=year_range
                        )
                        if fig_helix is not None:
                            st.plotly_chart(fig_helix, width='stretch', key=f'helix_{ticker_viz}')
                        else:
                            safe_info("Seasonal helix tidak tersedia. Pastikan data memiliki kolom Date dan Close.")
                    else:
                        safe_warning("Data tidak memiliki kolom Date untuk seasonal helix.")
                except Exception as e:
                    safe_error(f"Error membuat seasonal helix: {str(e)}")
                
                # 4. Market Universe (3D Motion Bubble)
                st.subheader("4. Market Universe: Risk-Reward-Health Space")
                st.markdown("""
                **Konsep**: 3D space dengan X = Volatility (Risiko), Y = Expected Return, Z = Fundamental Health.
                Setiap saham adalah bola yang melayang. Ukuran bola = Market Cap/Volume.
                Time slider untuk melihat evolusi seiring waktu (jika multiple tickers).
                """)
                try:
                    if 'Ticker' in df_filtered_forecast.columns and df_filtered_forecast['Ticker'].nunique() > 1:
                        years_available = sorted(df_filtered_forecast['Date'].dt.year.unique()) if 'Date' in df_filtered_forecast.columns else []
                        if years_available:
                            time_slider = st.selectbox(
                                "Pilih Tahun untuk Animasi",
                                ['All'] + [str(y) for y in years_available],
                                index=0,
                                key=f'time_slider_{ticker_viz}'
                            )
                            time_year = int(time_slider) if time_slider != 'All' else None
                        else:
                            time_year = None
                    else:
                        time_year = None
                    
                    fig_universe = plot_market_universe(
                        df_filtered_forecast, forecast_data,
                        ticker=ticker_viz,
                        time_slider=time_year
                    )
                    if fig_universe is not None:
                        st.plotly_chart(fig_universe, width='stretch', key=f'universe_{ticker_viz}')
                    else:
                        safe_info("Market universe memerlukan multiple tickers untuk perbandingan.")
                except Exception as e:
                    safe_error(f"Error membuat market universe: {str(e)}")
                
                # 5. What-If Terrain (3D Surface Simulation)
                st.subheader("5. What-If Terrain: Economic Simulation")
                st.markdown("""
                **Konsep**: Permukaan topografi 3D yang berubah berdasarkan variabel ekonomi.
                X = Waktu, Y = Interest Rate, Z = Harga Saham.
                Geser slider untuk mengubah suku bunga dan lihat permukaan berubah secara real-time!
                """)
                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        interest_rate = st.slider(
                            "Interest Rate (%)",
                            min_value=0.5,
                            max_value=10.0,
                            value=3.0,
                            step=0.1,
                            help="Geser untuk melihat impact suku bunga pada harga saham",
                            key=f'ir_slider_{ticker_viz}'
                        ) / 100
                    with col2:
                        inflation = st.slider(
                            "Inflation Rate (%)",
                            min_value=0.0,
                            max_value=10.0,
                            value=2.0,
                            step=0.1,
                            help="Geser untuk melihat impact inflasi",
                            key=f'inflation_slider_{ticker_viz}'
                        ) / 100
                    
                    fig_terrain = plot_what_if_terrain(
                        df_filtered_forecast, forecast_data,
                        ticker=ticker_viz,
                        interest_rate=interest_rate,
                        inflation=inflation
                    )
                    if fig_terrain is not None:
                        st.plotly_chart(fig_terrain, width='stretch', key=f'terrain_{ticker_viz}')
                    else:
                        safe_info("What-if terrain tidak tersedia untuk data ini.")
                except Exception as e:
                    safe_error(f"Error membuat what-if terrain: {str(e)}")
                
                # 6. Risk-Reward Motion Quadrant (KEPT - as requested)
                st.subheader("6. Risk-Reward Motion Quadrant: Stock Journey")
                st.markdown("""
                **Konsep**: Animated bubble chart dengan tombol Play.
                Sumbu X = Volatilitas (Risiko), Sumbu Y = ROE (Profitabilitas).
                Bubble bergerak dari 2005 ke 2020, menunjukkan evolusi performa saham.
                Lihat perjalanan dari "High Risk/Low Return" menuju "Low Risk/High Return"!
                """)
                try:
                    fig_motion = plot_risk_reward_motion(df_filtered_forecast, ticker=ticker_viz)
                    if fig_motion is not None:
                        st.plotly_chart(fig_motion, width='stretch', key=f'motion_{ticker_viz}')
                    else:
                        safe_info("Risk-reward motion quadrant tidak tersedia. Pastikan data memiliki kolom Volatility dan ROE.")
                except Exception as e:
                    safe_error(f"Error membuat risk-reward motion quadrant: {str(e)}")
            else:
                safe_info("👆 Silakan generate forecast terlebih dahulu untuk melihat visualisasi!")

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
