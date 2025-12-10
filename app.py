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
    plot_3d_sensitivity_vortex
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

# Ultra-Compact Design System CSS
st.markdown("""
<style>
    /* ============================================
       ULTRA-COMPACT DESIGN SYSTEM
       ============================================ */
    
    /* Root Variables */
    :root {
        --gray-50: #F9FAFB;
        --gray-100: #F3F4F6;
        --gray-200: #E5E7EB;
        --gray-300: #D1D5DB;
        --gray-400: #9CA3AF;
        --gray-500: #6B7280;
        --gray-600: #4B5563;
        --gray-700: #374151;
        --gray-900: #111827;
        --blue-50: #EFF6FF;
        --blue-100: #DBEAFE;
        --blue-500: #3B82F6;
        --blue-600: #2563EB;
        --green-600: #059669;
        --red-600: #DC2626;
        --orange-700: #C2410C;
    }
    
    /* Global Typography */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
    }
    
    /* Compact Typography System */
    h1 {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #111827 !important;
        line-height: 1.5 !important;
        margin: 0 0 8px 0 !important;
    }
    
    h2 {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #111827 !important;
        line-height: 1.5 !important;
        margin: 0 0 6px 0 !important;
    }
    
    h3 {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #111827 !important;
        line-height: 1.5 !important;
        margin: 0 0 4px 0 !important;
    }
    
    p, div, span, label {
        font-size: 12px !important;
        color: #374151 !important;
        line-height: 1.5 !important;
    }
    
    /* Sidebar Design - Lebar untuk Icon + Text */
    section[data-testid="stSidebar"] {
        width: 240px !important;
        min-width: 240px !important;
        background: white !important;
        border-right: 1px solid #E5E7EB !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 16px 12px !important;
    }
    
    /* Sidebar Navigation - Icon + Text */
    .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 8px 12px !important;
    }
    
    /* Compact Button System */
    .stButton > button {
        height: 28px !important;
        padding: 0 12px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        border: 1px solid #E5E7EB !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button[kind="primary"] {
        background: #3B82F6 !important;
        color: white !important;
        border-color: #3B82F6 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #2563EB !important;
        border-color: #2563EB !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: white !important;
        color: #374151 !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #F9FAFB !important;
        border-color: #D1D5DB !important;
    }
    
    /* Compact Input Fields */
    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stTextInput > div > div {
        font-size: 12px !important;
    }
    
    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label,
    .stCheckbox label {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #374151 !important;
        margin-bottom: 4px !important;
    }
    
    /* Compact Metrics */
    [data-testid="stMetricValue"] {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 10px !important;
        color: #6B7280 !important;
    }
    
    /* Compact Cards */
    .metric-card {
        background: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        margin-bottom: 8px;
    }
    
    /* Compact Info Boxes */
    .stAlert {
        padding: 8px 12px !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        margin-bottom: 8px !important;
    }
    
    /* Compact Dataframe */
    .stDataFrame {
        font-size: 11px !important;
    }
    
    /* Compact Expander */
    .streamlit-expanderHeader {
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 6px 0 !important;
    }
    
    /* Remove excessive padding */
    .main .block-container {
        padding: 16px 24px !important;
    }
    
    /* Compact spacing */
    .element-container {
        margin-bottom: 8px !important;
    }
    
    /* File uploader compact */
    .uploadedFile {
        padding: 6px 8px !important;
        border-radius: 6px !important;
        font-size: 11px !important;
        margin: 4px 0 !important;
    }
    
    /* Hide sidebar labels untuk icon-only */
    .sidebar-compact-label {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Compact Title
st.markdown('<h1 style="font-size: 18px; font-weight: 600; color: #111827; margin-bottom: 4px;">📈 FinScope</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 12px; color: #6B7280; margin-bottom: 16px;">Financial Forecasting Dashboard</p>', unsafe_allow_html=True)


# Initialize session state
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}

# Sidebar: File Upload (Compact)
with st.sidebar:
    # Compact file upload section
    st.markdown('<div style="font-size: 10px; color: #9CA3AF; margin-bottom: 4px;">📁 Upload</div>', unsafe_allow_html=True)
    
    # File size limit warning (compact)
    st.caption('<span style="font-size: 10px;">Max 200MB</span>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "CSV file",
        type=['csv'],
        help="File harus memiliki kolom: Date, Ticker, Open, High, Low, Close, Volume, ROE, Debt_Equity, EBIT_Margin",
        label_visibility="collapsed"
    )
    
    # Check file size before processing
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
    
    st.markdown("---")
    
    # Sidebar: Ultra-Compact Icon-Only Navigation
    # Initialize page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 ETL Results"
    
    # Navigation options dengan icon dan tooltip
    nav_items = [
        ("📊", "ETL Results", "📊 ETL Results"),
        ("🔍", "Exploratory Data Analysis", "🔍 Exploratory Data Analysis"),
        ("🤖", "Forecasting", "🤖 Forecasting"),
        ("📈", "Visualizations", "📈 Visualizations")
    ]
    
    # Check if data is available
    has_data = (st.session_state.df_processed is not None and 
               hasattr(st.session_state.df_processed, 'empty') and 
               not st.session_state.df_processed.empty)
    
    # Create navigation buttons dengan icon + text
    for icon, tooltip, nav_key in nav_items:
        # Determine if active
        is_active = st.session_state.current_page == nav_key
        
        # Check if data is required
        data_required = nav_key != "📊 ETL Results"
        disabled = data_required and not has_data
        
        # Create button dengan icon + text
        button_type = "primary" if is_active else "secondary"
        button_label = f"{icon} {tooltip}"
        
        clicked = st.button(
            button_label,
            key=f"nav_{nav_key}",
            disabled=disabled,
            use_container_width=True,
            type=button_type
        )
        
        if clicked:
            st.session_state.current_page = nav_key
            st.rerun()
    
    # Set current page - harus setelah semua button dibuat
    page = st.session_state.current_page
    
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
                        try:
                            chunk_reader = pd.read_csv(
                                uploaded_file, 
                                chunksize=chunk_size, 
                                parse_dates=['Date'], 
                                low_memory=True  # Use low_memory=True to reduce memory
                            )
                        except (KeyError, ValueError):
                            # If Date column doesn't exist or can't be parsed
                            uploaded_file.seek(0)
                            chunk_reader = pd.read_csv(
                                uploaded_file, 
                                chunksize=chunk_size, 
                                low_memory=True
                            )
                        
                        for i, chunk in enumerate(chunk_reader):
                            # Convert any extension arrays to standard pandas types to avoid pyarrow issues
                            chunk = chunk.copy()
                            for col in chunk.columns:
                                # Check if column uses extension array (pyarrow, nullable, etc.)
                                dtype_str = str(chunk[col].dtype).lower()
                                if ('arrow' in dtype_str or 
                                    'extension' in dtype_str or 
                                    'nullable' in dtype_str or
                                    hasattr(chunk[col].dtype, 'name') and 'Int' in str(chunk[col].dtype)):
                                    # Convert to standard type
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
                                # Take only what we need
                                remaining_rows = target_rows - total_rows
                                if remaining_rows > 0:
                                    chunk = chunk.iloc[:remaining_rows].copy()
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
                            
                            # Aggressive memory cleanup every 5 chunks dengan jeda untuk WebSocket
                            if len(chunk_list) % 5 == 0:
                                gc.collect()
                                time.sleep(0.02)  # Jeda kecil untuk mencegah WebSocket timeout
                        
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
                            # Combine in tiny batches of 5 chunks
                            batch_size = 5
                            combined_batches = []
                            
                            for i in range(0, len(chunk_list), batch_size):
                                batch = chunk_list[i:i+batch_size]
                                if len(batch) > 1:
                                    combined_batches.append(pd.concat(batch, ignore_index=True))
                                else:
                                    combined_batches.append(batch[0])
                                
                                # Aggressive cleanup
                                del batch
                                gc.collect()
                            
                            # Final combine
                            if len(combined_batches) > 1:
                                df_combined = pd.concat(combined_batches, ignore_index=True)
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
                        
                        # Process ETL dengan visualisasi yang lebih baik
                        st.session_state.df_processed, st.session_state.metadata = process_etl_from_df(
                            df_combined,
                            clean_method='ffill',
                            calculate_features=True,
                            scale_close=True,
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
                        safe_error("❌ Memory Error: File terlalu besar. Silakan aktifkan downsampling atau gunakan file yang lebih kecil.")
                        safe_info("💡 Tips: Aktifkan checkbox 'Downsample untuk file besar' di sidebar untuk mengurangi ukuran data.")
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
                        st.session_state.df_processed, st.session_state.metadata = process_etl(
                            tmp_path,
                            clean_method='ffill',
                            calculate_features=True,
                            scale_close=True
                        )
                        st.session_state.metadata['file_name'] = uploaded_file.name
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
            except MemoryError:
                safe_error("❌ Memory Error: File terlalu besar untuk diproses. Silakan gunakan file yang lebih kecil atau aktifkan downsampling.")
                st.session_state.df_processed = None
                st.session_state.metadata = {}
            except Exception as e:
                safe_error(f"❌ Error memproses file: {str(e)}")
                import traceback
                safe_error(f"Detail: {traceback.format_exc()}")
                st.session_state.df_processed = None
                st.session_state.metadata = {}
            
            # Check null percentage (only if processing succeeded)
            if (st.session_state.df_processed is not None and 
                hasattr(st.session_state.df_processed, 'empty') and 
                not st.session_state.df_processed.empty):
                try:
                    null_info = check_null_percentage(st.session_state.df_processed)
                    if null_info and null_info.get('has_warning', False):
                        safe_warning(null_info['warning_message'])
                except (AttributeError, TypeError, Exception) as e:
                    # Skip if error checking null percentage
                    pass

# Check if data exists
has_data = (st.session_state.df_processed is not None and 
           hasattr(st.session_state.df_processed, 'empty') and 
           not st.session_state.df_processed.empty)

# Main content based on selected page
# Allow ETL Results page even without data, but require data for other pages
if page != "📊 ETL Results" and not has_data:
    safe_warning("⚠️ Upload dan proses file terlebih dahulu untuk mengakses halaman ini.")
    safe_info("👈 Silakan upload file CSV di sidebar untuk memulai analisis.")
    st.stop()

if page == "📊 ETL Results":
    st.header("ETL Results")
    
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
        
        # Metadata display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df_filtered_etl):,}")
        with col2:
            st.metric("Total Columns", len(df_filtered_etl.columns))
        with col3:
            if 'Ticker' in df_filtered_etl.columns:
                st.metric("Number of Tickers", len(df_filtered_etl['Ticker'].unique()))
            else:
                st.metric("Number of Tickers", "N/A")
        with col4:
            if 'Date' in df_filtered_etl.columns:
                date_min = df_filtered_etl['Date'].min()
                date_max = df_filtered_etl['Date'].max()
                st.metric("Date Range", f"{date_min} to {date_max}")
            else:
                st.metric("Date Range", "N/A")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df_filtered_etl.head(20), width='stretch')
    
elif page == "🔍 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    # Filters untuk EDA page
    if (st.session_state.df_processed is not None and 
        hasattr(st.session_state.df_processed, 'columns') and 
        not st.session_state.df_processed.empty):
        with st.expander("🔍 Filters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if 'Ticker' in st.session_state.df_processed.columns:
                    tickers = sorted(st.session_state.df_processed['Ticker'].unique().tolist())
                    selected_ticker_eda = st.selectbox(
                        "Pilih Ticker",
                        ['All'] + tickers,
                        help="Filter data berdasarkan ticker symbol",
                        key="eda_ticker"
                    )
                    if selected_ticker_eda == 'All':
                        selected_ticker_eda = None
                else:
                    selected_ticker_eda = None
            
            with col2:
                if 'Date' in st.session_state.df_processed.columns:
                    try:
                        min_date = st.session_state.df_processed['Date'].min().date()
                        max_date = st.session_state.df_processed['Date'].max().date()
                        date_range_eda = st.date_input(
                            "Pilih Range Tanggal",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            help="Filter data berdasarkan range tanggal",
                            key="eda_date_range"
                        )
                    except Exception:
                        date_range_eda = None
                else:
                    date_range_eda = None
        
        # Apply filters (no need to copy - filtering creates new DataFrame)
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
            try:
                if df_ts is not None and 'Close' in df_ts.columns:
                    # Resample untuk dataset besar (>5 tahun)
                    if len(df_ts) > 1300:  # ~5 tahun daily data
                        resample_freq = 'W'
                        chart_cols = ['Close']
                        if 'Open' in df_ts.columns:
                            chart_cols.append('Open')
                        df_chart = df_ts[chart_cols].resample(resample_freq).last()
                        if 'Volume' in df_ts.columns:
                            df_chart['Volume'] = df_ts['Volume'].resample(resample_freq).sum()
                        if 'Volatility_30d' in df_ts.columns:
                            df_chart['Volatility_30d'] = df_ts['Volatility_30d'].resample(resample_freq).mean()
                        safe_info("📊 Data di-resample ke weekly untuk performa yang lebih baik.")
                    else:
                        chart_cols = ['Close']
                        if 'Open' in df_ts.columns:
                            chart_cols.append('Open')
                        df_chart = df_ts[chart_cols].copy()
                        if 'Volume' in df_ts.columns:
                            df_chart['Volume'] = df_ts['Volume']
                        if 'Volatility_30d' in df_ts.columns:
                            df_chart['Volatility_30d'] = df_ts['Volatility_30d']
                    
                    # Create figure with secondary y-axis
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Price traces (primary y-axis)
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], 
                                            name='Close Price', line=dict(color='#1f77b4', width=2)),
                                 secondary_y=False)
                    if 'Open' in df_chart.columns:
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Open'], 
                                                name='Open Price', line=dict(color='#2ca02c', width=1, dash='dot')),
                                     secondary_y=False)
                    
                    # Volatility overlay (primary y-axis, as area)
                    if 'Volatility_30d' in df_chart.columns:
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Volatility_30d'], 
                                                name='Volatility (30d)', line=dict(color='red', width=1, dash='dash'),
                                                fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'),
                                     secondary_y=False)
                    
                    # Volume trace (secondary y-axis)
                    if 'Volume' in df_chart.columns:
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Volume'], 
                                                name='Volume', line=dict(color='rgba(255,165,0,0.5)', width=1),
                                                fill='tozeroy', fillcolor='rgba(255,165,0,0.1)'),
                                     secondary_y=True)
                    
                    # Update axes
                    fig.update_xaxes(title_text="Date")
                    fig.update_yaxes(title_text="Price", secondary_y=False)
                    fig.update_yaxes(title_text="Volume", secondary_y=True)
                    
                    fig.update_layout(
                        title="Historical Price & Volume Trends dengan Volatility Overlay",
                        hovermode='x unified',
                        height=500
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    safe_warning("Data tidak memiliki kolom Date atau Close untuk line chart.")
            except Exception as e:
                safe_error(f"Error membuat line chart: {str(e)}")
            
            # 2. Histogram: Distribusi Harga & Ratios (dengan Volume vs Volatility)
            st.subheader("2. Distribution Analysis (Histogram)")
            try:
                # Price & Volume
                col1, col2 = st.columns(2)
                with col1:
                    if 'Close' in df_viz.columns:
                        try:
                            # Try with KDE first, fallback to simple histogram if it fails
                            close_data = df_viz['Close'].dropna()
                            if len(close_data) > 0:
                                try:
                                    fig_price = px.histogram(df_viz, x='Close', nbins=20, 
                                                            title="Distribution of Close Price",
                                                            labels={'Close': 'Close Price', 'count': 'Frequency'},
                                                            marginal="kde")
                                except Exception:
                                    # Fallback without KDE
                                    fig_price = px.histogram(df_viz, x='Close', nbins=20, 
                                                            title="Distribution of Close Price",
                                                            labels={'Close': 'Close Price', 'count': 'Frequency'})
                                fig_price.update_layout(height=400)
                                st.plotly_chart(fig_price, width='stretch')
                        except Exception as e:
                            safe_warning(f"Tidak dapat membuat histogram untuk Close Price: {str(e)}")
                with col2:
                    if 'Volume' in df_viz.columns:
                        try:
                            volume_data = df_viz['Volume'].dropna()
                            if len(volume_data) > 0:
                                try:
                                    fig_volume = px.histogram(df_viz, x='Volume', nbins=20,
                                                             title="Distribution of Volume",
                                                             labels={'Volume': 'Volume', 'count': 'Frequency'},
                                                             marginal="kde")
                                except Exception:
                                    fig_volume = px.histogram(df_viz, x='Volume', nbins=20,
                                                             title="Distribution of Volume",
                                                             labels={'Volume': 'Volume', 'count': 'Frequency'})
                                fig_volume.update_layout(height=400)
                                st.plotly_chart(fig_volume, width='stretch')
                        except Exception as e:
                            safe_warning(f"Tidak dapat membuat histogram untuk Volume: {str(e)}")
                
                # Volume vs Volatility comparison
                if 'Volume' in df_viz.columns and 'Volatility_30d' in df_viz.columns:
                    try:
                        st.markdown("**Volume vs Volatility Comparison**")
                        col3, col4 = st.columns(2)
                        with col3:
                            try:
                                fig_vol_comp = px.histogram(df_viz, x='Volume', nbins=20,
                                                           title="Volume Distribution",
                                                           labels={'Volume': 'Volume', 'count': 'Frequency'},
                                                           marginal="kde")
                            except Exception:
                                fig_vol_comp = px.histogram(df_viz, x='Volume', nbins=20,
                                                           title="Volume Distribution",
                                                           labels={'Volume': 'Volume', 'count': 'Frequency'})
                            fig_vol_comp.update_layout(height=350)
                            st.plotly_chart(fig_vol_comp, width='stretch')
                        with col4:
                            try:
                                fig_volatility = px.histogram(df_viz, x='Volatility_30d', nbins=20,
                                                             title="Volatility Distribution",
                                                             labels={'Volatility_30d': 'Volatility (30d)', 'count': 'Frequency'},
                                                             marginal="kde")
                            except Exception:
                                fig_volatility = px.histogram(df_viz, x='Volatility_30d', nbins=20,
                                                             title="Volatility Distribution",
                                                             labels={'Volatility_30d': 'Volatility (30d)', 'count': 'Frequency'})
                            fig_volatility.update_layout(height=350)
                            st.plotly_chart(fig_volatility, width='stretch')
                    except Exception as e:
                        safe_warning(f"Tidak dapat membuat perbandingan Volume vs Volatility: {str(e)}")
                
                # Financial Ratios
                ratio_cols = ['ROE', 'Debt_Equity', 'EBIT_Margin', 'Current_Ratio', 'Asset_Turnover', 'Gross_Margin']
                available_ratios = [col for col in ratio_cols if col in df_viz.columns]
                
                if available_ratios:
                    try:
                        st.markdown("**Financial Ratios Distribution**")
                        num_ratios = len(available_ratios)
                        cols_per_row = min(3, num_ratios)
                        for i in range(0, num_ratios, cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, ratio in enumerate(available_ratios[i:i+cols_per_row]):
                                with cols[j]:
                                    try:
                                        ratio_data = df_viz[ratio].dropna()
                                        if len(ratio_data) > 0:
                                            try:
                                                fig_ratio = px.histogram(df_viz, x=ratio, nbins=20,
                                                                       title=f"Distribution of {ratio}",
                                                                       labels={ratio: ratio, 'count': 'Frequency'},
                                                                       marginal="kde")
                                            except Exception:
                                                fig_ratio = px.histogram(df_viz, x=ratio, nbins=20,
                                                                       title=f"Distribution of {ratio}",
                                                                       labels={ratio: ratio, 'count': 'Frequency'})
                                            fig_ratio.update_layout(height=350)
                                            st.plotly_chart(fig_ratio, width='stretch')
                                    except Exception as e:
                                        safe_warning(f"Tidak dapat membuat histogram untuk {ratio}: {str(e)}")
                    except Exception as e:
                        safe_warning(f"Error membuat histogram ratios: {str(e)}")
            except Exception as e:
                safe_error(f"Error membuat histogram: {str(e)}")
            
            # 3. Scatter Plot: Harga vs. Key Ratios (dengan highlight outliers Volume)
            st.subheader("3. Price vs. Financial Ratios (Scatter Plot)")
            try:
                if 'Close' in df_viz.columns:
                    # Check for ratio columns (try both naming conventions and all possible column names)
                    ratio_cols = ['ROE', 'Debt_Equity_Ratio', 'Debt_Equity', 'EBIT_Margin', 
                                 'Current_Ratio', 'Asset_Turnover', 'Gross_Margin',
                                 'ROE_Ratio', 'EBIT_Margin_Ratio']
                    
                    # Also check all columns that might be ratios (numeric columns that are not price/volume)
                    numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
                    exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Volatility_30d', 
                                   'Daily_Return', 'Close_Scaled']
                    potential_ratios = [col for col in numeric_cols if col not in exclude_cols 
                                        and not col.startswith('SMA_') and not col.startswith('EMA_')]
                    
                    # Combine explicit ratio columns with potential ratios
                    all_ratio_candidates = list(set(ratio_cols + potential_ratios))
                    available_ratios = [col for col in all_ratio_candidates if col in df_viz.columns 
                                      and df_viz[col].notna().sum() > 0]
                    
                    if available_ratios:
                        selected_ratio = st.selectbox("Pilih Ratio untuk Scatter Plot", available_ratios)
                        
                        # Prepare data for scatter with Volume outliers highlight
                        df_scatter = df_viz.copy()
                        
                        # Identify Volume outliers (if Volume column exists)
                        if 'Volume' in df_scatter.columns:
                            Q1 = df_scatter['Volume'].quantile(0.25)
                            Q3 = df_scatter['Volume'].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df_scatter['Volume_Outlier'] = (df_scatter['Volume'] < lower_bound) | (df_scatter['Volume'] > upper_bound)
                        else:
                            df_scatter['Volume_Outlier'] = False
                        
                        # Create scatter plot with color by Volume outliers
                        if 'Volume_Outlier' in df_scatter.columns and df_scatter['Volume_Outlier'].any():
                            fig_scatter = px.scatter(df_scatter, x=selected_ratio, y='Close',
                                                    color='Volume_Outlier',
                                                    title=f"Close Price vs {selected_ratio} (Volume Outliers Highlighted)",
                                                    labels={selected_ratio: selected_ratio, 'Close': 'Close Price',
                                                           'Volume_Outlier': 'Volume Outlier'},
                                                    hover_data=['Date', 'Volume'] if 'Date' in df_scatter.columns else ['Volume'],
                                                    color_discrete_map={True: 'red', False: 'blue'})
                        else:
                            fig_scatter = px.scatter(df_scatter, x=selected_ratio, y='Close',
                                                    title=f"Close Price vs {selected_ratio}",
                                                    labels={selected_ratio: selected_ratio, 'Close': 'Close Price'},
                                                    hover_data=['Date'] if 'Date' in df_scatter.columns else None)
                        
                        # Add trend line (lazy import scipy.stats)
                        try:
                            from scipy import stats
                            x_vals = df_scatter[selected_ratio].dropna()
                            y_vals = df_scatter.loc[x_vals.index, 'Close']
                            valid_mask = ~(x_vals.isna() | y_vals.isna())
                            if valid_mask.sum() > 1:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(
                                    x_vals[valid_mask], y_vals[valid_mask]
                                )
                                x_trend = np.linspace(x_vals[valid_mask].min(), x_vals[valid_mask].max(), 100)
                                y_trend = slope * x_trend + intercept
                                fig_scatter.add_trace(go.Scatter(x=x_trend, y=y_trend, 
                                                                mode='lines', name='Trend Line',
                                                                line=dict(color='red', dash='dash', width=2)))
                        except ImportError:
                            # scipy not available, skip trend line
                            pass
                        except Exception:
                            pass
                        
                        fig_scatter.update_layout(height=500)
                        st.plotly_chart(fig_scatter, width='stretch')
                        
                        # Hitung korelasi
                        corr = df_viz[selected_ratio].corr(df_viz['Close'])
                        st.info(f"⚠️ **Korelasi**: {corr:.3f}. Ingat: Korelasi bukan berarti kausalitas!")
                    else:
                        # Show available numeric columns for debugging
                        numeric_cols_available = [col for col in df_viz.select_dtypes(include=[np.number]).columns.tolist() 
                                                  if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
                        if numeric_cols_available:
                            safe_warning(f"Tidak ada kolom financial ratios yang ditemukan. Kolom numerik yang tersedia: {', '.join(numeric_cols_available[:10])}")
                        else:
                            safe_warning("Tidak ada kolom financial ratios atau kolom numerik untuk scatter plot.")
                else:
                    safe_warning("Data tidak memiliki kolom Close untuk scatter plot.")
            except Exception as e:
                safe_error(f"Error membuat scatter plot: {str(e)}")
            
            # 4. Heatmap: Correlation Matrix Ratios (fokus ke ratios utama)
            st.subheader("4. Correlation Matrix (Heatmap)")
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
                        
                        # Create heatmap
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdYlGn',
                            zmid=0,
                            text=corr_matrix.round(2).values,
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            colorbar=dict(title="Correlation")
                        ))
                        
                        fig_corr.update_layout(
                            title="Correlation Matrix - Key Financial Ratios",
                            height=600,
                            xaxis_title="",
                            yaxis_title=""
                        )
                        st.plotly_chart(fig_corr, width='stretch')
                        
                        # Show high correlations
                        high_corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                val = corr_matrix.iloc[i, j]
                                if abs(val) > 0.5:
                                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
                        
                        if high_corr_pairs:
                            st.info(f"💡 **High Correlations (|r| > 0.5)**: {len(high_corr_pairs)} pairs found. "
                                   f"Focus on correlations > 0.5 for meaningful relationships.")
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
            
            # 5. Box Plot: Outliers per Quarter/Year
            st.subheader("5. Outlier Analysis (Box Plot)")
            try:
                if 'Close' in df_viz.columns:
                    df_box = df_viz.copy()
                    
                    # Create Quarter and Year columns if Date exists
                    if 'Date' in df_box.columns:
                        df_box['Date'] = pd.to_datetime(df_box['Date'])
                        df_box['Year'] = df_box['Date'].dt.year
                        df_box['Quarter'] = df_box['Date'].dt.quarter
                        df_box['Year_Quarter'] = df_box['Year'].astype(str) + '-Q' + df_box['Quarter'].astype(str)
                    
                    # Group selection
                    group_options = []
                    if 'Year_Quarter' in df_box.columns:
                        group_options.append(('Year_Quarter', 'Year-Quarter'))
                    if 'Quarter' in df_box.columns:
                        group_options.append(('Quarter', 'Quarter'))
                    if 'Year' in df_box.columns:
                        group_options.append(('Year', 'Year'))
                    if 'Ticker' in df_box.columns and df_box['Ticker'].nunique() > 1:
                        group_options.append(('Ticker', 'Ticker'))
                    
                    if group_options:
                        selected_group = st.selectbox("Group by:", [opt[0] for opt in group_options],
                                                      format_func=lambda x: dict(group_options)[x])
                        group_col = selected_group
                        title = f"Close Price Distribution by {dict(group_options)[selected_group]}"
                        
                        fig_box = px.box(df_box, x=group_col, y='Close', 
                                        title=title,
                                        labels={group_col: group_col, 'Close': 'Close Price'})
                        fig_box.update_layout(height=500, xaxis_tickangle=-45)
                        st.plotly_chart(fig_box, width='stretch')
                        
                        # Also show Volume box plot if available
                        if 'Volume' in df_box.columns:
                            fig_box_vol = px.box(df_box, x=group_col, y='Volume',
                                                title=f"Volume Distribution by {dict(group_options)[selected_group]}",
                                                labels={group_col: group_col, 'Volume': 'Volume'})
                            fig_box_vol.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_box_vol, width='stretch')
                    else:
                        # Single box plot
                        fig_box = px.box(df_box, y='Close', title="Close Price Distribution")
                        fig_box.update_layout(height=400)
                        st.plotly_chart(fig_box, width='stretch')
                else:
                    safe_warning("Data tidak memiliki kolom Close untuk box plot.")
            except Exception as e:
                safe_error(f"Error membuat box plot: {str(e)}")
            

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
        
        # 3D Visualizations Section
        if st.session_state.forecasts:
            st.divider()
            st.header("🎮 3D Immersive Visualizations (2025 Experience)")
            
            # Toggle untuk 2D/3D view
            view_mode = st.radio(
                "Pilih Mode Visualisasi:",
                ["3D Immersive", "2D Classic"],
                horizontal=True,
                help="3D memberikan depth dan insight lebih, 2D lebih mudah dipahami"
            )
            
            ticker_viz_3d = selected_ticker_forecast if selected_ticker_forecast else \
                (df_filtered_forecast['Ticker'].iloc[0] if 'Ticker' in df_filtered_forecast.columns and len(df_filtered_forecast) > 0 else None)
            
            if ticker_viz_3d and ticker_viz_3d in st.session_state.forecasts:
                forecast_3d = st.session_state.forecasts[ticker_viz_3d]
                model_3d = load_model(ticker_viz_3d) if ticker_viz_3d else None
                
                if view_mode == "3D Immersive":
                    # 1. 3D Forecast Terrain: Market Trend Projection
                    st.subheader("1. 3D Forecast Terrain: Market Trend Projection")
                    st.markdown("""
                    **Insight**: Seperti 3D heatmap di Grafana: "Terrain = market landscape, bukit = profit opportunity"—rotate + zoom seperti code profiler 3D.
                    """)
                    try:
                        fig_3d_terrain = plot_3d_forecast_terrain(
                            df_filtered_forecast,
                            forecast=forecast_3d,
                            ticker=ticker_viz_3d
                        )
                        if fig_3d_terrain:
                            st.plotly_chart(fig_3d_terrain, width='stretch', key=f'3d_terrain_{ticker_viz_3d}', config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['resetCameraDefault3d']
                            })
                            st.info("💡 **Tips**: Low CI (0.5 USD) tunjuk low-risk stock di stable market; Z melebar di Q4 (seasonal earnings volatility).")
                        else:
                            safe_warning("Tidak dapat membuat 3D terrain. Pastikan forecast memiliki CI bands.")
                    except Exception as e:
                        safe_error(f"Error membuat 3D terrain: {str(e)}")
                    
                    # 2. Interactive CI Ribbon: Risk-Adjusted Stock Paths
                    st.subheader("2. Interactive CI Ribbon: Risk-Adjusted Stock Paths")
                    st.markdown("""
                    **Insight**: Mirip 3D path di Unity: "Ribbon = forecast highway, twist = ROE bump seperti route optimization"—drag untuk lihat alternate paths.
                    """)
                    try:
                        fig_3d_ribbon = plot_3d_ci_ribbon(
                            df_filtered_forecast,
                            forecast=forecast_3d,
                            ticker=ticker_viz_3d
                        )
                        if fig_3d_ribbon:
                            st.plotly_chart(fig_3d_ribbon, width='stretch', key=f'3d_ribbon_{ticker_viz_3d}', config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['resetCameraDefault3d']
                            })
                            st.info("💡 **Insight**: Ribbon sempit di low Debt (0.72) = conservative finance; melebar di high seasonality (Q4 market rally).")
                        else:
                            safe_warning("Tidak dapat membuat 3D ribbon. Pastikan forecast memiliki CI bands.")
                    except Exception as e:
                        safe_error(f"Error membuat 3D ribbon: {str(e)}")
                    
                    # 3. 3D Component Orbit: Decomposition Galaxy
                    st.subheader("3. 3D Component Orbit: Decomposition Galaxy")
                    st.markdown("""
                    **Insight**: Seperti 3D solar system sim: "Orbit = forecast forces, satelit ROE = 'planet' pengaruh"—spin untuk lihat interactions seperti dependency graph 3D.
                    """)
                    try:
                        fig_3d_orbit = plot_3d_component_orbit(
                            df_filtered_forecast,
                            forecast=forecast_3d,
                            model=model_3d,
                            ticker=ticker_viz_3d
                        )
                        if fig_3d_orbit:
                            st.plotly_chart(fig_3d_orbit, width='stretch', key=f'3d_orbit_{ticker_viz_3d}', config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['resetCameraDefault3d']
                            })
                            st.info("💡 **Insight**: Trend (0.2 USD/hari) dominan 70% forecast, ROE satelit 'orbit' dekat di profitability-driven market.")
                        else:
                            safe_warning("Tidak dapat membuat 3D orbit. Pastikan forecast memiliki components.")
                    except Exception as e:
                        safe_error(f"Error membuat 3D orbit: {str(e)}")
                    
                    # 4. Dynamic Waterfall 3D: DuPont ROE Forecast Cascade
                    st.subheader("4. Dynamic Waterfall 3D: DuPont ROE Forecast Cascade")
                    st.markdown("""
                    **Insight**: Seperti 3D waterfall di Blender: "Cascade = ROE flow, drop = leverage drag seperti memory leak"—interact slice untuk dissect layers.
                    """)
                    try:
                        fig_3d_waterfall = plot_3d_waterfall_dupont(
                            df_filtered_forecast,
                            forecast=forecast_3d,
                            ticker=ticker_viz_3d
                        )
                        if fig_3d_waterfall:
                            st.plotly_chart(fig_3d_waterfall, width='stretch', key=f'3d_waterfall_{ticker_viz_3d}', config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['resetCameraDefault3d']
                            })
                            st.info("💡 **Insight**: Leverage (dari Debt 0.72) kontribusi -5% di forecast, tapi turnover 2.79 boost +10% di ekspansi market.")
                        else:
                            safe_warning("Tidak dapat membuat 3D waterfall. Pastikan data memiliki kolom ROE atau EBIT_Margin.")
                    except Exception as e:
                        safe_error(f"Error membuat 3D waterfall: {str(e)}")
                    
                    # 5. Bonus: 3D Sensitivity Vortex: Market Regressor Impact
                    st.subheader("5. Bonus: 3D Sensitivity Vortex: Market Regressor Impact")
                    st.markdown("""
                    **Insight**: Mirip 3D vortex di fluid sim: "Arms = market forces, spin = sensitivity swirl"—zoom arms seperti debug fluid dynamics.
                    """)
                    try:
                        fig_3d_vortex = plot_3d_sensitivity_vortex(
                            df_filtered_forecast,
                            forecast=forecast_3d,
                            ticker=ticker_viz_3d
                        )
                        if fig_3d_vortex:
                            st.plotly_chart(fig_3d_vortex, width='stretch', key=f'3d_vortex_{ticker_viz_3d}', config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['resetCameraDefault3d']
                            })
                            st.info("💡 **Insight**: ROE vortex 'spin' +3% di stock bull, Debt arm 'pull back' di high-inflation forecast.")
                        else:
                            safe_warning("Tidak dapat membuat 3D vortex. Pastikan data memiliki regressors (ROE, Debt, EBIT).")
                    except Exception as e:
                        safe_error(f"Error membuat 3D vortex: {str(e)}")
                    
                    # 6. Additional: 3D Profit Prism (if data available)
                    st.subheader("6. 3D Profit Prism: ROE Multiplier Forecast")
                    try:
                        fig_3d_prism = plot_3d_profit_prism(
                            df_filtered_forecast,
                            forecast=forecast_3d,
                            ticker=ticker_viz_3d
                        )
                        if fig_3d_prism:
                            st.plotly_chart(fig_3d_prism, width='stretch', key=f'3d_prism_{ticker_viz_3d}', config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['resetCameraDefault3d']
                            })
                            st.info("💡 **Insight**: ROE x turnover = multiplier, forecast prism lebar di growth economy!")
                        else:
                            safe_warning("Tidak dapat membuat 3D prism. Menggunakan data estimasi jika kolom ROE/EBIT_Margin tidak tersedia.")
                    except Exception as e:
                        safe_error(f"Error membuat 3D prism: {str(e)}")
                    
                    # 7. Additional: 3D Scatter Decomposition (if model available)
                    st.subheader("7. 3D Scatter: Forecast Decomposition (Additional)")
                    try:
                        if model_3d:
                            fig_3d_scatter = plot_3d_scatter_decomposition(
                                model_3d,
                                forecast=forecast_3d,
                                ticker=ticker_viz_3d
                            )
                            if fig_3d_scatter:
                                st.plotly_chart(fig_3d_scatter, width='stretch', key=f'3d_scatter_{ticker_viz_3d}', config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToAdd': ['resetCameraDefault3d']
                                })
                                st.info("💡 **Insight**: ROE kontribusi 20%—seperti CPU boost di server!")
                            else:
                                safe_warning("Tidak dapat membuat 3D scatter decomposition.")
                        else:
                            safe_warning("Model tidak tersedia untuk decomposition.")
                    except Exception as e:
                        safe_error(f"Error membuat 3D scatter: {str(e)}")
                    
                    # 8. Additional: 3D Funnel Scenarios
                    st.subheader("8. 3D Funnel: Scenario Projections (Additional)")
                    try:
                        fig_3d_funnel = plot_3d_funnel_scenarios(
                            df_filtered_forecast,
                            forecast=forecast_3d,
                            ticker=ticker_viz_3d
                        )
                        if fig_3d_funnel:
                            st.plotly_chart(fig_3d_funnel, width='stretch', key=f'3d_funnel_{ticker_viz_3d}', config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['resetCameraDefault3d']
                            })
                            
                            # Display scenario summary
                            try:
                                last_forecast = forecast_3d.iloc[-1]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Base Scenario", f"${last_forecast['yhat']:.2f}")
                                with col2:
                                    if 'yhat_upper' in last_forecast:
                                        st.metric("Optimistic", f"${last_forecast['yhat_upper']:.2f}", 
                                                 delta=f"+${last_forecast['yhat_upper'] - last_forecast['yhat']:.2f}")
                                    else:
                                        st.metric("Optimistic", f"${last_forecast['yhat'] * 1.1:.2f}", 
                                                 delta=f"+${last_forecast['yhat'] * 0.1:.2f}")
                                with col3:
                                    if 'yhat_lower' in last_forecast:
                                        st.metric("Pessimistic", f"${last_forecast['yhat_lower']:.2f}",
                                                 delta=f"-${last_forecast['yhat'] - last_forecast['yhat_lower']:.2f}")
                                    else:
                                        st.metric("Pessimistic", f"${last_forecast['yhat'] * 0.9:.2f}",
                                                 delta=f"-${last_forecast['yhat'] * 0.1:.2f}")
                            except Exception:
                                pass
                        else:
                            safe_warning("Tidak dapat membuat 3D funnel scenarios.")
                    except Exception as e:
                        safe_error(f"Error membuat 3D funnel: {str(e)}")
                    
                    # 9. Additional: 3D Sensitivity Mesh (if model available)
                    st.subheader("9. 3D Sensitivity Mesh: Regressor Impact (Additional)")
                    try:
                        if model_3d:
                            fig_3d_mesh = plot_3d_sensitivity_mesh(
                                model_3d,
                                forecast=forecast_3d,
                                ticker=ticker_viz_3d,
                                periods=st.session_state.forecast_periods
                            )
                            if fig_3d_mesh:
                                st.plotly_chart(fig_3d_mesh, width='stretch', key=f'3d_mesh_{ticker_viz_3d}', config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToAdd': ['resetCameraDefault3d']
                                })
                                st.info("💡 **Tips**: Area merah = high impact, kuning = medium, hitam = low impact.")
                            else:
                                safe_warning("Tidak dapat membuat 3D sensitivity mesh. Pastikan model menggunakan regressors.")
                        else:
                            safe_warning("Model tidak tersedia untuk sensitivity analysis.")
                    except Exception as e:
                        safe_error(f"Error membuat 3D sensitivity mesh: {str(e)}")
                
                else:
                    # 2D Classic view (existing visualizations)
                    st.info("📊 Menggunakan visualisasi 2D classic. Switch ke '3D Immersive' untuk pengalaman yang lebih immersive!")
            
            else:
                safe_info("👆 Silakan generate forecast terlebih dahulu untuk melihat visualisasi 3D!")

elif page == "📈 Visualizations":
    st.header("Advanced Visualizations")
    
    # Filters untuk Visualizations page
    if (st.session_state.df_processed is not None and 
        hasattr(st.session_state.df_processed, 'columns') and 
        not st.session_state.df_processed.empty):
        with st.expander("🔍 Filters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if 'Ticker' in st.session_state.df_processed.columns:
                    tickers = sorted(st.session_state.df_processed['Ticker'].unique().tolist())
                    selected_ticker_viz = st.selectbox(
                        "Pilih Ticker",
                        ['All'] + tickers,
                        help="Filter data berdasarkan ticker symbol",
                        key="viz_ticker"
                    )
                    if selected_ticker_viz == 'All':
                        selected_ticker_viz = None
                else:
                    selected_ticker_viz = None
            
            with col2:
                if 'Date' in st.session_state.df_processed.columns:
                    try:
                        min_date = st.session_state.df_processed['Date'].min().date()
                        max_date = st.session_state.df_processed['Date'].max().date()
                        date_range_viz = st.date_input(
                            "Pilih Range Tanggal",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            help="Filter data berdasarkan range tanggal",
                            key="viz_date_range"
                        )
                    except Exception:
                        date_range_viz = None
                else:
                    date_range_viz = None
        
        # Apply filters (no need to copy - filtering creates new DataFrame)
        df_filtered_viz = st.session_state.df_processed
        if selected_ticker_viz and 'Ticker' in df_filtered_viz.columns:
            df_filtered_viz = df_filtered_viz[df_filtered_viz['Ticker'] == selected_ticker_viz]
        if date_range_viz and len(date_range_viz) == 2 and 'Date' in df_filtered_viz.columns:
            start_date, end_date = date_range_viz
            df_filtered_viz = df_filtered_viz[
                (df_filtered_viz['Date'].dt.date >= start_date) &
                (df_filtered_viz['Date'].dt.date <= end_date)
            ]
    else:
        df_filtered_viz = pd.DataFrame()
        selected_ticker_viz = None
    
    if len(df_filtered_viz) == 0:
        safe_warning("Tidak ada data untuk visualisasi. Silakan pilih filter yang berbeda.")
    else:
        # Get forecast if available
        forecast = None
        ticker_viz = selected_ticker_viz if selected_ticker_viz else \
            (df_filtered_viz['Ticker'].iloc[0] if 'Ticker' in df_filtered_viz.columns else None)
        
        if ticker_viz and ticker_viz in st.session_state.forecasts:
            forecast = st.session_state.forecasts[ticker_viz]
        
        # Visualization 1: Correlation Heatmap
        st.subheader("1. Correlation Heatmap")
        try:
            fig_heatmap = plot_correlation_heatmap(df_filtered_viz, ticker=selected_ticker_viz)
            if fig_heatmap is not None:
                st.plotly_chart(fig_heatmap, width='stretch', key=f'heatmap_viz_{selected_ticker_viz}')
            else:
                safe_info("Correlation heatmap tidak tersedia. Pastikan ada minimal 2 kolom numerik untuk korelasi.")
        except Exception as e:
            safe_error(f"Error membuat correlation heatmap: {str(e)}")
        
        # Visualization 2: ROE Waterfall
        st.subheader("2. ROE Breakdown (DuPont Formula)")
        try:
            fig_waterfall = plot_roe_waterfall(df_filtered_viz, forecast=forecast, ticker=selected_ticker_viz)
            if fig_waterfall is not None:
                st.plotly_chart(fig_waterfall, width='stretch', key=f'waterfall_viz_{selected_ticker_viz}')
            else:
                safe_info("ROE waterfall chart tidak tersedia. Pastikan data memiliki kolom ROE dan EBIT_Margin.")
        except Exception as e:
            safe_error(f"Error membuat ROE waterfall: {str(e)}")
        
        # Visualization 3: Radar Chart
        st.subheader("3. Multi-Ratio Evolution (Radar Chart)")
        try:
            fig_radar = plot_radar_multi_ratio(df_filtered_viz, ticker=selected_ticker_viz)
            if fig_radar is not None:
                st.plotly_chart(fig_radar, width='stretch', key=f'radar_viz_{selected_ticker_viz}')
            else:
                safe_info("Radar chart tidak tersedia. Pastikan data memiliki kolom ROE, Debt_Equity, atau EBIT_Margin.")
        except Exception as e:
            safe_error(f"Error membuat radar chart: {str(e)}")
    

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
