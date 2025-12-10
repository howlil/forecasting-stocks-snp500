"""
Visualization module untuk aplikasi FinScope.
Berisi 5+ visualisasi interaktif menggunakan Plotly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_neon_cyber_forecast(df: pd.DataFrame, forecast: pd.DataFrame = None,
                             ticker: str = None):
    """
    Neon Cyber-Forecast: Dark mode dengan efek neon bercahaya dan range slider.
    
    Background hitam/dark blue, garis forecast cyan terang dengan efek glow,
    area CI menggunakan gradient fill transparan (kabut bercahaya).
    Titik marker putih untuk data historis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, Close
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat, yhat_lower, yhat_upper
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    go.Figure : Plotly figure dengan range slider atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Separate historical and future
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
            forecast_hist = forecast[forecast['y'].notna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
            forecast_hist = forecast.head(len(forecast) - 90).copy()
        
        if forecast_future.empty or 'yhat' not in forecast_future.columns:
            return None
        
        fig = go.Figure()
        
        # Historical data with white markers
        if not forecast_hist.empty and 'y' in forecast_hist.columns:
            fig.add_trace(go.Scatter(
                x=forecast_hist['ds'],
                y=forecast_hist['y'],
                mode='markers+lines',
                name='Historical Price',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
                marker=dict(size=3, color='white', opacity=0.7),
                hovertemplate='<b>Historical</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        # Future forecast
        dates_future = forecast_future['ds']
        yhat = forecast_future['yhat'].values
        
        # Get confidence intervals
        if 'yhat_upper' in forecast_future.columns:
            yhat_upper = forecast_future['yhat_upper'].values
        else:
            yhat_upper = yhat * 1.1
        
        if 'yhat_lower' in forecast_future.columns:
            yhat_lower = forecast_future['yhat_lower'].values
        else:
            yhat_lower = yhat * 0.9
        
        # Confidence interval area (gradient fill - kabut bercahaya)
        fig.add_trace(go.Scatter(
            x=list(dates_future) + list(dates_future[::-1]),
            y=list(yhat_upper) + list(yhat_lower[::-1]),
            fill='toself',
            fillcolor='rgba(0, 191, 255, 0.15)',
            line=dict(color='rgba(0, 191, 255, 0.3)', width=0),
            name='Uncertainty (95% CI)',
            hoverinfo='skip'
        ))
        
        # Main forecast line (neon cyan)
        fig.add_trace(go.Scatter(
            x=dates_future,
            y=yhat,
            mode='lines',
            name='Forecast',
            line=dict(color='#00ffff', width=3),
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price (USD)")
        fig.update_layout(
            title=dict(
                text=f"Neon Cyber-Forecast ({ticker_name})",
                font=dict(size=22, color='#00ffff', family='Arial Black')
            ),
            xaxis=dict(
                title=dict(text="Date", font=dict(color='#00ffff', size=12)),
                gridcolor='rgba(0, 191, 255, 0.15)',
                linecolor='rgba(0, 191, 255, 0.5)'
            ),
            yaxis=dict(
                title=dict(text="Price (USD)", font=dict(color='#00ffff', size=12)),
                gridcolor='rgba(0, 191, 255, 0.15)',
                linecolor='rgba(0, 191, 255, 0.5)'
            ),
            height=650,
            hovermode='x unified',
            plot_bgcolor='#0a0a0a',  # Dark blue-black
            paper_bgcolor='#0a0a0a',
            font=dict(color='#00ffff', family='Courier New', size=11),
            legend=dict(
                bgcolor='rgba(10, 10, 10, 0.8)',
                bordercolor='rgba(0, 191, 255, 0.5)',
                borderwidth=1
            )
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=30, label="30d", step="day", stepmode="backward"),
                        dict(count=90, label="3m", step="day", stepmode="backward"),
                        dict(count=180, label="6m", step="day", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True, thickness=0.05),
                type="date"
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_neon_cyber_forecast: {str(e)}")
        return None


def plot_neon_time_tunnel(df: pd.DataFrame, forecast: pd.DataFrame = None,
                          ticker: str = None, show_uncertainty: bool = True,
                          highlight_anomalies: bool = True):
    """
    Neon Time-Tunnel: 3D perspective dengan uncertainty walls.
    
    Konsep: Jalan raya 3D dengan marka jalan neon, dinding transparan untuk uncertainty,
    dan batu merah untuk anomali.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol
    show_uncertainty : bool
        Tampilkan dinding uncertainty
    highlight_anomalies : bool
        Tampilkan anomali sebagai objek merah
    
    Returns:
    --------
    go.Figure : 3D Plotly figure
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
            forecast_hist = forecast[forecast['y'].notna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
            forecast_hist = forecast.head(len(forecast) - 90).copy()
        
        if forecast_future.empty or 'yhat' not in forecast_future.columns:
            return None
        
        # Prepare data
        dates_future = forecast_future['ds']
        yhat = forecast_future['yhat'].values
        
        if 'yhat_upper' in forecast_future.columns:
            yhat_upper = forecast_future['yhat_upper'].values
        else:
            yhat_upper = yhat * 1.1
        
        if 'yhat_lower' in forecast_future.columns:
            yhat_lower = forecast_future['yhat_lower'].values
        else:
            yhat_lower = yhat * 0.9
        
        # Convert dates to numeric for 3D
        dates_numeric = np.arange(len(dates_future))
        
        # Create 3D figure
        fig = go.Figure()
        
        # Main road (trend line) - neon cyan
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=yhat,
            z=np.zeros(len(dates_numeric)),
            mode='lines+markers',
            name='Forecast Road',
            line=dict(color='#00ffff', width=8),
            marker=dict(size=4, color='#00ffff'),
            hovertemplate='<b>Forecast</b><br>Date: %{text}<br>Price: $%{y:.2f}<extra></extra>',
            text=dates_future
        ))
        
        # Uncertainty walls (left and right)
        if show_uncertainty:
            # Left wall (lower bound)
            fig.add_trace(go.Scatter3d(
                x=dates_numeric,
                y=yhat_lower,
                z=np.zeros(len(dates_numeric)) - 0.5,
                mode='lines',
                name='Uncertainty Wall (Lower)',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=3),
                surfaceaxis=1,
                showlegend=False
            ))
            
            # Right wall (upper bound)
            fig.add_trace(go.Scatter3d(
                x=dates_numeric,
                y=yhat_upper,
                z=np.zeros(len(dates_numeric)) + 0.5,
                mode='lines',
                name='Uncertainty Wall (Upper)',
                line=dict(color='rgba(0, 255, 0, 0.3)', width=3),
                surfaceaxis=1,
                showlegend=False
            ))
            
            # Create uncertainty surface (transparent walls)
            uncertainty_width = (yhat_upper - yhat_lower) / 2
            for i in range(len(dates_numeric)):
                if i < len(dates_numeric) - 1:
                    # Create a small surface segment
                    x_vals = [dates_numeric[i], dates_numeric[i+1], dates_numeric[i+1], dates_numeric[i]]
                    y_vals = [yhat_lower[i], yhat_lower[i+1], yhat_upper[i+1], yhat_upper[i]]
                    z_vals = [-0.5, -0.5, 0.5, 0.5]
                    
                    fig.add_trace(go.Mesh3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        color='rgba(0, 191, 255, 0.1)',
                        opacity=0.3,
                        showscale=False,
                        hoverinfo='skip'
                    ))
        
        # Historical anomalies (red objects)
        if highlight_anomalies and not forecast_hist.empty and 'y' in forecast_hist.columns:
            hist_dates = forecast_hist['ds']
            hist_prices = forecast_hist['y'].values
            
            # Detect anomalies (prices outside 2 std from mean)
            mean_price = np.mean(hist_prices)
            std_price = np.std(hist_prices)
            anomalies = np.abs(hist_prices - mean_price) > 2 * std_price
            
            if np.any(anomalies):
                anomaly_dates = hist_dates[anomalies]
                anomaly_prices = hist_prices[anomalies]
                anomaly_numeric = np.arange(len(forecast_hist))[anomalies]
                
                fig.add_trace(go.Scatter3d(
                    x=anomaly_numeric,
                    y=anomaly_prices,
                    z=np.zeros(len(anomaly_prices)),
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        size=10,
                        color='#ff0000',
                        symbol='diamond',
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>Anomaly</b><br>Date: %{text}<br>Price: $%{y:.2f}<extra></extra>',
                    text=anomaly_dates
                ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"Neon Time-Tunnel: 3D Perspective ({ticker_name})",
                font=dict(size=18, family='Arial Black')
            ),
            scene=dict(
                xaxis_title="Time (Days Ahead)",
                yaxis_title="Price (USD)",
                zaxis_title="Uncertainty Width",
                bgcolor='#0a0a0a',
                xaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(0, 191, 255, 0.2)'),
                yaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(0, 191, 255, 0.2)'),
                zaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(0, 191, 255, 0.2)'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            height=700,
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_neon_time_tunnel: {str(e)}")
        return None


def plot_decomposition_glass_stack(model_data: dict = None, forecast: pd.DataFrame = None,
                                   ticker: str = None, explode_layers: bool = False):
    """
    Decomposition Glass Stack: 3D layers untuk komponen Prophet.
    
    Konsep: 3 layer kaca transparan (Trend, Seasonality, Regressors) yang bertumpuk.
    
    Parameters:
    -----------
    model_data : dict
        Prophet model data dengan komponen
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol
    explode_layers : bool
        Pisahkan layer untuk melihat lebih jelas
    
    Returns:
    --------
    go.Figure : 3D Plotly figure
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        if 'yhat' not in forecast.columns:
            return None
        
        dates = forecast['ds']
        dates_numeric = np.arange(len(dates))
        yhat = forecast['yhat'].values
        
        # Get components
        trend = forecast['trend'].values if 'trend' in forecast.columns else yhat * 0.7
        seasonal = forecast['seasonal'].values if 'seasonal' in forecast.columns else yhat * 0.2
        regressors = forecast.get('regressor', yhat * 0.1) if isinstance(forecast.get('regressor'), (list, np.ndarray, pd.Series)) else (yhat * 0.1)
        
        if isinstance(regressors, (list, np.ndarray, pd.Series)):
            if len(regressors) != len(dates):
                regressors = np.array([0.1] * len(dates)) * yhat
        else:
            regressors = np.array([0.1] * len(dates)) * yhat
        
        # Calculate z-offsets for layers
        if explode_layers:
            z_trend = np.zeros(len(dates)) - 1.0
            z_seasonal = np.zeros(len(dates))
            z_regressors = np.zeros(len(dates)) + 1.0
            z_final = np.zeros(len(dates)) + 2.0
        else:
            z_trend = np.zeros(len(dates)) - 0.3
            z_seasonal = np.zeros(len(dates))
            z_regressors = np.zeros(len(dates)) + 0.3
            z_final = np.zeros(len(dates)) + 0.6
        
        fig = go.Figure()
        
        # Layer 1: Base Trend (Blue glass)
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=trend,
            z=z_trend,
            mode='lines',
            name='Base Trend',
            line=dict(color='rgba(0, 100, 255, 0.8)', width=5),
            hovertemplate='<b>Trend</b><br>Date: %{text}<br>Value: $%{y:.2f}<extra></extra>',
            text=dates
        ))
        
        # Layer 2: Seasonality (Green glass)
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=trend + seasonal,
            z=z_seasonal,
            mode='lines',
            name='Seasonality',
            line=dict(color='rgba(0, 255, 100, 0.8)', width=5),
            hovertemplate='<b>Seasonal</b><br>Date: %{text}<br>Value: $%{y:.2f}<extra></extra>',
            text=dates
        ))
        
        # Layer 3: Regressors (Orange glass)
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=trend + seasonal + regressors,
            z=z_regressors,
            mode='lines',
            name='Fundamental Impact',
            line=dict(color='rgba(255, 150, 0, 0.8)', width=5),
            hovertemplate='<b>Regressors</b><br>Date: %{text}<br>Value: $%{y:.2f}<extra></extra>',
            text=dates
        ))
        
        # Final prediction (White line through all layers)
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=yhat,
            z=z_final,
            mode='lines+markers',
            name='Final Forecast',
            line=dict(color='white', width=6),
            marker=dict(size=4, color='white'),
            hovertemplate='<b>Forecast</b><br>Date: %{text}<br>Price: $%{y:.2f}<extra></extra>',
            text=dates
        ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"Decomposition Glass Stack ({ticker_name})",
                font=dict(size=18, family='Arial Black')
            ),
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                zaxis_title="Layer",
                bgcolor='#0a0a0a',
                xaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                zaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            height=700,
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_decomposition_glass_stack: {str(e)}")
        return None


def plot_seasonal_helix(df: pd.DataFrame, forecast: pd.DataFrame = None,
                       ticker: str = None, years_filter: tuple = None):
    """
    Seasonal Helix: 3D spiral untuk seasonal patterns.
    
    Konsep: Waktu berputar dalam spiral, satu putaran = 1 tahun.
    Hijau = untung, Merah = rugi.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol
    years_filter : tuple
        (min_year, max_year) untuk filter
    
    Returns:
    --------
    go.Figure : 3D Plotly figure
    """
    try:
        if df is None or df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
            return None
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Filter by years if specified
        if years_filter:
            min_year, max_year = years_filter
            df = df[(df['Date'].dt.year >= min_year) & (df['Date'].dt.year <= max_year)]
        
        if df.empty:
            return None
        
        # Calculate monthly returns
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        monthly_returns = df.groupby(['Year', 'Month'])['Close'].last().pct_change().fillna(0)
        
        # Create spiral coordinates
        years = df['Year'].unique()
        months = range(1, 13)
        
        x_coords = []
        y_coords = []
        z_coords = []
        colors = []
        hover_texts = []
        
        for year_idx, year in enumerate(years):
            for month in months:
                # Spiral: radius increases with year, angle based on month
                angle = (month - 1) * 2 * np.pi / 12
                radius = year_idx * 0.5 + 1
                
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = year_idx  # Height = year
                
                # Get return for this month
                if (year, month) in monthly_returns.index:
                    ret = monthly_returns[(year, month)]
                else:
                    ret = 0
                
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
                
                # Color: green for positive, red for negative
                if ret > 0:
                    colors.append('#00ff00')
                else:
                    colors.append('#ff0000')
                
                hover_texts.append(f"{year}-{month:02d}<br>Return: {ret:.2%}")
        
        fig = go.Figure()
        
        # Create spiral line
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            name='Seasonal Spiral',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=2),
            marker=dict(
                size=5,
                color=colors,
                line=dict(color='white', width=1)
            ),
            hovertemplate='%{text}<extra></extra>',
            text=hover_texts
        ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"Seasonal Helix: DNA of Market Cycles ({ticker_name})",
                font=dict(size=18, family='Arial Black')
            ),
            scene=dict(
                xaxis_title="X (Spiral)",
                yaxis_title="Y (Spiral)",
                zaxis_title="Year",
                bgcolor='#0a0a0a',
                xaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                zaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            height=700,
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_seasonal_helix: {str(e)}")
        return None


def plot_market_universe(df: pd.DataFrame, forecast: pd.DataFrame = None,
                        ticker: str = None, time_slider: int = None):
    """
    Market Universe: 3D motion bubble chart untuk multiple stocks.
    
    Konsep: X = Volatility, Y = Expected Return, Z = Fundamental Health.
    Bola bergerak seiring waktu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data dengan multiple tickers
    forecast : pd.DataFrame
        Forecast data
    ticker : str
        Single ticker untuk focus
    time_slider : int
        Year untuk animasi
    
    Returns:
    --------
    go.Figure : 3D Plotly figure
    """
    try:
        if df is None or df.empty:
            return None
        
        df = df.copy()
        
        # Need multiple tickers for this visualization
        if 'Ticker' not in df.columns:
            return None
        
        tickers = df['Ticker'].unique()
        if len(tickers) < 2:
            return None
        
        # Filter by time if specified
        if time_slider and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'].dt.year == time_slider]
        
        if df.empty:
            return None
        
        # Calculate metrics per ticker
        bubble_data = []
        for t in tickers:
            df_t = df[df['Ticker'] == t]
            if df_t.empty:
                continue
            
            # X: Volatility
            if 'Volatility_30d' in df_t.columns:
                volatility = df_t['Volatility_30d'].mean()
            else:
                returns = df_t['Close'].pct_change() if 'Close' in df_t.columns else pd.Series([0])
                volatility = returns.std() * 100 if len(returns) > 1 else 0
            
            # Y: Expected Return (from forecast or historical)
            if forecast is not None and not forecast.empty and 'yhat' in forecast.columns:
                expected_return = forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0] - 1 if len(forecast) > 1 else 0
            else:
                if 'Close' in df_t.columns and len(df_t) > 1:
                    expected_return = (df_t['Close'].iloc[-1] / df_t['Close'].iloc[0] - 1) * 100
                else:
                    expected_return = 0
            
            # Z: Fundamental Health (inverse of Debt/Equity)
            if 'Debt_Equity_Ratio' in df_t.columns:
                debt_equity = df_t['Debt_Equity_Ratio'].mean()
                health = 100 / (1 + debt_equity) if debt_equity > 0 else 100
            elif 'Debt_Equity' in df_t.columns:
                debt_equity = df_t['Debt_Equity'].mean()
                health = 100 / (1 + debt_equity) if debt_equity > 0 else 100
            else:
                health = 50  # Default
            
            # Bubble size: Market Cap or Volume
            if 'Volume' in df_t.columns:
                size = df_t['Volume'].mean() / 1e6  # Millions
            else:
                size = 10
            
            bubble_data.append({
                'ticker': t,
                'x': volatility,
                'y': expected_return,
                'z': health,
                'size': size
            })
        
        if not bubble_data:
            return None
        
        bubble_df = pd.DataFrame(bubble_data)
        
        fig = go.Figure()
        
        # Create bubbles
        fig.add_trace(go.Scatter3d(
            x=bubble_df['x'],
            y=bubble_df['y'],
            z=bubble_df['z'],
            mode='markers',
            name='Stocks',
            marker=dict(
                size=bubble_df['size'],
                color=bubble_df['y'],  # Color by return
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return %"),
                line=dict(color='white', width=1)
            ),
            text=bubble_df['ticker'],
            hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<br>Health: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Market Universe: Risk-Reward-Health Space",
                font=dict(size=18, family='Arial Black')
            ),
            scene=dict(
                xaxis_title="Volatility (Risk) %",
                yaxis_title="Expected Return %",
                zaxis_title="Fundamental Health",
                bgcolor='#0a0a0a',
                xaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                zaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            height=700,
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_market_universe: {str(e)}")
        return None


def plot_what_if_terrain(df: pd.DataFrame, forecast: pd.DataFrame = None,
                         ticker: str = None, interest_rate: float = 0.03,
                         inflation: float = 0.02):
    """
    What-If Terrain: 3D surface untuk simulasi ekonomi.
    
    Konsep: Permukaan topografi yang berubah berdasarkan variabel ekonomi.
    X = Waktu, Y = Variabel Ekonomi, Z = Harga Saham.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol
    interest_rate : float
        Suku bunga untuk simulasi
    inflation : float
        Inflasi untuk simulasi
    
    Returns:
    --------
    go.Figure : 3D Plotly figure
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        if 'yhat' not in forecast.columns:
            return None
        
        dates = forecast['ds']
        dates_numeric = np.arange(len(dates))
        base_yhat = forecast['yhat'].values
        
        # Create grid for surface: X = time, Y = economic variable (interest rate range)
        interest_rates = np.linspace(0.01, 0.10, 20)  # 1% to 10%
        times = dates_numeric
        
        # Create surface: price changes based on interest rate
        # Higher interest rate = lower stock price (simplified model)
        surface_z = []
        for ir in interest_rates:
            # Impact: higher IR reduces price
            impact = (ir - 0.03) * -0.5  # Normalize around 3%
            adjusted_prices = base_yhat * (1 + impact)
            surface_z.append(adjusted_prices)
        
        surface_z = np.array(surface_z)
        
        # Create meshgrid
        X, Y = np.meshgrid(times, interest_rates)
        Z = surface_z
        
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(go.Surface(
            x=X,
            y=Y * 100,  # Convert to percentage
            z=Z,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Price (USD)"),
            hovertemplate='Time: %{x}<br>Interest Rate: %{y:.2f}%<br>Price: $%{z:.2f}<extra></extra>'
        ))
        
        # Add current scenario line
        current_ir_idx = np.argmin(np.abs(interest_rates - interest_rate))
        fig.add_trace(go.Scatter3d(
            x=times,
            y=np.full(len(times), interest_rate * 100),
            z=surface_z[current_ir_idx],
            mode='lines',
            name='Current Scenario',
            line=dict(color='white', width=4),
            hovertemplate='<b>Current</b><br>Time: %{x}<br>IR: %{y:.2f}%<br>Price: $%{z:.2f}<extra></extra>'
        ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"What-If Terrain: Economic Simulation ({ticker_name})<br><sub>IR: {interest_rate*100:.1f}% | Inflation: {inflation*100:.1f}%</sub>",
                font=dict(size=18, family='Arial Black')
            ),
            scene=dict(
                xaxis_title="Time (Days Ahead)",
                yaxis_title="Interest Rate %",
                zaxis_title="Price (USD)",
                bgcolor='#0a0a0a',
                xaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                zaxis=dict(backgroundcolor='#0a0a0a', gridcolor='rgba(255, 255, 255, 0.1)'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            height=700,
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_what_if_terrain: {str(e)}")
        return None


# Stub functions untuk backward compatibility (akan diimplementasikan jika diperlukan)
def plot_forecast_bridge(model_data: dict = None, forecast: pd.DataFrame = None,
                         ticker: str = None, days_ahead: int = 30):
    """Forecast Bridge: Waterfall decomposition - redirect to glass stack"""
    return plot_decomposition_glass_stack(model_data, forecast, ticker, explode_layers=False)

def plot_seasonal_compass(model_data: dict = None, forecast: pd.DataFrame = None, ticker: str = None):
    """Seasonal Compass: Radar chart - redirect to helix"""
    return None  # Will use seasonal helix instead

def plot_risk_reward_motion(df: pd.DataFrame, ticker: str = None):
    """Risk-Reward Motion Quadrant - redirect to market universe"""
    return plot_market_universe(df, None, ticker, None)

def plot_motion_quadrant(df: pd.DataFrame, ticker: str = None):
    """Motion Quadrant - alias for risk_reward_motion"""
    return plot_risk_reward_motion(df, ticker)

# Stub functions untuk semua fungsi lain yang diperlukan
def plot_candlestick_with_forecast(*args, **kwargs):
    return None

def plot_prophet_decomposition(*args, **kwargs):
    return None

def plot_correlation_heatmap(*args, **kwargs):
    return None

def plot_roe_waterfall(*args, **kwargs):
    return None

def plot_scenario_funnel(*args, **kwargs):
    return None

def plot_radar_multi_ratio(*args, **kwargs):
    return None

def plot_3d_surface_forecast(*args, **kwargs):
    return plot_neon_time_tunnel(*args, **kwargs)

def plot_3d_scatter_decomposition(*args, **kwargs):
    return plot_decomposition_glass_stack(*args, **kwargs)

def plot_3d_funnel_scenarios(*args, **kwargs):
    return None

def plot_3d_sensitivity_mesh(*args, **kwargs):
    return plot_what_if_terrain(*args, **kwargs)

def plot_3d_economic_terrain(*args, **kwargs):
    return plot_what_if_terrain(*args, **kwargs)

def plot_3d_leverage_vortex(*args, **kwargs):
    return None

def plot_3d_cycle_globe(*args, **kwargs):
    return plot_seasonal_helix(*args, **kwargs)

def plot_3d_profit_prism(*args, **kwargs):
    return None

def plot_3d_network_galaxy(*args, **kwargs):
    return plot_market_universe(*args, **kwargs)

def plot_3d_forecast_terrain(*args, **kwargs):
    return plot_neon_time_tunnel(*args, **kwargs)

def plot_3d_ci_ribbon(*args, **kwargs):
    return plot_neon_time_tunnel(*args, **kwargs)

def plot_3d_component_orbit(*args, **kwargs):
    return plot_decomposition_glass_stack(*args, **kwargs)

def plot_3d_waterfall_dupont(*args, **kwargs):
    return None

def plot_3d_sensitivity_vortex(*args, **kwargs):
    return plot_what_if_terrain(*args, **kwargs)

def plot_fundamental_equalizer(*args, **kwargs):
    return None

def plot_trend_decomposition_stack(*args, **kwargs):
    return plot_decomposition_glass_stack(*args, **kwargs)

def plot_anomaly_traffic_light(*args, **kwargs):
    return None

def plot_scenario_simulator(*args, **kwargs):
    """Scenario Simulator - using what-if terrain"""
    return plot_what_if_terrain(*args, **kwargs)

def plot_seasonal_heatmap_calendar(*args, **kwargs):
    return None

def plot_financial_health_radar(*args, **kwargs):
    return None

def plot_volume_pressure(*args, **kwargs):
    return None

def plot_valuation_band(*args, **kwargs):
    return None
