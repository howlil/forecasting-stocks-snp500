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
        

# ========== NEW FORECASTING VISUALIZATIONS ==========

def plot_fan_chart(df: pd.DataFrame, forecast: pd.DataFrame = None, ticker: str = None):
    """
    Fan Chart: Confidence Interval Bands (Bank of England style).
    
    Standar emas bank sentral untuk forecasting. Satu garis tengah dikelilingi
    pita dengan gradasi warna berbeda (50% yakin gelap, 95% yakin pudar).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat, yhat_lower, yhat_upper
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    go.Figure : Plotly figure dengan fan chart
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        if 'yhat' not in forecast.columns:
            return None
        
        # Separate historical and future
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
            forecast_hist = forecast[forecast['y'].notna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
            forecast_hist = forecast.head(len(forecast) - 90).copy()
        
        if forecast_future.empty:
            return None
        
        dates_future = forecast_future['ds']
        yhat = forecast_future['yhat'].values
        
        # Get confidence intervals
        if 'yhat_upper' in forecast_future.columns and 'yhat_lower' in forecast_future.columns:
            yhat_upper_95 = forecast_future['yhat_upper'].values
            yhat_lower_95 = forecast_future['yhat_lower'].values
        else:
            # Estimate from yhat
            std_dev = np.std(yhat) * 1.96
            yhat_upper_95 = yhat + std_dev
            yhat_lower_95 = yhat - std_dev
        
        # Calculate 50% confidence interval (approximately 0.67 std dev)
        std_dev_50 = np.std(yhat) * 0.67
        yhat_upper_50 = yhat + std_dev_50
        yhat_lower_50 = yhat - std_dev_50
        
        fig = go.Figure()
        
        # Historical data
        if not forecast_hist.empty and 'y' in forecast_hist.columns:
            fig.add_trace(go.Scatter(
                x=forecast_hist['ds'],
                y=forecast_hist['y'],
                mode='lines+markers',
                name='Historical Price',
                line=dict(color='rgba(100, 100, 100, 0.5)', width=2),
                marker=dict(size=3, color='gray')
            ))
        
        # 95% confidence band (outer, lighter)
        fig.add_trace(go.Scatter(
            x=list(dates_future) + list(dates_future[::-1]),
            y=list(yhat_upper_95) + list(yhat_lower_95[::-1]),
                    fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=0),
            name='95% Confidence',
            hoverinfo='skip'
        ))
        
        # 50% confidence band (inner, darker)
        fig.add_trace(go.Scatter(
            x=list(dates_future) + list(dates_future[::-1]),
            y=list(yhat_upper_50) + list(yhat_lower_50[::-1]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=0),
            name='50% Confidence',
            hoverinfo='skip'
        ))
        
        # Main forecast line
        fig.add_trace(go.Scatter(
            x=dates_future,
            y=yhat,
            mode='lines',
            name='Forecast',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Calculate fan width (uncertainty expansion)
        fan_width_95 = (yhat_upper_95 - yhat_lower_95) / yhat
        fan_width_50 = (yhat_upper_50 - yhat_lower_50) / yhat
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"Fan Chart: Confidence Interval Bands ({ticker_name})<br><sub>Fan Width (95%): {fan_width_95[-1]*100:.1f}% | Risk Assessment: {'High' if fan_width_95[-1] > 0.2 else 'Moderate' if fan_width_95[-1] > 0.1 else 'Low'}</sub>",
                font=dict(size=18, family='Arial Black')
            ),
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_fan_chart: {str(e)}")
        return None


def plot_forecast_bridge(model_data: dict = None, forecast: pd.DataFrame = None,
                         ticker: str = None, days_ahead: int = 30):
    """
    Forecast Bridge: Waterfall Decomposition.
    
    Memecah harga prediksi menjadi kontribusi: Trend, Seasonality, Fundamental (ROE, Debt).
    Batang hijau = faktor positif, batang merah = faktor negatif.
    
    Parameters:
    -----------
    model_data : dict
        Prophet model data dengan komponen
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol
    days_ahead : int
        Hari ke depan untuk breakdown
    
    Returns:
    --------
    go.Figure : Waterfall chart
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        if 'yhat' not in forecast.columns:
            return None
        
        # Get forecast for specific day
        if len(forecast) < days_ahead:
            days_ahead = len(forecast) - 1
        
        forecast_future = forecast[forecast['y'].isna()].copy() if 'y' in forecast.columns else forecast.tail(90).copy()
        if len(forecast_future) < days_ahead:
            days_ahead = len(forecast_future) - 1
        
        target_forecast = forecast_future.iloc[days_ahead]
        target_date = target_forecast['ds']
        target_price = target_forecast['yhat']
        
        # Get historical last price
        forecast_hist = forecast[forecast['y'].notna()].copy() if 'y' in forecast.columns else forecast.head(len(forecast) - 90).copy()
        if forecast_hist.empty:
            base_price = target_price * 0.9  # Estimate
        else:
            base_price = forecast_hist['y'].iloc[-1] if 'y' in forecast_hist.columns else forecast_hist['yhat'].iloc[-1]
        
        # Decompose contributions
        contributions = []
        
        # Trend contribution
        if 'trend' in forecast.columns:
            trend_start = forecast_hist['trend'].iloc[-1] if not forecast_hist.empty and 'trend' in forecast_hist.columns else base_price * 0.7
            trend_end = target_forecast['trend'] if 'trend' in target_forecast else base_price * 0.7
            trend_contrib = trend_end - trend_start
            contributions.append(('Base Trend', trend_contrib, '#3498db'))
        else:
            trend_contrib = (target_price - base_price) * 0.5
            contributions.append(('Base Trend', trend_contrib, '#3498db'))
        
        # Seasonality contribution
        if 'seasonal' in forecast.columns:
            seasonal_contrib = target_forecast['seasonal'] if 'seasonal' in target_forecast else 0
            contributions.append(('Seasonality', seasonal_contrib, '#2ecc71' if seasonal_contrib >= 0 else '#e74c3c'))
        else:
            contributions.append(('Seasonality', 0, '#95a5a6'))
        
        # Regressor contributions (if available)
        regressor_cols = ['ROE', 'Debt_Equity', 'EBIT_Margin']
        for col in regressor_cols:
            if col in forecast.columns:
                regressor_value = target_forecast[col] if col in target_forecast else 0
                # Simple impact model
                if col == 'ROE':
                    impact = regressor_value * 0.1  # Positive impact
                elif col == 'Debt_Equity':
                    impact = -regressor_value * 0.05  # Negative impact
                elif col == 'EBIT_Margin':
                    impact = regressor_value * 0.15  # Positive impact
                else:
                    impact = 0
                
                contributions.append((col, impact, '#2ecc71' if impact >= 0 else '#e74c3c'))
        
        # Calculate waterfall values
        waterfall_values = [base_price]
        waterfall_labels = ['Today']
        waterfall_colors = ['#95a5a6']
        
        current_value = base_price
        for label, contrib, color in contributions:
            current_value += contrib
            waterfall_values.append(current_value)
            waterfall_labels.append(label)
            waterfall_colors.append(color)
        
        # Final price
        waterfall_values.append(target_price)
        waterfall_labels.append('Forecast')
        waterfall_colors.append('#1f77b4')
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(contributions)) + ["total"],
            x=waterfall_labels,
            textposition="outside",
            text=[f"${v:.2f}" for v in waterfall_values],
            y=[waterfall_values[0]] + [waterfall_values[i+1] - waterfall_values[i] for i in range(len(waterfall_values)-1)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#1f77b4"}}
        ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"Forecast Bridge: Price Decomposition ({ticker_name})<br><sub>Breakdown for {target_date.strftime('%Y-%m-%d')} ({days_ahead} days ahead)</sub>",
                font=dict(size=18, family='Arial Black')
            ),
            xaxis_title="Components",
                yaxis_title="Price (USD)",
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_forecast_bridge: {str(e)}")
        return None

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

def plot_seasonal_heatmap_matrix(df: pd.DataFrame, ticker: str = None):
    """
    Seasonal Heatmap Matrix: Grid kalender untuk Market Timing.
    
    Grid kotak-kotak: Sumbu X = Bulan (Jan-Des), Sumbu Y = Tahun.
    Hijau pekat = Gain tinggi, Merah pekat = Loss dalam.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, Close
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    go.Figure : Heatmap matrix
    """
    try:
        if df is None or df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
            return None
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Calculate monthly returns
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Get last price per month
        monthly_data = df.groupby(['Year', 'Month'])['Close'].last().reset_index()
        monthly_data['Return'] = monthly_data.groupby('Year')['Close'].pct_change().fillna(0) * 100
        
        # Create pivot table
        heatmap_data = monthly_data.pivot(index='Year', columns='Month', values='Return')
        heatmap_data = heatmap_data.sort_index(ascending=False)  # Latest year on top
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=heatmap_data.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            text=heatmap_data.round(1).values,
            texttemplate='%{text}%',
            textfont={"size": 9},
            colorbar=dict(title="Return %")
            ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"Seasonal Heatmap Matrix: Market Timing ({ticker_name})<br><sub>Green = Gain, Red = Loss. Look for consistent patterns across years.</sub>",
                font=dict(size=18, family='Arial Black')
            ),
            xaxis_title="Month",
            yaxis_title="Year",
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_seasonal_heatmap_matrix: {str(e)}")
        return None


def plot_seasonal_heatmap_calendar(df: pd.DataFrame, ticker: str = None):
    """Alias for seasonal heatmap matrix"""
    return plot_seasonal_heatmap_matrix(df, ticker)


def plot_regime_change(df: pd.DataFrame, forecast: pd.DataFrame = None, 
                       ticker: str = None, changepoints: list = None):
    """
    Regime Change: Trend Changepoints Detection.
    
    Menyoroti perubahan struktur pasar dengan changepoints.
    Garis berubah warna atau memiliki garis vertikal putus-putus di changepoints.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol
    changepoints : list
        List of changepoint dates (if None, will detect from forecast)
    
    Returns:
    --------
    go.Figure : Line chart dengan changepoints
    """
    try:
        if df is None or df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
            return None
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        fig = go.Figure()
        
        # Plot historical price
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='rgba(100, 100, 100, 0.7)', width=2)
        ))
        
        # Detect changepoints from trend changes
        if changepoints is None and forecast is not None and not forecast.empty:
            forecast = forecast.copy()
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            
            if 'trend' in forecast.columns:
                # Calculate trend slope changes
                forecast['trend_slope'] = forecast['trend'].diff()
                forecast['trend_change'] = forecast['trend_slope'].diff()
                
                # Find significant changepoints (slope change > threshold)
                threshold = forecast['trend_change'].std() * 2
                changepoints = forecast[abs(forecast['trend_change']) > threshold]['ds'].tolist()
        
        # Add changepoint markers
        if changepoints:
            for cp_date in changepoints:
                if isinstance(cp_date, str):
                    cp_date = pd.to_datetime(cp_date)
                elif not isinstance(cp_date, pd.Timestamp):
                    cp_date = pd.to_datetime(cp_date)
                
                # Ensure cp_date is a Timestamp for proper comparison
                if not isinstance(cp_date, pd.Timestamp):
                    cp_date = pd.Timestamp(cp_date)
                
                # Find price at changepoint - ensure Date column is datetime
                df_date_col = pd.to_datetime(df['Date'])
                mask = df_date_col <= cp_date
                cp_price = df[mask]['Close'].iloc[-1] if mask.sum() > 0 else df['Close'].iloc[-1]
                
                # Add vertical line
                fig.add_vline(
                    x=cp_date,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.5,
                    annotation_text="Regime Change"
                )
                
                # Add annotation with slope
                if forecast is not None and not forecast.empty and 'trend' in forecast.columns:
                    forecast_ds = pd.to_datetime(forecast['ds'])
                    cp_forecast = forecast[forecast_ds <= cp_date]
                    if len(cp_forecast) > 1:
                        slope_before = (cp_forecast['trend'].iloc[-1] - cp_forecast['trend'].iloc[-2]) / cp_forecast['trend'].iloc[-2] * 100
                        fig.add_annotation(
                            x=cp_date,
                            y=cp_price,
                            text=f"Slope: {slope_before:.2f}%/day",
                            showarrow=True,
                            arrowhead=2,
                            bgcolor='rgba(255, 0, 0, 0.8)',
                            bordercolor='white',
                            font=dict(color='white', size=10)
                        )
        
        # Add forecast if available
        if forecast is not None and not forecast.empty:
            forecast = forecast.copy()
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            
            if 'y' in forecast.columns:
                forecast_future = forecast[forecast['y'].isna()].copy()
            else:
                forecast_future = forecast.tail(90).copy()
            
            if not forecast_future.empty and 'yhat' in forecast_future.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#1f77b4', width=3, dash='dot')
                ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"Regime Change: Trend Changepoints ({ticker_name})<br><sub>Red dashed lines = Structural breaks. Watch for momentum shifts.</sub>",
                font=dict(size=18, family='Arial Black')
            ),
            xaxis_title="Date",
                yaxis_title="Price (USD)",
            height=600,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_regime_change: {str(e)}")
        return None


def plot_scenario_simulator(df: pd.DataFrame, forecast: pd.DataFrame = None,
                           debt_level: float = None, profit_margin: float = None,
                           interest_rate: float = None, ticker: str = None):
    """
    Scenario Simulator: Interactive Sensitivity Analysis.
    
    Grafik forecast dengan sliders untuk variabel kunci: Interest Rate, Debt Level, Profit Margin.
    Stress testing untuk melihat sensitivitas harga terhadap perubahan fundamental.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Base Prophet forecast
    debt_level : float
        Debt/Equity ratio untuk scenario
    profit_margin : float
        Profit margin untuk scenario
    interest_rate : float
        Interest rate untuk scenario
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    go.Figure : Interactive forecast dengan scenario adjustments
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        if 'yhat' not in forecast.columns:
            return None
        
        # Separate historical and future
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
            forecast_hist = forecast[forecast['y'].notna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
            forecast_hist = forecast.head(len(forecast) - 90).copy()
        
        if forecast_future.empty:
            return None
        
        dates_future = forecast_future['ds']
        base_yhat = forecast_future['yhat'].values
        
        # Get base values
        base_debt = debt_level if debt_level is not None else 0.72
        base_margin = profit_margin if profit_margin is not None else 0.29
        base_ir = interest_rate if interest_rate is not None else 0.03
        
        # Calculate scenario adjustments
        # Debt impact: higher debt = lower price
        debt_impact = (base_debt - 0.5) * -0.1
        
        # Margin impact: higher margin = higher price
        margin_impact = (base_margin - 0.2) * 0.15
        
        # Interest rate impact: higher IR = lower price
        ir_impact = (base_ir - 0.03) * -0.2
        
        # Combined impact
        total_impact = debt_impact + margin_impact + ir_impact
        adjusted_yhat = base_yhat * (1 + total_impact)
        
        fig = go.Figure()
        
        # Historical data
        if not forecast_hist.empty and 'y' in forecast_hist.columns:
            fig.add_trace(go.Scatter(
                x=forecast_hist['ds'],
                y=forecast_hist['y'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='rgba(100, 100, 100, 0.5)', width=2),
                marker=dict(size=3, color='gray')
            ))
        
        # Base forecast (gray, dashed)
        fig.add_trace(go.Scatter(
            x=dates_future,
            y=base_yhat,
            mode='lines',
            name='Base Forecast',
            line=dict(color='rgba(149, 165, 166, 0.5)', width=2, dash='dash'),
            hovertemplate='<b>Base</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Scenario forecast (bright, solid)
        fig.add_trace(go.Scatter(
            x=dates_future,
            y=adjusted_yhat,
            mode='lines+markers',
            name='Scenario Forecast',
            line=dict(color='#00ff00', width=4),
            marker=dict(size=6, color='#00ff00'),
            hovertemplate=f'<b>Scenario</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<br>Debt: {base_debt:.2f}<br>Margin: {base_margin*100:.2f}%<br>IR: {base_ir*100:.2f}%<extra></extra>'
        ))
        
        # Calculate impact annotation
        price_change = adjusted_yhat[-1] - base_yhat[-1]
        price_change_pct = (price_change / base_yhat[-1]) * 100 if base_yhat[-1] > 0 else 0
        
        fig.add_annotation(
            x=dates_future.iloc[-1],
            y=adjusted_yhat[-1],
            text=f"{'+' if price_change >= 0 else ''}${price_change:.2f}<br>({price_change_pct:+.1f}%)",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bgcolor='rgba(0, 255, 0, 0.8)' if price_change >= 0 else 'rgba(255, 0, 0, 0.8)',
            bordercolor='white',
            font=dict(color='white', size=12, weight='bold')
        )
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=dict(
                text=f"Scenario Simulator: Sensitivity Analysis ({ticker_name})<br><sub>Debt: {base_debt:.2f} | Margin: {base_margin*100:.1f}% | IR: {base_ir*100:.1f}%</sub>",
                font=dict(size=18, family='Arial Black')
            ),
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_scenario_simulator: {str(e)}")
        return None

def plot_financial_health_radar(*args, **kwargs):
    return None

def plot_volume_pressure(*args, **kwargs):
    return None

def plot_valuation_band(*args, **kwargs):
        return None
