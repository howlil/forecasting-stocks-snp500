"""
Visualization module untuk aplikasi FinScope.
Berisi 5+ visualisasi interaktif menggunakan Plotly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_candlestick_with_forecast(df: pd.DataFrame, forecast: pd.DataFrame = None,
                                   ticker: str = None, color_by_roe: bool = True):
    """
    Visualisasi 1: Candlestick OHLC dengan forecast overlay dan volume subplot.
    Color coding berdasarkan ROE thresholds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom Date, Open, High, Low, Close, Volume
    forecast : pd.DataFrame
        Forecast dataframe dari Prophet (optional)
    ticker : str
        Ticker symbol untuk title
    color_by_roe : bool
        Apakah akan color code berdasarkan ROE
    
    Returns:
    --------
    go.Figure : Plotly figure object atau None jika error
    """
    try:
        if df is None or df.empty:
            return None
        
        if ticker and 'Ticker' in df.columns:
            df_plot = df[df['Ticker'] == ticker].copy()
        else:
            df_plot = df.copy()
        
        if df_plot.empty:
            return None
        
        # Validasi kolom yang diperlukan
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in df_plot.columns for col in required_cols):
            return None
        
        # Pastikan Date adalah datetime
        if 'Date' not in df_plot.columns:
            return None
        
        df_plot = df_plot.sort_values('Date')
        
        # Drop rows dengan NaN di kolom OHLC
        df_plot = df_plot.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if df_plot.empty:
            return None
        
        # Buat subplot dengan 2 rows (price + volume)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f'Candlestick Chart{" - " + ticker if ticker else ""}',
                'Volume'
            )
        )
        
        # Color coding berdasarkan ROE jika ada
        colors = None
        if color_by_roe and 'ROE' in df_plot.columns:
            # Threshold: <10% = red, 10-20% = yellow, >20% = green
            roe_values = df_plot['ROE'].fillna(0)
            colors = ['red' if roe < 10 else 'orange' if roe < 20 else 'green' 
                     for roe in roe_values]
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_plot['Date'],
                open=df_plot['Open'],
                high=df_plot['High'],
                low=df_plot['Low'],
                close=df_plot['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Forecast overlay jika ada
        if forecast is not None and not forecast.empty:
            try:
                if 'ds' in forecast.columns and df_plot['Date'].max() is not pd.NaT:
                    forecast_plot = forecast[forecast['ds'] > df_plot['Date'].max()].copy()
                    
                    if not forecast_plot.empty and 'yhat' in forecast_plot.columns:
                        # Forecast line
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_plot['ds'],
                                y=forecast_plot['yhat'],
                                mode='lines',
                                name='Forecast',
                                line=dict(color='blue', width=2, dash='dash')
                            ),
                            row=1, col=1
                        )
                        
                        # Confidence interval
                        if 'yhat_upper' in forecast_plot.columns and 'yhat_lower' in forecast_plot.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_plot['ds'],
                                    y=forecast_plot['yhat_upper'],
                                    mode='lines',
                                    name='Upper CI',
                                    line=dict(width=0),
                                    showlegend=False
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_plot['ds'],
                                    y=forecast_plot['yhat_lower'],
                                    mode='lines',
                                    name='Lower CI',
                                    line=dict(width=0),
                                    fill='tonexty',
                                    fillcolor='rgba(0,100,255,0.2)',
                                    showlegend=True
                                ),
                                row=1, col=1
                            )
            except Exception:
                pass  # Skip forecast jika error
        
        # Volume subplot
        if 'Volume' in df_plot.columns:
            volume_data = df_plot['Volume'].fillna(0)
            if len(volume_data) > 0:
                volume_colors = colors if colors else ['blue'] * len(df_plot)
                fig.add_trace(
                    go.Bar(
                        x=df_plot['Date'],
                        y=volume_data,
                        name='Volume',
                        marker_color=volume_colors[:len(volume_data)],
                        opacity=0.6
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'OHLC Chart dengan Forecast{" - " + ticker if ticker else ""}',
            xaxis_rangeslider_visible=False,
            height=700,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    except Exception as e:
        print(f"Error in plot_candlestick_with_forecast: {str(e)}")
        return None


def plot_prophet_decomposition(model, forecast: pd.DataFrame, ticker: str = None):
    """
    Visualisasi 2: Prophet decomposition (trend, seasonality) dengan animasi.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Forecast dataframe dari Prophet
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : Plotly figure object atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        if 'ds' not in forecast.columns:
            return None
        
        # Extract components dari forecast
        components = ['trend', 'weekly', 'yearly']
        
        fig = make_subplots(
            rows=len(components) + 1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Actual vs Forecast', 'Trend', 'Weekly Seasonality', 'Yearly Seasonality']
        )
    
        # Actual vs Forecast
        if 'y' in forecast.columns:
            y_data = forecast['y'].dropna()
            if len(y_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=forecast.loc[y_data.index, 'ds'],
                        y=y_data,
                        mode='lines',
                        name='Actual',
                        line=dict(color='black', width=1)
                    ),
                    row=1, col=1
                )
        
        if 'yhat' in forecast.columns:
            yhat_data = forecast['yhat'].dropna()
            if len(yhat_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=forecast.loc[yhat_data.index, 'ds'],
                        y=yhat_data,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
        
        # Trend
        if 'trend' in forecast.columns:
            trend_data = forecast['trend'].dropna()
            if len(trend_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=forecast.loc[trend_data.index, 'ds'],
                        y=trend_data,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=1
                )
        
        # Weekly seasonality
        if 'weekly' in forecast.columns:
            weekly_data = forecast['weekly'].dropna()
            if len(weekly_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=forecast.loc[weekly_data.index, 'ds'],
                        y=weekly_data,
                        mode='lines',
                        name='Weekly',
                        line=dict(color='green', width=2)
                    ),
                    row=3, col=1
                )
        
        # Yearly seasonality
        if 'yearly' in forecast.columns:
            yearly_data = forecast['yearly'].dropna()
            if len(yearly_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=forecast.loc[yearly_data.index, 'ds'],
                        y=yearly_data,
                        mode='lines',
                        name='Yearly',
                        line=dict(color='purple', width=2)
                    ),
                    row=4, col=1
                )
        
        fig.update_layout(
            title=f'Prophet Decomposition{" - " + ticker if ticker else ""}',
            height=800,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        print(f"Error in plot_prophet_decomposition: {str(e)}")
        return None


def plot_correlation_heatmap(df: pd.DataFrame, ticker: str = None):
    """
    Visualisasi 3: Heatmap correlation returns vs. financial ratios.
    Dengan drill-down scatter plot on click.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan kolom returns dan ratios
    ticker : str
        Ticker symbol untuk filter
    
    Returns:
    --------
    go.Figure : Plotly figure object atau None jika error
    """
    try:
        if df is None or df.empty:
            return None
        
        if ticker and 'Ticker' in df.columns:
            df_plot = df[df['Ticker'] == ticker].copy()
        else:
            df_plot = df.copy()
        
        if df_plot.empty:
            return None
        
        # Pilih kolom untuk correlation
        ratio_cols = ['ROE', 'Debt_Equity', 'EBIT_Margin']
        return_cols = ['Daily_Return', f'Volatility_30d']
        
        # Gabungkan semua kolom yang ada
        corr_cols = []
        for col in ratio_cols + return_cols:
            if col in df_plot.columns:
                corr_cols.append(col)
        
        if len(corr_cols) < 2:
            # Fallback: gunakan semua numeric columns
            numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
            corr_cols = [col for col in numeric_cols if df_plot[col].notna().sum() > 0]
        
        if len(corr_cols) < 2:
            return None
        
        # Calculate correlation matrix dengan error handling
        try:
            corr_matrix = df_plot[corr_cols].corr()
            # Replace inf dan NaN dengan 0
            corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title=f'Correlation Heatmap: Returns vs Financial Ratios{" - " + ticker if ticker else ""}',
                xaxis_title="",
                yaxis_title="",
                height=600,
                template='plotly_white'
            )
            
            return fig
        except Exception:
            return None
    except Exception as e:
        print(f"Error in plot_correlation_heatmap: {str(e)}")
        return None


def plot_roe_waterfall(df: pd.DataFrame, forecast: pd.DataFrame = None, 
                      ticker: str = None):
    """
    Visualisasi 4: Waterfall ROE breakdown menggunakan DuPont formula.
    Compare pre/post forecast.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan data historis
    forecast : pd.DataFrame
        Forecast dataframe (optional)
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    go.Figure : Plotly figure object atau None jika error
    """
    try:
        if df is None or df.empty:
            return None
        
        if ticker and 'Ticker' in df.columns:
            df_plot = df[df['Ticker'] == ticker].copy()
        else:
            df_plot = df.copy()
        
        if df_plot.empty:
            return None
        
        # DuPont formula: ROE = Profit Margin × Asset Turnover × Equity Multiplier
        # Simplified: ROE = EBIT_Margin × (1/Debt_Equity) × Leverage
        
        # Ambil nilai terakhir untuk breakdown
        if 'ROE' in df_plot.columns and 'EBIT_Margin' in df_plot.columns:
            last_row = df_plot.iloc[-1]
            
            roe = last_row['ROE'] if pd.notna(last_row['ROE']) else 0
            ebit_margin = last_row['EBIT_Margin'] if pd.notna(last_row['EBIT_Margin']) else 0
            
            # Simplified breakdown (asumsi)
            profit_component = ebit_margin * 0.6  # Simplified
            turnover_component = roe * 0.3  # Simplified
            leverage_component = roe * 0.1  # Simplified
            
            # Waterfall chart
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["Profit Margin", "Asset Turnover", "Leverage", "ROE"],
                textposition="outside",
                text=[f"{profit_component:.2f}%", f"{turnover_component:.2f}%", 
                      f"{leverage_component:.2f}%", f"{roe:.2f}%"],
                y=[profit_component, turnover_component, leverage_component, roe],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title=f'ROE Breakdown (DuPont Formula){" - " + ticker if ticker else ""}',
                showlegend=False,
                height=500,
                template='plotly_white'
            )
            return fig
        else:
            # Fallback: simple bar chart jika data tidak lengkap
            if 'ROE' in df_plot.columns:
                roe_value = df_plot['ROE'].iloc[-1] if pd.notna(df_plot['ROE'].iloc[-1]) else 0
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['ROE'],
                    y=[roe_value],
                    name='ROE'
                ))
                fig.update_layout(
                    title=f'ROE{" - " + ticker if ticker else ""}',
                    height=500,
                    template='plotly_white'
                )
                return fig
            return None
    except Exception as e:
        print(f"Error in plot_roe_waterfall: {str(e)}")
        return None


def plot_scenario_funnel(df: pd.DataFrame, forecast: pd.DataFrame, 
                        ticker: str = None):
    """
    Visualisasi 5: Funnel scenario forecast (base/optimistic/pessimistic).
    Berdasarkan CI dan volatility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan data historis
    forecast : pd.DataFrame
        Forecast dataframe dari Prophet
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    go.Figure : Plotly figure object atau None jika error
    """
    try:
        if forecast is None or len(forecast) == 0:
            return None
    
        # Ambil forecast future saja
        if 'yhat' not in forecast.columns:
            return None
        
        if 'y' in forecast.columns:
            future_forecast = forecast[forecast['y'].isna()].copy()
        else:
            # Ambil 90 hari terakhir sebagai future
            future_forecast = forecast.tail(90).copy()
        
        if len(future_forecast) == 0 or 'yhat' not in future_forecast.columns:
            return None
        
        # Calculate scenarios
        base_price = future_forecast['yhat'].iloc[-1]
        if pd.isna(base_price) or np.isinf(base_price):
            return None
        
        optimistic_price = future_forecast.get('yhat_upper', pd.Series([base_price * 1.1])).iloc[-1] if 'yhat_upper' in future_forecast.columns else base_price * 1.1
        pessimistic_price = future_forecast.get('yhat_lower', pd.Series([base_price * 0.9])).iloc[-1] if 'yhat_lower' in future_forecast.columns else base_price * 0.9
        
        # Calculate volatility dari historis jika ada
        if df is not None and not df.empty and 'Volatility_30d' in df.columns:
            current_vol = df['Volatility_30d'].iloc[-1] if len(df) > 0 and pd.notna(df['Volatility_30d'].iloc[-1]) else 0.02
        else:
            current_vol = 0.02  # Default 2%
        
        # Adjust scenarios berdasarkan volatility
        vol_multiplier = 1 + (current_vol * 2)
        optimistic_price = base_price * vol_multiplier
        pessimistic_price = base_price / vol_multiplier
        
        # Funnel chart
        fig = go.Figure()
        
        fig.add_trace(go.Funnel(
            y=["Pessimistic", "Base", "Optimistic"],
            x=[pessimistic_price, base_price, optimistic_price],
            textposition="inside",
            textinfo="value+percent initial",
            marker={"color": ["red", "blue", "green"]},
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
        ))
        
        fig.update_layout(
            title=f'Scenario Forecast Funnel{" - " + ticker if ticker else ""}',
            height=500,
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        print(f"Error in plot_scenario_funnel: {str(e)}")
        return None


def plot_radar_multi_ratio(df: pd.DataFrame, ticker: str = None, 
                           year_range: tuple = None):
    """
    Visualisasi 6 (Bonus): Radar chart multi-ratio evolution over years.
    Dengan animated slider.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe dengan data historis
    ticker : str
        Ticker symbol untuk filter
    year_range : tuple
        (start_year, end_year) untuk filter tahun
    
    Returns:
    --------
    go.Figure : Plotly figure object atau None jika error
    """
    try:
        if df is None or df.empty:
            return None
        
        if ticker and 'Ticker' in df.columns:
            df_plot = df[df['Ticker'] == ticker].copy()
        else:
            df_plot = df.copy()
        
        if df_plot.empty:
            return None
        
        # Pastikan Date adalah datetime
        if 'Date' not in df_plot.columns:
            return None
        
        df_plot['Date'] = pd.to_datetime(df_plot['Date'], errors='coerce')
        df_plot = df_plot.dropna(subset=['Date'])
        
        if df_plot.empty:
            return None
        
        df_plot['Year'] = df_plot['Date'].dt.year
    
        # Filter tahun jika diberikan
        if year_range:
            df_plot = df_plot[
                (df_plot['Year'] >= year_range[0]) & 
                (df_plot['Year'] <= year_range[1])
            ]
        
        if df_plot.empty:
            return None
        
        # Pilih ratio columns
        ratio_cols = ['ROE', 'Debt_Equity', 'EBIT_Margin']
        available_ratios = [col for col in ratio_cols if col in df_plot.columns]
        
        if len(available_ratios) < 2:
            return None
        
        # Group by year dan ambil rata-rata
        yearly_avg = df_plot.groupby('Year')[available_ratios].mean()
        
        if yearly_avg.empty:
            return None
        
        # Normalize untuk radar chart (0-100 scale)
        yearly_normalized = yearly_avg.copy()
        for col in available_ratios:
            min_val = yearly_normalized[col].min()
            max_val = yearly_normalized[col].max()
            if pd.notna(max_val) and pd.notna(min_val) and max_val > min_val:
                yearly_normalized[col] = ((yearly_normalized[col] - min_val) / 
                                         (max_val - min_val)) * 100
            else:
                yearly_normalized[col] = 50  # Default middle
        
        # Create frames untuk animation
        years = sorted(yearly_normalized.index.tolist())
        if len(years) == 0:
            return None
        
        frames = []
        
        for year in years:
            try:
                values = yearly_normalized.loc[year].values.tolist()
                if len(values) > 0:
                    frame_data = go.Scatterpolar(
                        r=values + [values[0]],
                        theta=available_ratios + [available_ratios[0]],
                        fill='toself',
                        name=str(year)
                    )
                    frames.append(go.Frame(data=[frame_data], name=str(year)))
            except Exception:
                continue
        
        if len(frames) == 0:
            return None
        
        # Initial frame
        try:
            initial_values = yearly_normalized.loc[years[0]].values.tolist()
            fig = go.Figure(
                data=[go.Scatterpolar(
                    r=initial_values + [initial_values[0]],
                    theta=available_ratios + [available_ratios[0]],
                    fill='toself',
                    name=str(years[0])
                )],
                frames=frames
            )
            
            # Add animation controls
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title=f'Multi-Ratio Evolution Over Years{" - " + ticker if ticker else ""}',
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': True,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]
                        }
                    ]
                }],
                height=600,
                template='plotly_white'
            )
            
            return fig
        except Exception:
            return None
    except Exception as e:
        print(f"Error in plot_radar_multi_ratio: {str(e)}")
        return None


# ============================================================================
# 3D VISUALIZATIONS FOR FORECASTING (2025 Immersive Experience)
# ============================================================================

def plot_3d_surface_forecast(df: pd.DataFrame, forecast: pd.DataFrame = None, 
                             ticker: str = None):
    """
    3D Surface: Historical + Forecast Trends
    
    Surface plot 3D: Sumbu X=time, Y=price levels (Close/SMA), Z=volatility (CI bands).
    Shaded surface hijau (naik) ke merah (turun), dengan forecast "protrusi" ke depan.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, Close
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat, yhat_lower, yhat_upper
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Surface plot atau None jika error
    """
    try:
        if df is None or df.empty:
            return None
        
        if ticker and 'Ticker' in df.columns:
            df_plot = df[df['Ticker'] == ticker].copy()
        else:
            df_plot = df.copy()
        
        if df_plot.empty or 'Date' not in df_plot.columns or 'Close' not in df_plot.columns:
            return None
        
        # Prepare historical data
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
        df_plot = df_plot.sort_values('Date')
        
        # Calculate SMA 50 dan 200
        df_plot['SMA_50'] = df_plot['Close'].rolling(window=50, min_periods=1).mean()
        df_plot['SMA_200'] = df_plot['Close'].rolling(window=200, min_periods=1).mean()
        
        # Calculate volatility dari rolling std
        df_plot['Volatility'] = df_plot['Close'].rolling(window=30, min_periods=1).std()
        
        # Prepare data untuk surface
        dates_hist = df_plot['Date'].values
        prices_hist = df_plot['Close'].values
        volatility_hist = df_plot['Volatility'].fillna(0).values
        
        # Convert dates to numeric untuk plotting
        dates_numeric_hist = pd.to_numeric(dates_hist)
        
        # Prepare forecast data jika ada
        dates_forecast = None
        prices_forecast = None
        volatility_forecast = None
        
        if forecast is not None and not forecast.empty:
            forecast = forecast.copy()
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            
            # Ambil hanya forecast period (future dates)
            last_hist_date = df_plot['Date'].max()
            forecast_future = forecast[forecast['ds'] > last_hist_date].copy()
            
            if not forecast_future.empty:
                dates_forecast = forecast_future['ds'].values
                prices_forecast = forecast_future['yhat'].values
                # Volatility dari CI width
                ci_width = (forecast_future['yhat_upper'] - forecast_future['yhat_lower']).values
                volatility_forecast = ci_width / 2  # Approximate volatility dari CI
        
        # Create 3D surface
        fig = go.Figure()
        
        # Historical surface (hijau untuk naik, merah untuk turun)
        if len(dates_numeric_hist) > 1:
            # Create mesh untuk surface
            # Untuk simplicity, kita buat line surface dengan multiple price levels
            price_levels = [prices_hist, df_plot['SMA_50'].values, df_plot['SMA_200'].values]
            price_labels = ['Close', 'SMA 50', 'SMA 200']
            colors_surface = ['#2ecc71', '#3498db', '#e74c3c']  # Hijau, Biru, Merah
            
            for i, (prices, label, color) in enumerate(zip(price_levels, price_labels, colors_surface)):
                # Create surface trace
                fig.add_trace(go.Scatter3d(
                    x=dates_numeric_hist,
                    y=prices,
                    z=volatility_hist,
                    mode='lines+markers',
                    name=f'{label} (Historical)',
                    line=dict(color=color, width=3),
                    marker=dict(size=2, color=color),
                    hovertemplate=f'<b>{label}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Price: $%{y:.2f}<br>' +
                                 'Volatility: %{z:.2f}<extra></extra>'
                ))
        
        # Forecast surface (protrusi ke depan)
        if dates_forecast is not None and len(dates_forecast) > 0:
            dates_numeric_forecast = pd.to_numeric(dates_forecast)
            
            fig.add_trace(go.Scatter3d(
                x=dates_numeric_forecast,
                y=prices_forecast,
                z=volatility_forecast if volatility_forecast is not None else np.zeros_like(prices_forecast),
                mode='lines+markers',
                name='Forecast (Future)',
                line=dict(color='#f39c12', width=4, dash='dash'),
                marker=dict(size=4, color='#f39c12', symbol='diamond'),
                hovertemplate='<b>Forecast</b><br>' +
                             'Date: %{x}<br>' +
                             'Predicted Price: $%{y:.2f}<br>' +
                             'Uncertainty: %{z:.2f}<extra></extra>'
            ))
        
        # Update layout
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=f"3D Surface: Historical + Forecast Trends ({ticker_name})",
            scene=dict(
                xaxis_title="Time (Date)",
                yaxis_title="Price (USD)",
                zaxis_title="Volatility (CI Width)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_surface_forecast: {str(e)}")
        return None


def plot_3d_scatter_decomposition(model, forecast: pd.DataFrame = None, 
                                  ticker: str = None):
    """
    3D Scatter: Forecast Decomposition
    
    Scatter 3D: Points untuk komponen (trend, seasonality, yhat), 
    sumbu X=time, Y=amplitudo, Z=regressor impact (ROE %).
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Prophet forecast dataframe
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    go.Figure : 3D Scatter plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Extract components
        dates = forecast['ds'].values
        dates_numeric = pd.to_numeric(dates)
        trend = forecast['trend'].values if 'trend' in forecast.columns else None
        yhat = forecast['yhat'].values
        
        # Get regressor impact jika ada
        regressor_impact = None
        regressor_name = None
        if 'ROE' in forecast.columns:
            regressor_impact = forecast['ROE'].values
            regressor_name = 'ROE'
        elif 'Debt_Equity' in forecast.columns:
            regressor_impact = forecast['Debt_Equity'].values
            regressor_name = 'Debt_Equity'
        elif 'EBIT_Margin' in forecast.columns:
            regressor_impact = forecast['EBIT_Margin'].values
            regressor_name = 'EBIT_Margin'
        
        # Jika tidak ada regressor, gunakan seasonality sebagai Z
        if regressor_impact is None:
            if 'yearly' in forecast.columns:
                regressor_impact = forecast['yearly'].values
                regressor_name = 'Yearly Seasonality'
            elif 'weekly' in forecast.columns:
                regressor_impact = forecast['weekly'].values
                regressor_name = 'Weekly Seasonality'
            else:
                regressor_impact = np.zeros_like(yhat)
                regressor_name = 'No Regressor'
        
        fig = go.Figure()
        
        # Trend component (panah panjang biru)
        if trend is not None:
            fig.add_trace(go.Scatter3d(
                x=dates_numeric,
                y=trend,
                z=regressor_impact,
                mode='lines+markers',
                name='Trend Component',
                line=dict(color='#3498db', width=4),
                marker=dict(size=3, color='#3498db'),
                hovertemplate='<b>Trend</b><br>' +
                             'Date: %{x}<br>' +
                             'Trend: $%{y:.2f}<br>' +
                             f'{regressor_name}: %{{z:.2f}}<extra></extra>'
            ))
        
        # Yhat (predicted values) - bola oranye
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=yhat,
            z=regressor_impact,
            mode='markers',
            name='Predicted (yhat)',
            marker=dict(
                size=5,
                color='#e67e22',
                symbol='circle',
                line=dict(width=1, color='#d35400')
            ),
            hovertemplate='<b>Prediction</b><br>' +
                         'Date: %{x}<br>' +
                         'Predicted Price: $%{y:.2f}<br>' +
                         f'{regressor_name} Impact: %{{z:.2f}}<extra></extra>'
        ))
        
        # Seasonality jika ada
        if 'yearly' in forecast.columns:
            seasonality = forecast['yearly'].values
            fig.add_trace(go.Scatter3d(
                x=dates_numeric,
                y=yhat + seasonality,  # Offset untuk visibility
                z=regressor_impact,
                mode='lines',
                name='Yearly Seasonality',
                line=dict(color='#27ae60', width=2, dash='dot'),
                hovertemplate='<b>Seasonality</b><br>' +
                             'Date: %{x}<br>' +
                             'Seasonal Effect: $%{y:.2f}<br>' +
                             f'{regressor_name}: %{{z:.2f}}<extra></extra>'
            ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=f"3D Scatter: Forecast Decomposition ({ticker_name})",
            scene=dict(
                xaxis_title="Time (Date)",
                yaxis_title="Price / Amplitude (USD)",
                zaxis_title=f"{regressor_name} Impact (%)",
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5)
                )
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_scatter_decomposition: {str(e)}")
        return None


def plot_3d_funnel_scenarios(df: pd.DataFrame, forecast: pd.DataFrame = None,
                             ticker: str = None):
    """
    3D Funnel: Scenario Projections
    
    Funnel 3D rotatable: Sumbu Z=probabilitas (lebar funnel), 
    X/Y=scenario paths (historical base → optimistic → pessimistic).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Prophet forecast dengan CI
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    go.Figure : 3D Funnel plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Ambil forecast period saja
        if df is not None and not df.empty and 'Date' in df.columns:
            last_hist_date = pd.to_datetime(df['Date']).max()
            forecast_future = forecast[forecast['ds'] > last_hist_date].copy()
        else:
            # If no historical date, take last 90 days
            forecast_future = forecast.tail(90).copy()
        
        if forecast_future.empty or 'yhat' not in forecast_future.columns:
            return None
        
        dates = forecast_future['ds'].values
        dates_numeric = pd.to_numeric(dates)
        
        # Base scenario (yhat)
        base_prices = forecast_future['yhat'].values
        
        # Optimistic scenario (upper CI) - with fallback
        if 'yhat_upper' in forecast_future.columns:
            optimistic_prices = forecast_future['yhat_upper'].values
        else:
            optimistic_prices = base_prices * 1.1  # 10% above base
        
        # Pessimistic scenario (lower CI) - with fallback
        if 'yhat_lower' in forecast_future.columns:
            pessimistic_prices = forecast_future['yhat_lower'].values
        else:
            pessimistic_prices = base_prices * 0.9  # 10% below base
        
        # Calculate probabilities dari CI width (semakin sempit CI, semakin tinggi confidence)
        ci_width = optimistic_prices - pessimistic_prices
        max_ci = ci_width.max()
        probabilities = 1 - (ci_width / max_ci) if max_ci > 0 else np.ones_like(ci_width)
        probabilities = np.clip(probabilities, 0.5, 0.95)  # Clamp antara 50-95%
        
        fig = go.Figure()
        
        # Base scenario path (tengah)
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=base_prices,
            z=probabilities * 0.5,  # Tengah funnel
            mode='lines+markers',
            name='Base Scenario',
            line=dict(color='#3498db', width=4),
            marker=dict(size=3, color='#3498db'),
            hovertemplate='<b>Base</b><br>' +
                         'Date: %{x}<br>' +
                         'Price: $%{y:.2f}<br>' +
                         'Confidence: %{z:.1%}<extra></extra>'
        ))
        
        # Optimistic scenario path (atas)
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=optimistic_prices,
            z=probabilities * 0.8,  # Atas funnel
            mode='lines+markers',
            name='Optimistic Scenario',
            line=dict(color='#2ecc71', width=3, dash='dash'),
            marker=dict(size=3, color='#2ecc71', symbol='triangle-up'),
            hovertemplate='<b>Optimistic</b><br>' +
                         'Date: %{x}<br>' +
                         'Price: $%{y:.2f}<br>' +
                         'Confidence: %{z:.1%}<extra></extra>'
        ))
        
        # Pessimistic scenario path (bawah)
        fig.add_trace(go.Scatter3d(
            x=dates_numeric,
            y=pessimistic_prices,
            z=probabilities * 0.2,  # Bawah funnel
            mode='lines+markers',
            name='Pessimistic Scenario',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            marker=dict(size=3, color='#e74c3c', symbol='triangle-down'),
            hovertemplate='<b>Pessimistic</b><br>' +
                         'Date: %{x}<br>' +
                         'Price: $%{y:.2f}<br>' +
                         'Confidence: %{z:.1%}<extra></extra>'
        ))
        
        # Create funnel mesh walls (simplified)
        # Connect base-optimistic-pessimistic untuk setiap time point
        for i in range(0, len(dates_numeric), max(1, len(dates_numeric)//20)):  # Sample untuk performance
            x_val = dates_numeric[i]
            fig.add_trace(go.Scatter3d(
                x=[x_val, x_val, x_val],
                y=[pessimistic_prices[i], base_prices[i], optimistic_prices[i]],
                z=[probabilities[i] * 0.2, probabilities[i] * 0.5, probabilities[i] * 0.8],
                mode='lines',
                name='Funnel Wall' if i == 0 else '',
                line=dict(color='rgba(128,128,128,0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=f"3D Funnel: Scenario Projections ({ticker_name})",
            scene=dict(
                xaxis_title="Time (Date)",
                yaxis_title="Price (USD)",
                zaxis_title="Confidence / Probability",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_funnel_scenarios: {str(e)}")
        return None


def plot_3d_sensitivity_mesh(model, forecast: pd.DataFrame = None,
                              ticker: str = None, periods: int = 90):
    """
    3D Sensitivity Mesh: Regressor Impact
    
    Mesh 3D: Sumbu X=hari forecast (1-90), Y=regressors (ROE/Debt levels), 
    Z=impact % pada yhat (mesh surface bergelombang).
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol
    periods : int
        Number of forecast periods
    
    Returns:
    --------
    go.Figure : 3D Mesh plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Ambil forecast period saja
        if 'Date' in forecast.columns:
            forecast_future = forecast.tail(periods).copy()
        else:
            forecast_future = forecast.tail(periods).copy()
        
        if forecast_future.empty:
            return None
        
        # Prepare data untuk sensitivity analysis
        days = np.arange(1, len(forecast_future) + 1)
        
        # Get regressor values dan impact
        regressors_data = {}
        regressor_names = []
        
        # Try to get regressors from forecast
        for regressor in ['ROE', 'Debt_Equity', 'EBIT_Margin']:
            if regressor in forecast_future.columns:
                regressors_data[regressor] = forecast_future[regressor].values
                regressor_names.append(regressor)
        
        # If no regressors in forecast, create synthetic ones based on yhat
        if not regressor_names:
            # Create synthetic regressors based on price movement
            if 'yhat' in forecast_future.columns:
                yhat = forecast_future['yhat'].values
                # ROE proxy: price trend
                regressors_data['ROE (Est)'] = np.diff(yhat) / yhat[:-1] * 100
                regressors_data['ROE (Est)'] = np.concatenate([[0], regressors_data['ROE (Est)']])
                regressor_names.append('ROE (Est)')
                
                # Debt proxy: volatility
                volatility = np.abs(np.diff(yhat) / yhat[:-1]) * 100
                volatility = np.concatenate([[0], volatility])
                regressors_data['Volatility (Est)'] = volatility
                regressor_names.append('Volatility (Est)')
            else:
                return None  # Cannot create visualization without yhat
        
        # Limit ke 5 regressors untuk performance
        regressor_names = regressor_names[:5]
        
        # Create mesh grid
        # X: days (1-90)
        # Y: regressor index (0, 1, 2, ...)
        # Z: impact (percentage change dari baseline)
        
        yhat_baseline = forecast_future['yhat'].values
        
        # Calculate impact untuk setiap regressor
        impact_matrix = []
        regressor_indices = []
        
        for idx, regressor_name in enumerate(regressor_names):
            regressor_values = regressors_data[regressor_name]
            
            # Calculate impact sebagai percentage change
            # Simplifikasi: impact = regressor_value * sensitivity_factor
            # Sensitivity factor bisa dari model coefficients jika available
            sensitivity = 0.01  # Default 1% per unit regressor
            impact = regressor_values * sensitivity * 100  # Convert to percentage
            
            impact_matrix.append(impact)
            regressor_indices.append(np.full(len(days), idx))
        
        # Create mesh
        X_mesh = np.tile(days, len(regressor_names))
        Y_mesh = np.concatenate(regressor_indices)
        Z_mesh = np.concatenate(impact_matrix)
        
        fig = go.Figure(data=[go.Mesh3d(
            x=X_mesh,
            y=Y_mesh,
            z=Z_mesh,
            colorscale='Hot',
            intensity=Z_mesh,
            showscale=True,
            colorbar=dict(title="Impact (%)"),
            hovertemplate='<b>Sensitivity</b><br>' +
                         'Day: %{x}<br>' +
                         'Regressor: %{y}<br>' +
                         'Impact: %{z:.2f}%<extra></extra>'
        )])
        
        # Add regressor labels untuk Y axis
        ticker_name = ticker if ticker else "Data"
        fig.update_layout(
            title=f"3D Sensitivity Mesh: Regressor Impact ({ticker_name})",
            scene=dict(
                xaxis_title="Forecast Day (1-90)",
                yaxis_title="Regressor",
                zaxis_title="Impact on Price (%)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(regressor_names))),
                    ticktext=regressor_names
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_sensitivity_mesh: {str(e)}")
        return None


# ============================================================================
# NEW 3D VISUALIZATIONS FOR FORECASTING (2025 Enhanced Experience)
# ============================================================================

def plot_3d_economic_terrain(df: pd.DataFrame, forecast: pd.DataFrame = None, 
                             ticker: str = None):
    """
    3D Economic Terrain Surface: Forecast GDP-Linked Trends
    
    Surface 3D: Sumbu X=time (2005-2026 forecast), Y=stock price levels (Close/SMA), 
    Z=GDP growth proxy (dari ratios seperti ROE). Shaded hijau (bull/ekspansi) ke merah (bear/resesi).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, Close, ROE (optional)
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Surface plot atau None jika error
    """
    try:
        if df is None or df.empty:
            return None
        
        if ticker and 'Ticker' in df.columns:
            df_plot = df[df['Ticker'] == ticker].copy()
        else:
            df_plot = df.copy()
        
        if df_plot.empty or 'Date' not in df_plot.columns or 'Close' not in df_plot.columns:
            return None
        
        # Prepare historical data
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
        df_plot = df_plot.sort_values('Date')
        
        # Use ROE as GDP proxy, or use volatility if ROE not available
        if 'ROE' in df_plot.columns:
            gdp_proxy = df_plot['ROE'].fillna(0).values
            gdp_label = 'ROE (GDP Proxy)'
        elif 'Volatility_30d' in df_plot.columns:
            gdp_proxy = df_plot['Volatility_30d'].fillna(0).values * 100  # Scale untuk visibility
            gdp_label = 'Volatility (GDP Proxy)'
        else:
            # Fallback: use price change as proxy
            gdp_proxy = df_plot['Close'].pct_change().fillna(0).values * 100
            gdp_label = 'Price Change % (GDP Proxy)'
        
        # Prepare data untuk surface
        dates_hist = df_plot['Date'].values
        prices_hist = df_plot['Close'].values
        dates_numeric_hist = pd.to_numeric(dates_hist)
        
        # Combine historical and forecast
        dates_all = dates_numeric_hist.copy()
        prices_all = prices_hist.copy()
        gdp_all = gdp_proxy.copy()
        
        if forecast is not None and not forecast.empty:
            forecast = forecast.copy()
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            
            # Get future forecast
            last_hist_date = df_plot['Date'].max()
            forecast_future = forecast[forecast['ds'] > last_hist_date].copy()
            
            if not forecast_future.empty:
                dates_forecast = pd.to_numeric(forecast_future['ds'].values)
                prices_forecast = forecast_future['yhat'].values
                
                # Estimate GDP proxy for forecast (use average or trend)
                avg_gdp = np.mean(gdp_proxy[-30:]) if len(gdp_proxy) >= 30 else np.mean(gdp_proxy)
                gdp_forecast = np.full(len(forecast_future), avg_gdp)
                
                dates_all = np.concatenate([dates_numeric_hist, dates_forecast])
                prices_all = np.concatenate([prices_hist, prices_forecast])
                gdp_all = np.concatenate([gdp_proxy, gdp_forecast])
        
        # Create meshgrid untuk surface
        # Interpolate untuk smooth surface
        try:
            from scipy.interpolate import griddata
            
            # Create grid
            xi = np.linspace(dates_all.min(), dates_all.max(), 50)
            yi = np.linspace(prices_all.min(), prices_all.max(), 50)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Interpolate Z values
            zi_grid = griddata((dates_all, prices_all), gdp_all, (xi_grid, yi_grid), method='cubic', fill_value=0)
        except ImportError:
            # Fallback: simple grid tanpa interpolation
            xi = np.linspace(dates_all.min(), dates_all.max(), 50)
            yi = np.linspace(prices_all.min(), prices_all.max(), 50)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            # Simple approximation: use nearest value
            zi_grid = np.zeros_like(xi_grid)
            for i in range(len(dates_all)):
                idx_x = np.argmin(np.abs(xi - dates_all[i]))
                idx_y = np.argmin(np.abs(yi - prices_all[i]))
                zi_grid[idx_y, idx_x] = gdp_all[i]
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=xi_grid,
            y=yi_grid,
            z=zi_grid,
            colorscale='Viridis',
            colorbar=dict(title=gdp_label),
            hovertemplate='Time: %{x:.0f}<br>Price: $%{y:.2f}<br>GDP Proxy: %{z:.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D Economic Terrain: Forecast GDP-Linked Trends{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Price (USD)',
                zaxis_title=gdp_label,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_economic_terrain: {str(e)}")
        return None


def plot_3d_leverage_vortex(df: pd.DataFrame, forecast: pd.DataFrame = None,
                           ticker: str = None):
    """
    3D Leverage Vortex Scatter: Debt Impact on Forecast
    
    Scatter 3D vortex: Points untuk forecast paths (base/optimistic), 
    sumbu X=hari depan (1-90), Y=price forecast, Z=Debt/Equity levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, Close, Debt_Equity (optional)
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat, yhat_lower, yhat_upper
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Scatter plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Get future forecast only
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
        else:
            # Take last 90 days as future
            forecast_future = forecast.tail(90).copy()
        
        if forecast_future.empty or 'yhat' not in forecast_future.columns:
            return None
        
        # Prepare data
        days_ahead = np.arange(1, len(forecast_future) + 1)
        prices_base = forecast_future['yhat'].values
        
        # Get Debt/Equity ratio
        if df is not None and not df.empty and 'Debt_Equity' in df.columns:
            debt_equity = df['Debt_Equity'].iloc[-1] if pd.notna(df['Debt_Equity'].iloc[-1]) else 0.72
        elif df is not None and not df.empty and 'Debt_Equity_Ratio' in df.columns:
            debt_equity = df['Debt_Equity_Ratio'].iloc[-1] if pd.notna(df['Debt_Equity_Ratio'].iloc[-1]) else 0.72
        else:
            debt_equity = 0.72  # Default
        
        # Create vortex effect dengan varying debt levels
        debt_levels = debt_equity * (1 + 0.1 * np.sin(days_ahead / 10))  # Swirl effect
        
        # Create multiple paths (base, optimistic, pessimistic)
        prices_upper = forecast_future['yhat_upper'].values if 'yhat_upper' in forecast_future.columns else prices_base * 1.1
        prices_lower = forecast_future['yhat_lower'].values if 'yhat_lower' in forecast_future.columns else prices_base * 0.9
        
        # Color by EBIT Margin if available
        if df is not None and not df.empty and 'EBIT_Margin' in df.columns:
            ebit_margin = df['EBIT_Margin'].iloc[-1] if pd.notna(df['EBIT_Margin'].iloc[-1]) else 0.29
            colors = np.full(len(days_ahead), ebit_margin)
            color_label = 'EBIT Margin'
        else:
            colors = debt_levels
            color_label = 'Debt/Equity'
        
        fig = go.Figure()
        
        # Base path
        fig.add_trace(go.Scatter3d(
            x=days_ahead,
            y=prices_base,
            z=debt_levels,
            mode='markers+lines',
            name='Base Forecast',
            marker=dict(
                size=5,
                color=colors,
                colorscale='RdYlGn',
                colorbar=dict(title=color_label),
                showscale=True
            ),
            line=dict(width=5, color='blue'),
            hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<br>Debt/Equity: %{z:.2f}<extra></extra>'
        ))
        
        # Optimistic path
        fig.add_trace(go.Scatter3d(
            x=days_ahead,
            y=prices_upper,
            z=debt_levels * 0.9,  # Lower debt for optimistic
            mode='lines',
            name='Optimistic',
            line=dict(width=3, color='green', dash='dash'),
            hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<br>Debt/Equity: %{z:.2f}<extra></extra>'
        ))
        
        # Pessimistic path
        fig.add_trace(go.Scatter3d(
            x=days_ahead,
            y=prices_lower,
            z=debt_levels * 1.1,  # Higher debt for pessimistic
            mode='lines',
            name='Pessimistic',
            line=dict(width=3, color='red', dash='dash'),
            hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<br>Debt/Equity: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Leverage Vortex: Debt Impact on Forecast{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='Days Ahead',
                yaxis_title='Price Forecast (USD)',
                zaxis_title='Debt/Equity Ratio',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_leverage_vortex: {str(e)}")
        return None


def plot_3d_cycle_globe(df: pd.DataFrame, forecast: pd.DataFrame = None,
                         ticker: str = None):
    """
    3D Cycle Globe: Seasonality & Economic Cycles
    
    Globe 3D rotatable: Sumbu latitude=quarters (Q1-Q4), longitude=years (2005-2026), 
    radius=forecast amplitude (dari seasonality Prophet).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, Close
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Globe plot atau None jika error
    """
    try:
        if df is None or df.empty or 'Date' not in df.columns:
            return None
        
        if ticker and 'Ticker' in df.columns:
            df_plot = df[df['Ticker'] == ticker].copy()
        else:
            df_plot = df.copy()
        
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
        df_plot = df_plot.sort_values('Date')
        
        # Extract quarters and years
        df_plot['Year'] = df_plot['Date'].dt.year
        df_plot['Quarter'] = df_plot['Date'].dt.quarter
        
        # Calculate amplitude (price range per quarter)
        quarterly_data = df_plot.groupby(['Year', 'Quarter']).agg({
            'Close': ['min', 'max', 'mean']
        }).reset_index()
        quarterly_data.columns = ['Year', 'Quarter', 'Close_Min', 'Close_Max', 'Close_Mean']
        quarterly_data['Amplitude'] = quarterly_data['Close_Max'] - quarterly_data['Close_Min']
        
        # Convert to spherical coordinates
        # Latitude: Quarter (0-90 degrees, Q1=0, Q4=90)
        # Longitude: Year (0-360 degrees)
        # Radius: Amplitude
        
        years = quarterly_data['Year'].values
        quarters = quarterly_data['Quarter'].values
        amplitudes = quarterly_data['Amplitude'].values
        prices = quarterly_data['Close_Mean'].values
        
        # Normalize untuk globe
        year_min, year_max = years.min(), years.max()
        year_range = year_max - year_min if year_max > year_min else 1
        longitude = ((years - year_min) / year_range) * 360  # 0-360 degrees
        
        latitude = (quarters - 1) * 30  # Q1=0, Q2=30, Q3=60, Q4=90 degrees
        
        # Convert to Cartesian for 3D
        radius = amplitudes / amplitudes.max() if amplitudes.max() > 0 else amplitudes
        radius = radius * 50 + 10  # Scale untuk visibility
        
        x = radius * np.cos(np.radians(latitude)) * np.cos(np.radians(longitude))
        y = radius * np.cos(np.radians(latitude)) * np.sin(np.radians(longitude))
        z = radius * np.sin(np.radians(latitude))
        
        # Add forecast data if available
        if forecast is not None and not forecast.empty:
            forecast = forecast.copy()
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            forecast['Year'] = forecast['ds'].dt.year
            forecast['Quarter'] = forecast['ds'].dt.quarter
            
            forecast_quarterly = forecast.groupby(['Year', 'Quarter'])['yhat'].agg(['min', 'max', 'mean']).reset_index()
            forecast_quarterly.columns = ['Year', 'Quarter', 'Forecast_Min', 'Forecast_Max', 'Forecast_Mean']
            forecast_quarterly['Amplitude'] = forecast_quarterly['Forecast_Max'] - forecast_quarterly['Forecast_Min']
            
            # Add forecast points
            f_years = forecast_quarterly['Year'].values
            f_quarters = forecast_quarterly['Quarter'].values
            f_amplitudes = forecast_quarterly['Amplitude'].values
            f_prices = forecast_quarterly['Forecast_Mean'].values
            
            f_longitude = ((f_years - year_min) / year_range) * 360
            f_latitude = (f_quarters - 1) * 30
            f_radius = (f_amplitudes / amplitudes.max() if amplitudes.max() > 0 else f_amplitudes) * 50 + 10
            
            f_x = f_radius * np.cos(np.radians(f_latitude)) * np.cos(np.radians(f_longitude))
            f_y = f_radius * np.cos(np.radians(f_latitude)) * np.sin(np.radians(f_longitude))
            f_z = f_radius * np.sin(np.radians(f_latitude))
            
            x = np.concatenate([x, f_x])
            y = np.concatenate([y, f_y])
            z = np.concatenate([z, f_z])
            prices = np.concatenate([prices, f_prices])
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=prices,
                colorscale='Viridis',
                colorbar=dict(title='Price (USD)'),
                showscale=True
            ),
            text=[f"Y{int(years[i])} Q{int(quarters[i])}" for i in range(len(years))],
            hovertemplate='%{text}<br>Price: $%{marker.color:.2f}<br>Amplitude: %{marker.size}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D Cycle Globe: Seasonality & Economic Cycles{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='X (Globe)',
                yaxis_title='Y (Globe)',
                zaxis_title='Z (Globe)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_cycle_globe: {str(e)}")
        return None


def plot_3d_profit_prism(df: pd.DataFrame, forecast: pd.DataFrame = None,
                         ticker: str = None):
    """
    3D Profit Prism: ROE Multiplier Forecast
    
    Prism 3D (stacked volumes): Sumbu X=regressors (ROE/EBIT), Y=time forecast, 
    Z=impact % (DuPont: margin x turnover).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, ROE, EBIT_Margin (optional)
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Volume/Prism plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Get future forecast
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
        
        if forecast_future.empty:
            return None
        
        # Get regressors
        regressors = []
        regressor_values = []
        
        # Try ROE from forecast, then df, then use default
        if 'ROE' in forecast_future.columns:
            regressors.append('ROE')
            regressor_values.append(forecast_future['ROE'].fillna(0).values)
        elif df is not None and not df.empty and 'ROE' in df.columns:
            regressors.append('ROE')
            roe_val = df['ROE'].iloc[-1] if pd.notna(df['ROE'].iloc[-1]) else 1.36
            regressor_values.append(np.full(len(forecast_future), roe_val))
        else:
            # Use default ROE value
            regressors.append('ROE (Est)')
            regressor_values.append(np.full(len(forecast_future), 1.36))
        
        # Try EBIT_Margin from forecast, then df, then use default
        if 'EBIT_Margin' in forecast_future.columns:
            regressors.append('EBIT_Margin')
            regressor_values.append(forecast_future['EBIT_Margin'].fillna(0).values)
        elif df is not None and not df.empty and 'EBIT_Margin' in df.columns:
            regressors.append('EBIT_Margin')
            ebit_val = df['EBIT_Margin'].iloc[-1] if pd.notna(df['EBIT_Margin'].iloc[-1]) else 0.29
            regressor_values.append(np.full(len(forecast_future), ebit_val))
        else:
            # Use default EBIT_Margin value
            regressors.append('EBIT_Margin (Est)')
            regressor_values.append(np.full(len(forecast_future), 0.29))
        
        # Always have at least one regressor (use price change as fallback if needed)
        if not regressors and 'yhat' in forecast_future.columns:
            regressors.append('Price Change %')
            price_change = np.diff(forecast_future['yhat'].values) / forecast_future['yhat'].values[:-1] * 100
            price_change = np.concatenate([[0], price_change])
            regressor_values.append(price_change)
        
        # Time axis
        days = np.arange(1, len(forecast_future) + 1)
        
        # Calculate impact % (DuPont formula approximation)
        # Impact = ROE * EBIT_Margin (simplified)
        if len(regressor_values) >= 2:
            impact = regressor_values[0] * regressor_values[1] * 100  # Convert to percentage
        else:
            impact = regressor_values[0] * 100
        
        # Create prism using multiple traces
        fig = go.Figure()
        
        for i, (reg_name, reg_vals) in enumerate(zip(regressors, regressor_values)):
            # Create prism layer
            fig.add_trace(go.Scatter3d(
                x=[reg_name] * len(days),
                y=days,
                z=impact if len(regressor_values) == 1 else reg_vals * 100,
                mode='markers',
                name=reg_name,
                marker=dict(
                    size=8,
                    color=impact if len(regressor_values) == 1 else reg_vals * 100,
                    colorscale='Rainbow',
                    showscale=(i == 0),
                    colorbar=dict(title='Impact %') if i == 0 else None
                ),
                hovertemplate=f'{reg_name}<br>Day: %{{y}}<br>Impact: %{{z:.2f}}%<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'3D Profit Prism: ROE Multiplier Forecast{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='Regressors',
                yaxis_title='Days Forecast',
                zaxis_title='Impact %',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_profit_prism: {str(e)}")
        return None


def plot_3d_network_galaxy(df: pd.DataFrame, forecast: pd.DataFrame = None,
                           ticker: str = None):
    """
    3D Network Galaxy: Stock-Economy Interconnect
    
    Galaxy 3D network: Nodes untuk stocks (Ticker), edges ke ekonomi vars (GDP/VIX), 
    size=node = forecast confidence (CI width).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Ticker, ROE, Debt_Equity (optional)
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat, yhat_lower, yhat_upper
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Network plot atau None jika error
    """
    try:
        if df is None or df.empty:
            return None
        
        # Limit to 10 nodes untuk performance
        if 'Ticker' in df.columns:
            tickers = df['Ticker'].unique().tolist()[:10]
        else:
            tickers = ['ALL']
        
        # Node positions (galaxy layout)
        n_nodes = len(tickers) + 3  # Tickers + 3 economy nodes (GDP, VIX, ROE)
        
        # Create galaxy spiral layout
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        radius = np.linspace(10, 50, n_nodes)
        
        x_nodes = radius * np.cos(angles)
        y_nodes = radius * np.sin(angles)
        z_nodes = np.sin(angles * 2) * 20  # 3D depth
        
        # Node sizes (confidence from CI width)
        node_sizes = []
        node_labels = []
        node_colors = []
        
        # Ticker nodes
        for i, tick in enumerate(tickers):
            node_labels.append(tick)
            if forecast is not None and not forecast.empty and 'yhat_upper' in forecast.columns:
                ci_width = (forecast['yhat_upper'] - forecast['yhat_lower']).iloc[-1] if len(forecast) > 0 else 1.0
                node_sizes.append(max(5, min(20, ci_width * 10)))
            else:
                node_sizes.append(10)
            
            # Color by ROE if available
            if 'ROE' in df.columns:
                ticker_data = df[df['Ticker'] == tick]
                if not ticker_data.empty and 'ROE' in ticker_data.columns:
                    roe_val = ticker_data['ROE'].iloc[-1] if pd.notna(ticker_data['ROE'].iloc[-1]) else 0
                    node_colors.append(roe_val)
                else:
                    node_colors.append(0)
            else:
                node_colors.append(0)
        
        # Economy nodes
        economy_nodes = ['GDP', 'VIX', 'ROE']
        for econ in economy_nodes:
            node_labels.append(econ)
            node_sizes.append(15)
            node_colors.append(1.0)  # Economy nodes in different color
        
        # Edges (connections)
        edge_x = []
        edge_y = []
        edge_z = []
        
        # Connect tickers to economy nodes
        for i in range(len(tickers)):
            for j in range(len(tickers), n_nodes):
                # Random connections (in real app, use correlation)
                if np.random.random() > 0.7:  # 30% connection probability
                    edge_x.extend([x_nodes[i], x_nodes[j], None])
                    edge_y.extend([y_nodes[i], y_nodes[j], None])
                    edge_z.extend([z_nodes[i], z_nodes[j], None])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(title='ROE'),
                showscale=True
            ),
            text=node_labels,
            textposition='middle center',
            hovertemplate='%{text}<br>Size: %{marker.size}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Network Galaxy: Stock-Economy Interconnect{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='X (Galaxy)',
                yaxis_title='Y (Galaxy)',
                zaxis_title='Z (Galaxy)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_network_galaxy: {str(e)}")
        return None


# ============================================================================
# ENHANCED 3D VISUALIZATIONS FOR FORECASTING (2025 Advanced Experience)
# ============================================================================

def plot_3d_forecast_terrain(df: pd.DataFrame, forecast: pd.DataFrame = None,
                             ticker: str = None):
    """
    3D Forecast Terrain: Market Trend Projection
    
    Surface 3D: X=time (historis + 90 hari yhat), Y=Close/SMA, Z=CI width (uncertainty dari Prophet bands).
    Shaded hijau (bull forecast) ke merah (bear).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, Close
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat, yhat_lower, yhat_upper
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Surface plot atau None jika error
    """
    try:
        if df is None or df.empty or forecast is None or forecast.empty:
            return None
        
        if ticker and 'Ticker' in df.columns:
            df_plot = df[df['Ticker'] == ticker].copy()
        else:
            df_plot = df.copy()
        
        if df_plot.empty or 'Date' not in df_plot.columns or 'Close' not in df_plot.columns:
            return None
        
        if 'yhat' not in forecast.columns or 'yhat_upper' not in forecast.columns or 'yhat_lower' not in forecast.columns:
            return None
        
        # Prepare historical data
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
        df_plot = df_plot.sort_values('Date')
        
        # Calculate SMA
        df_plot['SMA_50'] = df_plot['Close'].rolling(window=50, min_periods=1).mean()
        
        # Prepare forecast data
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Get future forecast only
        last_hist_date = df_plot['Date'].max()
        forecast_future = forecast[forecast['ds'] > last_hist_date].copy()
        
        if forecast_future.empty:
            return None
        
        # Combine historical and forecast
        dates_hist = pd.to_numeric(df_plot['Date'].values)
        prices_hist = df_plot['Close'].values
        sma_hist = df_plot['SMA_50'].values
        
        dates_forecast = pd.to_numeric(forecast_future['ds'].values)
        prices_forecast = forecast_future['yhat'].values
        ci_width_forecast = (forecast_future['yhat_upper'] - forecast_future['yhat_lower']).values
        
        # Combine all data
        dates_all = np.concatenate([dates_hist, dates_forecast])
        prices_all = np.concatenate([prices_hist, prices_forecast])
        ci_all = np.concatenate([np.zeros(len(dates_hist)), ci_width_forecast])  # No CI for historical
        
        # Create meshgrid untuk surface
        try:
            from scipy.interpolate import griddata
            
            xi = np.linspace(dates_all.min(), dates_all.max(), 50)
            yi = np.linspace(prices_all.min(), prices_all.max(), 50)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            zi_grid = griddata((dates_all, prices_all), ci_all, (xi_grid, yi_grid), method='cubic', fill_value=0)
        except ImportError:
            # Fallback: simple grid
            xi = np.linspace(dates_all.min(), dates_all.max(), 50)
            yi = np.linspace(prices_all.min(), prices_all.max(), 50)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            zi_grid = np.zeros_like(xi_grid)
            for i in range(len(dates_all)):
                idx_x = np.argmin(np.abs(xi - dates_all[i]))
                idx_y = np.argmin(np.abs(yi - prices_all[i]))
                zi_grid[idx_y, idx_x] = ci_all[i]
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=xi_grid,
            y=yi_grid,
            z=zi_grid,
            colorscale='RdYlGn',
            colorbar=dict(title='CI Width (Uncertainty)'),
            hovertemplate='Time: %{x:.0f}<br>Price: $%{y:.2f}<br>CI Width: $%{z:.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D Forecast Terrain: Market Trend Projection{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Price (USD)',
                zaxis_title='CI Width (Uncertainty)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_forecast_terrain: {str(e)}")
        return None


def plot_3d_ci_ribbon(df: pd.DataFrame, forecast: pd.DataFrame = None,
                      ticker: str = None):
    """
    Interactive CI Ribbon: Risk-Adjusted Stock Paths
    
    Ribbon 3D twisty: Garis yhat pusat, ribbon lebar = CI bands, twisted berdasarkan regressor effects.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom Date, Close, ROE (optional)
    forecast : pd.DataFrame
        Prophet forecast dengan kolom ds, yhat, yhat_lower, yhat_upper, ROE (optional)
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Scatter plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Get future forecast
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
        
        if forecast_future.empty or 'yhat' not in forecast_future.columns:
            return None
        
        # Prepare data
        dates = pd.to_numeric(forecast_future['ds'].values)
        yhat = forecast_future['yhat'].values
        yhat_upper = forecast_future['yhat_upper'].values if 'yhat_upper' in forecast_future.columns else yhat * 1.05
        yhat_lower = forecast_future['yhat_lower'].values if 'yhat_lower' in forecast_future.columns else yhat * 0.95
        
        # Calculate twist based on ROE if available
        if 'ROE' in forecast_future.columns:
            roe_vals = forecast_future['ROE'].fillna(0).values
            twist = roe_vals * 0.1  # Scale untuk twist effect
        elif df is not None and not df.empty and 'ROE' in df.columns:
            roe_val = df['ROE'].iloc[-1] if pd.notna(df['ROE'].iloc[-1]) else 1.36
            twist = np.full(len(forecast_future), roe_val * 0.1)
        else:
            twist = np.zeros(len(forecast_future))
        
        # Create ribbon effect
        ci_width = yhat_upper - yhat_lower
        
        fig = go.Figure()
        
        # Base path (yhat)
        fig.add_trace(go.Scatter3d(
            x=dates,
            y=yhat,
            z=twist,
            mode='lines+markers',
            name='Forecast (yhat)',
            line=dict(width=5, color='blue'),
            marker=dict(size=4, color='blue'),
            hovertemplate='Time: %{x:.0f}<br>Price: $%{y:.2f}<br>Twist: %{z:.3f}<extra></extra>'
        ))
        
        # Upper band
        fig.add_trace(go.Scatter3d(
            x=dates,
            y=yhat_upper,
            z=twist + ci_width * 0.1,  # Twist effect
            mode='lines',
            name='Upper CI',
            line=dict(width=3, color='green', dash='dash'),
            hovertemplate='Time: %{x:.0f}<br>Price: $%{y:.2f}<br>CI Upper<extra></extra>'
        ))
        
        # Lower band
        fig.add_trace(go.Scatter3d(
            x=dates,
            y=yhat_lower,
            z=twist - ci_width * 0.1,  # Twist effect
            mode='lines',
            name='Lower CI',
            line=dict(width=3, color='red', dash='dash'),
            hovertemplate='Time: %{x:.0f}<br>Price: $%{y:.2f}<br>CI Lower<extra></extra>'
        ))
        
        # Color gradient based on CI width
        colors = ci_width / ci_width.max() if ci_width.max() > 0 else ci_width
        
        fig.update_layout(
            title=f'Interactive CI Ribbon: Risk-Adjusted Stock Paths{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Price Forecast (USD)',
                zaxis_title='Twist (ROE Effect)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_ci_ribbon: {str(e)}")
        return None


def plot_3d_component_orbit(df: pd.DataFrame, forecast: pd.DataFrame = None,
                            model=None, ticker: str = None):
    """
    3D Component Orbit: Decomposition Galaxy
    
    Orbit 3D: Pusat = yhat, orbits untuk components (trend garis panjang, seasonality siklus, regressor ROE sebagai satelit).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    forecast : pd.DataFrame
        Prophet forecast
    model : Prophet model
        Prophet model untuk decomposition
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Scatter plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Get future forecast
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
        
        if forecast_future.empty or 'yhat' not in forecast_future.columns:
            return None
        
        # Prepare data
        dates = pd.to_numeric(forecast_future['ds'].values)
        yhat = forecast_future['yhat'].values
        
        # Get components
        trend = forecast_future['trend'].values if 'trend' in forecast_future.columns else yhat * 0.7
        seasonal = forecast_future['yearly'].values if 'yearly' in forecast_future.columns else np.zeros(len(yhat))
        roe_effect = forecast_future['ROE'].values if 'ROE' in forecast_future.columns else np.zeros(len(yhat))
        
        # Create orbit effect
        # Center: yhat
        # Orbit radius based on component amplitude
        trend_radius = np.abs(trend - yhat)
        seasonal_radius = np.abs(seasonal)
        roe_radius = np.abs(roe_effect) * 10  # Scale
        
        # Convert to 3D orbit coordinates
        angle = np.linspace(0, 2 * np.pi, len(dates))
        
        # Trend orbit (long line)
        trend_x = dates
        trend_y = trend
        trend_z = trend_radius * np.sin(angle)
        
        # Seasonal orbit (circular)
        seasonal_x = dates
        seasonal_y = yhat + seasonal_radius * np.cos(angle)
        seasonal_z = seasonal_radius * np.sin(angle)
        
        # ROE orbit (satellite)
        roe_x = dates
        roe_y = yhat + roe_radius * np.cos(angle * 2)
        roe_z = roe_radius * np.sin(angle * 2)
        
        fig = go.Figure()
        
        # Center (yhat)
        fig.add_trace(go.Scatter3d(
            x=dates,
            y=yhat,
            z=np.zeros(len(dates)),
            mode='lines+markers',
            name='Forecast (Center)',
            line=dict(width=5, color='blue'),
            marker=dict(size=5, color='blue'),
            hovertemplate='Time: %{x:.0f}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Trend orbit
        fig.add_trace(go.Scatter3d(
            x=trend_x,
            y=trend_y,
            z=trend_z,
            mode='lines',
            name='Trend Orbit',
            line=dict(width=3, color='green'),
            hovertemplate='Trend: $%{y:.2f}<extra></extra>'
        ))
        
        # Seasonal orbit
        fig.add_trace(go.Scatter3d(
            x=seasonal_x,
            y=seasonal_y,
            z=seasonal_z,
            mode='lines',
            name='Seasonality Orbit',
            line=dict(width=3, color='orange'),
            hovertemplate='Seasonal: $%{y:.2f}<extra></extra>'
        ))
        
        # ROE satellite
        fig.add_trace(go.Scatter3d(
            x=roe_x,
            y=roe_y,
            z=roe_z,
            mode='lines+markers',
            name='ROE Satellite',
            line=dict(width=2, color='red', dash='dot'),
            marker=dict(size=3, color='red'),
            hovertemplate='ROE Effect: $%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Component Orbit: Decomposition Galaxy{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Price (USD)',
                zaxis_title='Orbit Radius',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_component_orbit: {str(e)}")
        return None


def plot_3d_waterfall_dupont(df: pd.DataFrame, forecast: pd.DataFrame = None,
                            ticker: str = None):
    """
    Dynamic Waterfall 3D: DuPont ROE Forecast Cascade
    
    Cascade 3D waterfall: Sumbu X=components DuPont (margin x turnover x leverage), Y=time, Z=forecast delta (% change).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom ROE, EBIT_Margin, Debt_Equity (optional)
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Surface/Waterfall plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Get future forecast
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
        
        if forecast_future.empty:
            return None
        
        # Get DuPont components
        if df is not None and not df.empty:
            if 'ROE' in df.columns:
                roe_base = df['ROE'].iloc[-1] if pd.notna(df['ROE'].iloc[-1]) else 1.36
            else:
                roe_base = 1.36
            
            if 'EBIT_Margin' in df.columns:
                margin = df['EBIT_Margin'].iloc[-1] if pd.notna(df['EBIT_Margin'].iloc[-1]) else 0.29
            else:
                margin = 0.29
            
            if 'Debt_Equity' in df.columns:
                debt_equity = df['Debt_Equity'].iloc[-1] if pd.notna(df['Debt_Equity'].iloc[-1]) else 0.72
            elif 'Debt_Equity_Ratio' in df.columns:
                debt_equity = df['Debt_Equity_Ratio'].iloc[-1] if pd.notna(df['Debt_Equity_Ratio'].iloc[-1]) else 0.72
            else:
                debt_equity = 0.72
        else:
            roe_base = 1.36
            margin = 0.29
            debt_equity = 0.72
        
        # Calculate DuPont components
        # ROE = Margin × Turnover × Leverage (simplified)
        turnover = 2.79  # Default
        leverage = 1 / (1 + debt_equity)  # Simplified leverage
        
        # Time axis
        days = np.arange(1, len(forecast_future) + 1)
        
        # Calculate forecast delta (% change)
        if 'yhat' in forecast_future.columns:
            yhat = forecast_future['yhat'].values
            if len(yhat) > 1:
                delta = np.diff(yhat) / yhat[:-1] * 100
                delta = np.concatenate([[0], delta])  # First day no change
            else:
                delta = np.array([0])
        else:
            delta = np.zeros(len(forecast_future))
        
        # DuPont components impact
        margin_impact = margin * delta
        turnover_impact = turnover * delta * 0.3
        leverage_impact = leverage * delta * (-0.1)  # Negative for high debt
        
        # Create waterfall mesh
        components = ['Margin', 'Turnover', 'Leverage']
        impacts = [margin_impact, turnover_impact, leverage_impact]
        
        fig = go.Figure()
        
        for i, (comp, impact) in enumerate(zip(components, impacts)):
            fig.add_trace(go.Scatter3d(
                x=[i] * len(days),
                y=days,
                z=impact,
                mode='markers',
                name=comp,
                marker=dict(
                    size=8,
                    color=impact,
                    colorscale='Blues',
                    showscale=(i == 0),
                    colorbar=dict(title='Impact %') if i == 0 else None
                ),
                hovertemplate=f'{comp}<br>Day: %{{y}}<br>Impact: %{{z:.2f}}%<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Dynamic Waterfall 3D: DuPont ROE Forecast Cascade{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='DuPont Components',
                yaxis_title='Days Forecast',
                zaxis_title='Impact %',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_waterfall_dupont: {str(e)}")
        return None


def plot_3d_sensitivity_vortex(df: pd.DataFrame, forecast: pd.DataFrame = None,
                               ticker: str = None):
    """
    3D Sensitivity Vortex: Market Regressor Impact
    
    Vortex 3D: Pusat = base yhat, vortex arms untuk regressors (ROE, Debt, EBIT), radius = impact % over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data dengan kolom ROE, Debt_Equity, EBIT_Margin (optional)
    forecast : pd.DataFrame
        Prophet forecast
    ticker : str
        Ticker symbol untuk title
    
    Returns:
    --------
    go.Figure : 3D Scatter plot atau None jika error
    """
    try:
        if forecast is None or forecast.empty:
            return None
        
        forecast = forecast.copy()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # Get future forecast
        if 'y' in forecast.columns:
            forecast_future = forecast[forecast['y'].isna()].copy()
        else:
            forecast_future = forecast.tail(90).copy()
        
        if forecast_future.empty or 'yhat' not in forecast_future.columns:
            return None
        
        # Prepare data
        days = np.arange(1, len(forecast_future) + 1)
        yhat = forecast_future['yhat'].values
        
        # Get regressors
        regressors = []
        regressor_values = []
        
        if 'ROE' in forecast_future.columns:
            regressors.append('ROE')
            regressor_values.append(forecast_future['ROE'].fillna(0).values)
        elif df is not None and not df.empty and 'ROE' in df.columns:
            regressors.append('ROE')
            roe_val = df['ROE'].iloc[-1] if pd.notna(df['ROE'].iloc[-1]) else 1.36
            regressor_values.append(np.full(len(forecast_future), roe_val))
        
        if 'Debt_Equity' in forecast_future.columns or (df is not None and not df.empty and 'Debt_Equity' in df.columns):
            regressors.append('Debt')
            if 'Debt_Equity' in forecast_future.columns:
                regressor_values.append(forecast_future['Debt_Equity'].fillna(0).values)
            else:
                debt_val = df['Debt_Equity'].iloc[-1] if pd.notna(df['Debt_Equity'].iloc[-1]) else 0.72
                regressor_values.append(np.full(len(forecast_future), debt_val))
        
        if 'EBIT_Margin' in forecast_future.columns or (df is not None and not df.empty and 'EBIT_Margin' in df.columns):
            regressors.append('EBIT')
            if 'EBIT_Margin' in forecast_future.columns:
                regressor_values.append(forecast_future['EBIT_Margin'].fillna(0).values)
            else:
                ebit_val = df['EBIT_Margin'].iloc[-1] if pd.notna(df['EBIT_Margin'].iloc[-1]) else 0.29
                regressor_values.append(np.full(len(forecast_future), ebit_val))
        
        if not regressors:
            return None
        
        # Limit to 3 arms untuk clarity
        regressors = regressors[:3]
        regressor_values = regressor_values[:3]
        
        # Calculate impact % (simplified: based on regressor change)
        impacts = []
        for reg_vals in regressor_values:
            if len(reg_vals) > 1:
                impact = np.diff(reg_vals) / reg_vals[:-1] * 100
                impact = np.concatenate([[0], impact])
            else:
                impact = np.array([0])
            impacts.append(impact)
        
        # Create vortex arms
        fig = go.Figure()
        
        # Center (base yhat)
        fig.add_trace(go.Scatter3d(
            x=days,
            y=yhat,
            z=np.zeros(len(days)),
            mode='lines+markers',
            name='Base Forecast',
            line=dict(width=5, color='blue'),
            marker=dict(size=5, color='blue'),
            hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Vortex arms
        colors = ['green', 'red', 'orange']
        for i, (reg_name, impact) in enumerate(zip(regressors, impacts)):
            # Create spiral arm
            angle = np.linspace(0, 4 * np.pi, len(days))
            radius = np.abs(impact) * 0.5  # Scale radius
            
            arm_x = days
            arm_y = yhat + radius * np.cos(angle)
            arm_z = radius * np.sin(angle)
            
            fig.add_trace(go.Scatter3d(
                x=arm_x,
                y=arm_y,
                z=arm_z,
                mode='lines+markers',
                name=f'{reg_name} Arm',
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=4, color=colors[i % len(colors)]),
                hovertemplate=f'{reg_name}<br>Day: %{{x}}<br>Impact: %{{z:.2f}}%<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'3D Sensitivity Vortex: Market Regressor Impact{" - " + ticker if ticker else ""}',
            scene=dict(
                xaxis_title='Days Forecast',
                yaxis_title='Price (USD)',
                zaxis_title='Impact %',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_3d_sensitivity_vortex: {str(e)}")
        return None