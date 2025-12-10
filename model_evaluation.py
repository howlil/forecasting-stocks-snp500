"""
Modul untuk Analisis dan Evaluasi Model Forecasting.
Menyediakan fungsi-fungsi komprehensif untuk evaluasi model Prophet.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Menghitung metrics evaluasi model yang komprehensif.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Nilai aktual
    y_pred : np.ndarray
        Nilai prediksi
    
    Returns:
    --------
    dict : Dictionary berisi semua metrics evaluasi
    """
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE dengan handling untuk nilai 0
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Scaled Error (MASE) - lebih robust untuk time series
    if len(y_true) > 1:
        naive_forecast = np.roll(y_true, 1)[1:]
        naive_actual = y_true[1:]
        mae_naive = mean_absolute_error(naive_actual, naive_forecast)
        mase = mae / (mae_naive + 1e-8) if mae_naive > 0 else np.inf
    else:
        mase = np.inf
    
    # Directional Accuracy (DA) - persentase prediksi arah yang benar
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        directional_accuracy = 0.0
    
    # Mean Error (bias)
    mean_error = np.mean(y_pred - y_true)
    
    # Mean Percentage Error
    mean_pct_error = np.mean((y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100
    
    # Theil's U statistic (normalized RMSE)
    theil_u = rmse / (np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2)) + 1e-8)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'MASE': mase,
        'Directional_Accuracy': directional_accuracy,
        'Mean_Error': mean_error,
        'Mean_Pct_Error': mean_pct_error,
        'Theil_U': theil_u
    }


def evaluate_model_performance(
    model: Prophet,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Evaluasi komprehensif performa model Prophet.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    forecast_df : pd.DataFrame, optional
        Forecast results (jika None, akan di-generate)
    
    Returns:
    --------
    dict : Dictionary berisi hasil evaluasi lengkap
    """
    # Generate forecast jika belum ada
    if forecast_df is None:
        forecast_df = model.predict(test_df[['ds']])
    
    # Extract actual and predicted values
    y_true = test_df['y'].values
    y_pred = forecast_df['yhat'].values[:len(y_true)]
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # Residual analysis
    residuals = y_true - y_pred
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    # Normality test (Jarque-Bera approximation)
    from scipy import stats
    try:
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        is_normal = jb_pvalue > 0.05
    except:
        jb_stat, jb_pvalue = None, None
        is_normal = None
    
    # Autocorrelation of residuals (Ljung-Box test)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lags = min(10, len(residuals) - 1)
        if lags > 0:
            lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            if len(lb_result) > 0:
                lb_pvalue = lb_result['lb_pvalue'].values[-1]
                lb_stat = lb_result['lb_stat'].values[-1]
                has_autocorr = lb_pvalue < 0.05 if lb_pvalue is not None else None
            else:
                lb_stat, lb_pvalue = None, None
                has_autocorr = None
        else:
            lb_stat, lb_pvalue = None, None
            has_autocorr = None
    except Exception:
        lb_stat, lb_pvalue = None, None
        has_autocorr = None
    
    # Forecast accuracy by horizon (jika data cukup panjang)
    horizon_metrics = {}
    if len(test_df) >= 30:
        # Split test set menjadi beberapa horizon
        horizons = [7, 14, 30, min(90, len(test_df))]
        for horizon in horizons:
            if len(test_df) >= horizon:
                y_true_h = y_true[:horizon]
                y_pred_h = y_pred[:horizon]
                horizon_metrics[f'RMSE_{horizon}d'] = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
                horizon_metrics[f'MAPE_{horizon}d'] = np.mean(np.abs((y_true_h - y_pred_h) / (np.abs(y_true_h) + 1e-8))) * 100
    
    return {
        'metrics': metrics,
        'residuals': {
            'mean': residual_mean,
            'std': residual_std,
            'min': np.min(residuals),
            'max': np.max(residuals)
        },
        'diagnostics': {
            'is_normal': is_normal,
            'jb_statistic': jb_stat,
            'jb_pvalue': jb_pvalue,
            'has_autocorrelation': has_autocorr,
            'lb_pvalue': lb_pvalue if lb_pvalue is not None else None
        },
        'horizon_metrics': horizon_metrics,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'forecast_df': forecast_df
    }


def plot_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis"
) -> go.Figure:
    """
    Visualisasi analisis residual untuk evaluasi model.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Nilai aktual
    y_pred : np.ndarray
        Nilai prediksi
    title : str
        Title untuk plot
    
    Returns:
    --------
    go.Figure : Plotly figure dengan 4 subplots
    """
    residuals = y_true - y_pred
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Residuals Over Time',
            'Residual Distribution',
            'Q-Q Plot (Normality Check)',
            'Actual vs Predicted'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Residuals over time
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='lines+markers',
            name='Residuals',
            line=dict(color='blue', width=1),
            marker=dict(size=3)
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. Residual distribution (histogram)
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Residual Distribution',
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # Add normal distribution overlay
    from scipy import stats
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, np.mean(residuals), np.std(residuals))
    y_norm = y_norm * len(residuals) * (residuals.max() - residuals.min()) / 30
    
    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=2
    )
    
    # 3. Q-Q Plot
    from scipy import stats
    qq_data = stats.probplot(residuals, dist="norm")
    theoretical_quantiles = qq_data[0][0]
    sample_quantiles = qq_data[0][1]
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            name='Q-Q Points',
            marker=dict(size=4, color='blue')
        ),
        row=2, col=1
    )
    
    # Add diagonal line
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Normal',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    # 4. Actual vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=4, color='green', opacity=0.6)
        ),
        row=2, col=2
    )
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time Index", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    
    fig.update_xaxes(title_text="Actual", row=2, col=2)
    fig.update_yaxes(title_text="Predicted", row=2, col=2)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=800,
        showlegend=True
    )
    
    return fig


def plot_forecast_accuracy_by_horizon(
    evaluation_results: Dict,
    title: str = "Forecast Accuracy by Horizon"
) -> go.Figure:
    """
    Visualisasi akurasi forecast berdasarkan horizon waktu.
    
    Parameters:
    -----------
    evaluation_results : dict
        Hasil dari evaluate_model_performance
    title : str
        Title untuk plot
    
    Returns:
    --------
    go.Figure : Plotly figure
    """
    horizon_metrics = evaluation_results.get('horizon_metrics', {})
    
    if not horizon_metrics:
        # Create empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for horizon analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract horizon and metrics
    horizons = []
    rmse_values = []
    mape_values = []
    
    for key, value in horizon_metrics.items():
        if 'RMSE' in key:
            horizon = int(key.split('_')[1].replace('d', ''))
            horizons.append(horizon)
            rmse_values.append(value)
        elif 'MAPE' in key:
            mape_values.append(value)
    
    # Sort by horizon
    sorted_data = sorted(zip(horizons, rmse_values, mape_values))
    horizons, rmse_values, mape_values = zip(*sorted_data) if sorted_data else ([], [], [])
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE by Horizon', 'MAPE by Horizon'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # RMSE plot
    fig.add_trace(
        go.Bar(
            x=list(horizons),
            y=list(rmse_values),
            name='RMSE',
            marker_color='lightblue',
            text=[f'{v:.2f}' for v in rmse_values],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # MAPE plot
    fig.add_trace(
        go.Bar(
            x=list(horizons),
            y=list(mape_values),
            name='MAPE (%)',
            marker_color='lightcoral',
            text=[f'{v:.2f}%' for v in mape_values],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Forecast Horizon (days)", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    
    fig.update_xaxes(title_text="Forecast Horizon (days)", row=1, col=2)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=500,
        showlegend=False
    )
    
    return fig


def generate_model_evaluation_report(
    evaluation_results: Dict,
    ticker: str = "Unknown"
) -> str:
    """
    Generate text report untuk evaluasi model.
    
    Parameters:
    -----------
    evaluation_results : dict
        Hasil dari evaluate_model_performance
    ticker : str
        Ticker symbol
    
    Returns:
    --------
    str : Text report
    """
    metrics = evaluation_results.get('metrics', {})
    diagnostics = evaluation_results.get('diagnostics', {})
    residuals = evaluation_results.get('residuals', {})
    
    report = f"""
# Model Evaluation Report - {ticker}

## 1. Model Performance Metrics

### Basic Metrics:
- **RMSE (Root Mean Squared Error)**: {metrics.get('RMSE', 0):.4f}
- **MAE (Mean Absolute Error)**: {metrics.get('MAE', 0):.4f}
- **MAPE (Mean Absolute Percentage Error)**: {metrics.get('MAPE', 0):.2f}%
- **R² (Coefficient of Determination)**: {metrics.get('R2', 0):.4f}

### Advanced Metrics:
- **MASE (Mean Absolute Scaled Error)**: {metrics.get('MASE', 0):.4f}
  - MASE < 1: Model lebih baik dari naive forecast
  - MASE = 1: Model sama baiknya dengan naive forecast
  - MASE > 1: Model lebih buruk dari naive forecast

- **Directional Accuracy**: {metrics.get('Directional_Accuracy', 0):.2f}%
  - Persentase prediksi arah pergerakan yang benar

- **Theil's U Statistic**: {metrics.get('Theil_U', 0):.4f}
  - U < 1: Model lebih baik dari naive forecast
  - U = 1: Model sama baiknya dengan naive forecast
  - U > 1: Model lebih buruk dari naive forecast

## 2. Residual Analysis

### Residual Statistics:
- **Mean**: {residuals.get('mean', 0):.4f} (ideal: 0)
- **Standard Deviation**: {residuals.get('std', 0):.4f}
- **Min**: {residuals.get('min', 0):.4f}
- **Max**: {residuals.get('max', 0):.4f}

### Diagnostic Tests:
- **Normality Test (Jarque-Bera)**:
  - P-value: {diagnostics.get('jb_pvalue', 'N/A')}
  - Residuals are {'normally distributed' if diagnostics.get('is_normal') else 'NOT normally distributed'}

- **Autocorrelation Test (Ljung-Box)**:
  - P-value: {diagnostics.get('lb_pvalue', 'N/A')}
  - Residuals {'have' if diagnostics.get('has_autocorrelation') else 'do NOT have'} significant autocorrelation

## 3. Model Interpretation

### Model Quality Assessment:
"""
    
    # Add interpretation
    r2 = metrics.get('R2', 0)
    mape = metrics.get('MAPE', 0)
    mase = metrics.get('MASE', 0)
    
    if r2 > 0.9:
        report += "- **R² > 0.9**: Excellent model fit\n"
    elif r2 > 0.7:
        report += "- **R² > 0.7**: Good model fit\n"
    elif r2 > 0.5:
        report += "- **R² > 0.5**: Moderate model fit\n"
    else:
        report += "- **R² < 0.5**: Poor model fit - model needs improvement\n"
    
    if mape < 5:
        report += "- **MAPE < 5%**: Excellent forecast accuracy\n"
    elif mape < 10:
        report += "- **MAPE < 10%**: Good forecast accuracy\n"
    elif mape < 20:
        report += "- **MAPE < 20%**: Acceptable forecast accuracy\n"
    else:
        report += "- **MAPE > 20%**: Poor forecast accuracy - model needs improvement\n"
    
    if mase < 1:
        report += "- **MASE < 1**: Model performs better than naive forecast\n"
    elif mase == 1:
        report += "- **MASE = 1**: Model performs as well as naive forecast\n"
    else:
        report += "- **MASE > 1**: Model performs worse than naive forecast - needs improvement\n"
    
    report += f"""
## 4. Recommendations

"""
    
    # Add recommendations based on metrics
    if r2 < 0.5 or mape > 20:
        report += "- Consider adding more features or regressors\n"
        report += "- Try different model parameters (changepoint_prior_scale, seasonality_prior_scale)\n"
        report += "- Check for data quality issues or outliers\n"
    
    if diagnostics.get('has_autocorrelation'):
        report += "- Residuals show autocorrelation - consider adding lag features or using different model\n"
    
    if not diagnostics.get('is_normal'):
        report += "- Residuals are not normally distributed - consider transformation or robust model\n"
    
    if metrics.get('Directional_Accuracy', 0) < 50:
        report += "- Directional accuracy < 50% - model struggles to predict direction of movement\n"
    
    report += f"""
## 5. Data Summary

- **Training Size**: {evaluation_results.get('train_size', 0):,} samples
- **Test Size**: {evaluation_results.get('test_size', 0):,} samples
- **Train/Test Split**: {evaluation_results.get('train_size', 0) / (evaluation_results.get('train_size', 1) + evaluation_results.get('test_size', 1)) * 100:.1f}% / {evaluation_results.get('test_size', 0) / (evaluation_results.get('train_size', 1) + evaluation_results.get('test_size', 1)) * 100:.1f}%

---
*Report generated automatically by FinScope Model Evaluation Module*
"""
    
    return report

