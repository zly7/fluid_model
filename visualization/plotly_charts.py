import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import seaborn as sns

def plot_time_series(df: pd.DataFrame, 
                    columns: List[str], 
                    title: str = "Time Series Plot",
                    height: int = 600,
                    show_legend: bool = True) -> go.Figure:
    """
    Create interactive time series line plots with zoom capabilities.
    
    Args:
        df: DataFrame with TIME column and data columns
        columns: List of column names to plot
        title: Plot title
        height: Plot height in pixels
        show_legend: Whether to show legend
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Define color palette
    colors = px.colors.qualitative.Set3
    
    for i, col in enumerate(columns):
        if col in df.columns and col != 'TIME':
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=df['TIME'] if 'TIME' in df.columns else df.index,
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(color=color, width=2),
                    hovertemplate=f'<b>{col}</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y:.4f}<br>' +
                                '<extra></extra>'
                )
            )
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title="Time",
        yaxis_title="Value",
        height=height,
        showlegend=show_legend,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Add range selector
    if 'TIME' in df.columns:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=12, label="12h", step="hour", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
    
    return fig

def plot_parameter_distribution(df: pd.DataFrame, 
                               columns: List[str],
                               plot_type: str = "histogram",
                               title: str = "Parameter Distribution") -> go.Figure:
    """
    Create parameter distribution histograms or box plots.
    
    Args:
        df: DataFrame with data
        columns: List of columns to plot
        plot_type: "histogram", "box", or "violin"
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    if plot_type == "histogram":
        fig = make_subplots(
            rows=len(columns), cols=1,
            subplot_titles=columns,
            vertical_spacing=0.05
        )
        
        for i, col in enumerate(columns):
            if col in df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        name=col,
                        showlegend=False,
                        nbinsx=50,
                        opacity=0.7
                    ),
                    row=i+1, col=1
                )
    
    elif plot_type == "box":
        fig = go.Figure()
        
        for col in columns:
            if col in df.columns:
                fig.add_trace(
                    go.Box(
                        y=df[col],
                        name=col,
                        boxpoints='outliers'
                    )
                )
    
    elif plot_type == "violin":
        fig = go.Figure()
        
        for col in columns:
            if col in df.columns:
                fig.add_trace(
                    go.Violin(
                        y=df[col],
                        name=col,
                        box_visible=True,
                        line_color='black'
                    )
                )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        height=600 if plot_type == "histogram" else 400,
        template='plotly_white'
    )
    
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, 
                           columns: Optional[List[str]] = None,
                           method: str = 'pearson',
                           title: str = "Correlation Heatmap") -> go.Figure:
    """
    Create correlation heatmap between parameters.
    
    Args:
        df: DataFrame with data
        columns: List of columns to include (if None, use all numeric columns)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    # Select columns
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove TIME-related columns
        numeric_cols = [col for col in numeric_cols if 'TIME' not in col.upper()]
        columns = numeric_cols[:50]  # Limit to 50 for readability
    
    # Calculate correlation matrix
    corr_df = df[columns].corr(method=method)
    
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>' +
                         f'{method.title()} correlation: %{{z:.3f}}<br>' +
                         '<extra></extra>'
        )
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        width=800,
        height=800,
        template='plotly_white'
    )
    
    return fig

def plot_equipment_comparison(equipment_data: Dict[str, pd.DataFrame],
                            parameter: str = "p_in",
                            sample_equipment: int = 5,
                            title: str = "Equipment Parameter Comparison") -> go.Figure:
    """
    Compare parameter values across different equipment types.
    
    Args:
        equipment_data: Dictionary mapping equipment type to DataFrame
        parameter: Parameter to compare (e.g., "p_in", "q_out", "t_in")
        sample_equipment: Number of equipment units to sample per type
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, (eq_type, df) in enumerate(equipment_data.items()):
        # Find columns containing the parameter
        param_cols = [col for col in df.columns if parameter in col and col != 'TIME']
        
        # Sample columns to avoid overcrowding
        if len(param_cols) > sample_equipment:
            param_cols = param_cols[:sample_equipment]
        
        for j, col in enumerate(param_cols):
            color = colors[i % len(colors)]
            
            # Add some transparency for overlapping lines
            fig.add_trace(
                go.Scatter(
                    x=df['TIME'] if 'TIME' in df.columns else df.index,
                    y=df[col],
                    mode='lines',
                    name=f"{eq_type}_{col}",
                    line=dict(color=color, width=1.5),
                    opacity=0.7,
                    legendgroup=eq_type,
                    legendgrouptitle_text=eq_type,
                    hovertemplate=f'<b>{col}</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y:.4f}<br>' +
                                '<extra></extra>'
                )
            )
    
    fig.update_layout(
        title=dict(text=f"{title} - {parameter}", x=0.5, font=dict(size=18)),
        xaxis_title="Time",
        yaxis_title=f"{parameter} Value",
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def plot_boundary_conditions(boundary_df: pd.DataFrame,
                           condition_types: List[str] = None,
                           title: str = "Boundary Conditions Over Time") -> go.Figure:
    """
    Plot boundary conditions time series.
    
    Args:
        boundary_df: DataFrame with boundary conditions
        condition_types: List of condition types to plot (e.g., ['SNQ', 'SP', 'ST'])
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    if condition_types is None:
        condition_types = ['SNQ', 'SP', 'ST', 'SPD', 'FR']
    
    # Create subplots for different condition types
    fig = make_subplots(
        rows=len(condition_types), cols=1,
        subplot_titles=[f"{ctype} Parameters" for ctype in condition_types],
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    colors = px.colors.qualitative.Pastel
    
    for i, ctype in enumerate(condition_types):
        # Find columns of this condition type
        type_cols = [col for col in boundary_df.columns 
                    if f':{ctype}' in col and col != 'TIME']
        
        # Sample columns to avoid overcrowding (max 10 per subplot)
        if len(type_cols) > 10:
            type_cols = type_cols[:10]
        
        for j, col in enumerate(type_cols):
            color = colors[j % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=boundary_df['TIME'] if 'TIME' in boundary_df.columns else boundary_df.index,
                    y=boundary_df[col],
                    mode='lines',
                    name=col,
                    line=dict(color=color, width=1.5),
                    showlegend=(i == 0),  # Only show legend for first subplot
                    hovertemplate=f'<b>{col}</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y:.4f}<br>' +
                                '<extra></extra>'
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        height=200 * len(condition_types),
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Update x-axis for last subplot only
    fig.update_xaxes(title_text="Time", row=len(condition_types), col=1)
    
    return fig

def plot_data_quality_metrics(df: pd.DataFrame, 
                             title: str = "Data Quality Metrics") -> go.Figure:
    """
    Create data quality visualization showing missing values, outliers, etc.
    
    Args:
        df: DataFrame to analyze
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    # Calculate metrics
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    
    # Get numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create subplot structure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Missing Values (%)", "Data Distribution", 
                       "Value Ranges", "Outlier Detection"),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # Missing values plot
    if len(missing_pct) > 0 and missing_pct.max() > 0:
        top_missing = missing_pct.head(20)  # Top 20 columns with missing values
        fig.add_trace(
            go.Bar(x=top_missing.index, y=top_missing.values, name="Missing %"),
            row=1, col=1
        )
    
    # Data distribution (sample of numeric columns)
    if len(numeric_cols) > 0:
        sample_col = numeric_cols[0]
        fig.add_trace(
            go.Histogram(x=df[sample_col], name=f"Dist: {sample_col}", nbinsx=50),
            row=1, col=2
        )
    
    # Value ranges (box plot of first few numeric columns)
    sample_cols = numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
    for col in sample_cols:
        fig.add_trace(
            go.Box(y=df[col], name=col, showlegend=False),
            row=2, col=1
        )
    
    # Outlier scatter (Z-score based)
    if len(numeric_cols) >= 2:
        from scipy import stats
        col1, col2 = numeric_cols[0], numeric_cols[1]
        
        # Calculate Z-scores
        z_scores_1 = np.abs(stats.zscore(df[col1].dropna()))
        z_scores_2 = np.abs(stats.zscore(df[col2].dropna()))
        
        # Identify outliers (Z-score > 3)
        outliers = (z_scores_1 > 3) | (z_scores_2 > 3)
        
        fig.add_trace(
            go.Scatter(
                x=df[col1], y=df[col2],
                mode='markers',
                marker=dict(
                    color=outliers,
                    colorscale='Viridis',
                    size=6,
                    colorbar=dict(title="Outlier")
                ),
                name="Outliers",
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        height=800,
        template='plotly_white'
    )
    
    return fig