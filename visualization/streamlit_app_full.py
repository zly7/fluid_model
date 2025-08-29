import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import data modules
sys.path.append(str(Path(__file__).parent.parent))

from data.processor import DataProcessor
from visualization.plotly_charts import (
    plot_time_series, plot_parameter_distribution, 
    plot_correlation_heatmap, plot_equipment_comparison,
    plot_boundary_conditions, plot_data_quality_metrics
)

def load_data_processor():
    """Load DataProcessor with proper path handling."""
    try:
        # Try to find data directory
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data"
        
        if not data_dir.exists():
            st.error(f"Data directory not found at: {data_dir}")
            return None
            
        processor = DataProcessor(str(data_dir))
        return processor
    except Exception as e:
        st.error(f"Failed to initialize DataProcessor: {e}")
        return None

@st.cache_data
def load_sample_data(sample_name: str, split: str = 'train'):
    """Load and cache sample data."""
    processor = load_data_processor()
    if processor is None:
        return None, None
    
    try:
        sample_dirs = processor.get_sample_directories(split)
        sample_dir = None
        
        for dir_path in sample_dirs:
            if dir_path.name == sample_name:
                sample_dir = dir_path
                break
        
        if sample_dir is None:
            st.error(f"Sample {sample_name} not found in {split} data")
            return None, None
        
        boundary_data, equipment_data = processor.load_sample_data(sample_dir)
        return boundary_data, equipment_data
    
    except Exception as e:
        st.error(f"Failed to load sample data: {e}")
        return None, None

@st.cache_data 
def get_available_samples(split: str = 'train'):
    """Get list of available samples."""
    processor = load_data_processor()
    if processor is None:
        return []
    
    try:
        sample_dirs = processor.get_sample_directories(split)
        return [d.name for d in sample_dirs]
    except Exception as e:
        st.error(f"Failed to get samples: {e}")
        return []

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Gas Pipeline Network Data Explorer",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 Gas Pipeline Network Data Explorer")
    st.markdown("""
    Interactive data exploration interface for gas pipeline network fluid dynamics data.
    Explore boundary conditions, equipment parameters, and time series patterns.
    """)
    
    # Sidebar for data selection
    st.sidebar.header("Data Selection")
    
    # Split selection
    split = st.sidebar.selectbox(
        "Select Dataset Split",
        ["train", "test"],
        help="Choose between training data (with targets) or test data (boundary conditions only)"
    )
    
    # Get available samples
    samples = get_available_samples(split)
    
    if not samples:
        st.error("No samples found. Please check your data directory structure.")
        return
    
    # Sample selection
    selected_sample = st.sidebar.selectbox(
        "Select Sample",
        samples,
        help="Choose a specific sample to explore"
    )
    
    # Load data
    with st.spinner(f"Loading {selected_sample} data..."):
        boundary_data, equipment_data = load_sample_data(selected_sample, split)
    
    if boundary_data is None:
        st.error("Failed to load data.")
        return
    
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Time Steps", len(boundary_data))
    
    with col2:
        st.metric("Boundary Parameters", len(boundary_data.columns) - 1)  # Exclude TIME
    
    with col3:
        if equipment_data is not None:
            st.metric("Prediction Parameters", len(equipment_data.columns) - 1)  # Exclude TIME
        else:
            st.metric("Prediction Parameters", "N/A (Test Data)")
    
    with col4:
        time_span = "24 hours"
        if 'TIME' in boundary_data.columns:
            start_time = boundary_data['TIME'].min()
            end_time = boundary_data['TIME'].max()
            time_span = f"{(end_time - start_time).total_seconds() / 3600:.1f} hours"
        st.metric("Time Span", time_span)
    
    # Display raw data samples
    with st.expander("📋 Raw Data Sample", expanded=False):
        st.subheader("Boundary Conditions (First 10 rows)")
        st.dataframe(boundary_data.head(10), width='stretch')
        
        if equipment_data is not None:
            st.subheader("Equipment Predictions (First 10 rows)")
            st.dataframe(equipment_data.head(10), width='stretch')
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕐 Time Series", "📈 Distributions", "🔗 Correlations", 
        "⚙️ Equipment Comparison", "🔍 Data Quality"
    ])
    
    with tab1:
        st.subheader("Time Series Analysis")
        
        # Boundary conditions time series
        st.write("### Boundary Conditions")
        
        # Select condition types to plot
        available_types = []
        for col in boundary_data.columns:
            if ':' in col:
                ctype = col.split(':')[1]
                if ctype not in available_types:
                    available_types.append(ctype)
        
        if available_types:
            selected_types = st.multiselect(
                "Select boundary condition types to plot:",
                available_types,
                default=available_types[:3] if len(available_types) >= 3 else available_types,
                help="Choose which types of boundary conditions to visualize"
            )
            
            if selected_types:
                fig_boundary = plot_boundary_conditions(boundary_data, selected_types)
                st.plotly_chart(fig_boundary, width='stretch')
        
        # Equipment time series (if available)
        if equipment_data is not None:
            st.write("### Equipment Parameters")
            
            # Get available equipment types
            equipment_types = set()
            for col in equipment_data.columns:
                if '_' in col and col != 'TIME':
                    eq_type = col.split('_')[0]
                    equipment_types.add(eq_type)
            
            if equipment_types:
                selected_eq_type = st.selectbox(
                    "Select equipment type:",
                    sorted(list(equipment_types)),
                    help="Choose equipment type to visualize"
                )
                
                # Get parameters for selected equipment type
                eq_cols = [col for col in equipment_data.columns 
                          if col.startswith(f"{selected_eq_type}_") and col != 'TIME']
                
                if len(eq_cols) > 10:
                    eq_cols = eq_cols[:10]  # Limit to first 10 for performance
                    st.info(f"Showing first 10 parameters for {selected_eq_type}. Total available: {len([col for col in equipment_data.columns if col.startswith(f'{selected_eq_type}_')])}")
                
                if eq_cols:
                    fig_eq = plot_time_series(
                        equipment_data, eq_cols, 
                        title=f"{selected_eq_type} Parameters Over Time"
                    )
                    st.plotly_chart(fig_eq, width='stretch')
    
    with tab2:
        st.subheader("Parameter Distributions")
        
        # Data source selection
        data_source = st.selectbox(
            "Select data source:",
            ["Boundary Conditions", "Equipment Data"] if equipment_data is not None else ["Boundary Conditions"]
        )
        
        if data_source == "Boundary Conditions":
            df_for_dist = boundary_data
        else:
            df_for_dist = equipment_data
        
        # Get numeric columns
        numeric_cols = df_for_dist.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if 'TIME' not in col.upper()]
        
        if numeric_cols:
            # Select columns for distribution analysis
            selected_dist_cols = st.multiselect(
                "Select parameters for distribution analysis:",
                numeric_cols[:20],  # Limit options for UI performance
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                help="Choose parameters to analyze distributions"
            )
            
            if selected_dist_cols:
                # Distribution plot type
                dist_type = st.selectbox(
                    "Select plot type:",
                    ["histogram", "box", "violin"],
                    help="Choose visualization type for distributions"
                )
                
                fig_dist = plot_parameter_distribution(
                    df_for_dist, selected_dist_cols, dist_type
                )
                st.plotly_chart(fig_dist, width='stretch')
                
                # Basic statistics
                st.write("### Basic Statistics")
                stats_df = df_for_dist[selected_dist_cols].describe()
                st.dataframe(stats_df, width='stretch')
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Data source selection for correlation
        data_source_corr = st.selectbox(
            "Select data source for correlation:",
            ["Boundary Conditions", "Equipment Data"] if equipment_data is not None else ["Boundary Conditions"],
            key="corr_source"
        )
        
        if data_source_corr == "Boundary Conditions":
            df_for_corr = boundary_data
        else:
            df_for_corr = equipment_data
        
        # Correlation method
        corr_method = st.selectbox(
            "Correlation method:",
            ["pearson", "spearman", "kendall"],
            help="Choose correlation calculation method"
        )
        
        # Get numeric columns for correlation
        numeric_cols_corr = df_for_corr.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_corr = [col for col in numeric_cols_corr if 'TIME' not in col.upper()]
        
        if len(numeric_cols_corr) >= 2:
            # Limit columns for performance
            max_corr_cols = st.slider(
                "Maximum columns for correlation (for performance):",
                min_value=5, max_value=min(50, len(numeric_cols_corr)), 
                value=min(20, len(numeric_cols_corr)),
                help="Limit number of columns to avoid performance issues"
            )
            
            selected_corr_cols = numeric_cols_corr[:max_corr_cols]
            
            fig_corr = plot_correlation_heatmap(
                df_for_corr, selected_corr_cols, corr_method
            )
            st.plotly_chart(fig_corr, width='stretch')
            
            # Show top correlations
            st.write("### Top Correlations")
            corr_matrix = df_for_corr[selected_corr_cols].corr(method=corr_method)
            
            # Get upper triangle correlations (avoiding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.where(mask).stack().reset_index()
            corr_values.columns = ['Variable 1', 'Variable 2', 'Correlation']
            corr_values = corr_values.sort_values('Correlation', key=abs, ascending=False)
            
            st.dataframe(corr_values.head(10), width='stretch')
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
    
    with tab4:
        st.subheader("Equipment Comparison")
        
        if equipment_data is not None:
            # Get equipment types
            equipment_types_comp = set()
            for col in equipment_data.columns:
                if '_' in col and col != 'TIME':
                    eq_type = col.split('_')[0]
                    equipment_types_comp.add(eq_type)
            
            if equipment_types_comp:
                # Select equipment types to compare
                selected_eq_types = st.multiselect(
                    "Select equipment types to compare:",
                    sorted(list(equipment_types_comp)),
                    default=list(sorted(equipment_types_comp))[:3],
                    help="Choose equipment types for comparison"
                )
                
                if selected_eq_types:
                    # Create equipment data dictionary
                    equipment_dict = {}
                    for eq_type in selected_eq_types:
                        eq_cols = ['TIME'] + [col for col in equipment_data.columns 
                                            if col.startswith(f"{eq_type}_")]
                        equipment_dict[eq_type] = equipment_data[eq_cols]
                    
                    # Select parameter to compare
                    common_params = ['p_in', 'p_out', 'q_in', 'q_out', 't_in', 't_out']
                    available_params = []
                    
                    for param in common_params:
                        for eq_type in selected_eq_types:
                            eq_cols = [col for col in equipment_data.columns 
                                     if col.startswith(f"{eq_type}_") and param in col]
                            if eq_cols:
                                available_params.append(param)
                                break
                    
                    if available_params:
                        selected_param = st.selectbox(
                            "Select parameter to compare:",
                            available_params,
                            help="Choose parameter for equipment comparison"
                        )
                        
                        sample_count = st.slider(
                            "Number of equipment units per type:",
                            min_value=1, max_value=10, value=3,
                            help="How many units of each type to show"
                        )
                        
                        fig_eq_comp = plot_equipment_comparison(
                            equipment_dict, selected_param, sample_count
                        )
                        st.plotly_chart(fig_eq_comp, width='stretch')
                    else:
                        st.warning("No common parameters found for selected equipment types.")
                else:
                    st.info("Please select at least one equipment type for comparison.")
            else:
                st.warning("No equipment data found for comparison.")
        else:
            st.warning("Equipment comparison not available for test data.")
    
    with tab5:
        st.subheader("Data Quality Analysis")
        
        # Analyze boundary data quality
        st.write("### Boundary Conditions Quality")
        fig_quality_boundary = plot_data_quality_metrics(
            boundary_data, "Boundary Conditions Data Quality"
        )
        st.plotly_chart(fig_quality_boundary, width='stretch')
        
        # Analyze equipment data quality (if available)
        if equipment_data is not None:
            st.write("### Equipment Data Quality")
            
            # Sample equipment data for quality analysis (to avoid performance issues)
            sample_cols = equipment_data.columns[:100] if len(equipment_data.columns) > 100 else equipment_data.columns
            sampled_equipment_data = equipment_data[sample_cols]
            
            fig_quality_equipment = plot_data_quality_metrics(
                sampled_equipment_data, "Equipment Data Quality (Sample)"
            )
            st.plotly_chart(fig_quality_equipment, width='stretch')
        
        # Summary statistics
        st.write("### Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Boundary Data**")
            boundary_info = {
                "Total Rows": len(boundary_data),
                "Total Columns": len(boundary_data.columns),
                "Missing Values": boundary_data.isnull().sum().sum(),
                "Complete Rows": len(boundary_data.dropna()),
                "Numeric Columns": len(boundary_data.select_dtypes(include=[np.number]).columns)
            }
            
            for key, value in boundary_info.items():
                st.metric(key, value)
        
        with col2:
            if equipment_data is not None:
                st.write("**Equipment Data**")
                equipment_info = {
                    "Total Rows": len(equipment_data),
                    "Total Columns": len(equipment_data.columns),
                    "Missing Values": equipment_data.isnull().sum().sum(),
                    "Complete Rows": len(equipment_data.dropna()),
                    "Numeric Columns": len(equipment_data.select_dtypes(include=[np.number]).columns)
                }
                
                for key, value in equipment_info.items():
                    st.metric(key, value)
            else:
                st.info("Equipment data not available (test split)")

def run_dashboard():
    """Entry point for running the dashboard."""
    main()

if __name__ == "__main__":
    run_dashboard()