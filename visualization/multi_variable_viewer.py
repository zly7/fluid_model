import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple
import uuid

# Add parent directory to path to import data modules
sys.path.append(str(Path(__file__).parent.parent))

from data.processor import DataProcessor

@st.cache_data
def get_available_datasets():
    """Get list of available datasets from train directory."""
    try:
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data"
        
        if not data_dir.exists():
            st.error(f"Data directory not found at: {data_dir}")
            return []
        
        processor = DataProcessor(str(data_dir))
        sample_dirs = processor.get_sample_directories('train')
        return [d.name for d in sample_dirs]
    except Exception as e:
        st.error(f"Failed to get datasets: {e}")
        return []

@st.cache_data
def get_available_variables(dataset_name: str) -> Tuple[List[str], List[str]]:
    """Get available variables from a specific dataset.
    
    Returns:
        Tuple of (boundary_variables, equipment_variables)
    """
    try:
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data"
        processor = DataProcessor(str(data_dir))
        
        sample_dirs = processor.get_sample_directories('train')
        sample_dir = None
        
        for dir_path in sample_dirs:
            if dir_path.name == dataset_name:
                sample_dir = dir_path
                break
        
        if sample_dir is None:
            return [], []
        
        boundary_data, equipment_data = processor.load_sample_data(sample_dir)
        
        # Get boundary variables (exclude TIME)
        boundary_vars = [col for col in boundary_data.columns if col != 'TIME']
        
        # Get equipment variables (exclude TIME)
        equipment_vars = []
        if equipment_data is not None:
            equipment_vars = [col for col in equipment_data.columns if col != 'TIME']
        
        return boundary_vars, equipment_vars
        
    except Exception as e:
        st.error(f"Failed to get variables for {dataset_name}: {e}")
        return [], []

@st.cache_data
def load_dataset_data(dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load data for a specific dataset."""
    try:
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data"
        processor = DataProcessor(str(data_dir))
        
        sample_dirs = processor.get_sample_directories('train')
        sample_dir = None
        
        for dir_path in sample_dirs:
            if dir_path.name == dataset_name:
                sample_dir = dir_path
                break
        
        if sample_dir is None:
            return None, None
        
        boundary_data, equipment_data = processor.load_sample_data(sample_dir)
        return boundary_data, equipment_data
        
    except Exception as e:
        st.error(f"Failed to load dataset {dataset_name}: {e}")
        return None, None

def create_multi_variable_plot(variable_selections: List[Dict]) -> go.Figure:
    """Create multi-variable time series plot."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    color_idx = 0
    
    for selection in variable_selections:
        if not selection['dataset'] or not selection['variable']:
            continue
            
        try:
            boundary_data, equipment_data = load_dataset_data(selection['dataset'])
            
            # Determine which data source contains the variable
            data_source = None
            if boundary_data is not None and selection['variable'] in boundary_data.columns:
                data_source = boundary_data
            elif equipment_data is not None and selection['variable'] in equipment_data.columns:
                data_source = equipment_data
            
            if data_source is None:
                st.warning(f"Variable {selection['variable']} not found in dataset {selection['dataset']}")
                continue
            
            # Add trace to the plot
            color = colors[color_idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=data_source['TIME'] if 'TIME' in data_source.columns else data_source.index,
                    y=data_source[selection['variable']],
                    mode='lines',
                    name=f"{selection['dataset']}: {selection['variable']}",
                    line=dict(color=color, width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y:.4f}<br>' +
                                '<extra></extra>'
                )
            )
            color_idx += 1
            
        except Exception as e:
            st.error(f"Error loading data for {selection['dataset']}.{selection['variable']}: {e}")
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Multi-Variable Time Series Comparison",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title="Time",
        yaxis_title="Value",
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Add range selector if we have time data
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

def variable_selection_row(row_id: str, available_datasets: List[str]):
    """Create a single variable selection row."""
    col1, col2, col3, col4 = st.columns([3, 4, 2, 1])
    
    with col1:
        selected_dataset = st.selectbox(
            "Dataset",
            options=[""] + available_datasets,
            key=f"dataset_{row_id}",
            label_visibility="collapsed"
        )
    
    with col2:
        variables = []
        if selected_dataset:
            boundary_vars, equipment_vars = get_available_variables(selected_dataset)
            variables = boundary_vars + equipment_vars
        
        selected_variable = st.selectbox(
            "Variable",
            options=[""] + variables,
            key=f"variable_{row_id}",
            label_visibility="collapsed"
        )
    
    with col3:
        # Show data source type
        if selected_dataset and selected_variable:
            boundary_vars, equipment_vars = get_available_variables(selected_dataset)
            if selected_variable in boundary_vars:
                st.text("Boundary")
            elif selected_variable in equipment_vars:
                st.text("Equipment")
    
    with col4:
        remove_clicked = st.button("🗑️", key=f"remove_{row_id}", help="Remove this variable")
    
    return {
        'dataset': selected_dataset,
        'variable': selected_variable,
        'remove_clicked': remove_clicked,
        'row_id': row_id
    }

def run_multi_variable_viewer():
    """Main application for multi-variable time series viewer."""
    
    st.set_page_config(
        page_title="Gas Pipeline Multi-Variable Viewer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Gas Pipeline Multi-Variable Time Series Viewer")
    st.markdown("""
    Compare multiple variables across different datasets in a single time series plot.
    Add variable selection rows to compare parameters from different pipeline samples.
    """)
    
    # Initialize session state
    if 'variable_rows' not in st.session_state:
        st.session_state.variable_rows = [str(uuid.uuid4())]
    
    # Get available datasets
    available_datasets = get_available_datasets()
    
    if not available_datasets:
        st.error("No datasets found. Please check your data directory structure.")
        return
    
    # Sidebar for variable selection
    st.sidebar.header("Variable Selection")
    st.sidebar.markdown("Select datasets and variables to compare:")
    
    # Add new variable row button
    if st.sidebar.button("➕ Add Variable", key="add_variable"):
        st.session_state.variable_rows.append(str(uuid.uuid4()))
        st.rerun()
    
    # Create variable selection rows
    variable_selections = []
    rows_to_remove = []
    
    for row_id in st.session_state.variable_rows:
        st.sidebar.markdown(f"**Variable {len(variable_selections) + 1}:**")
        
        with st.sidebar.container():
            selection = variable_selection_row(row_id, available_datasets)
            
            if selection['remove_clicked'] and len(st.session_state.variable_rows) > 1:
                rows_to_remove.append(row_id)
            else:
                variable_selections.append(selection)
        
        st.sidebar.markdown("---")
    
    # Remove rows that were marked for removal
    for row_id in rows_to_remove:
        st.session_state.variable_rows.remove(row_id)
        st.rerun()
    
    # Filter valid selections
    valid_selections = [sel for sel in variable_selections 
                       if sel['dataset'] and sel['variable']]
    
    # Main content area
    if valid_selections:
        st.subheader(f"Comparing {len(valid_selections)} Variables")
        
        # Show selected variables info
        with st.expander("📋 Selected Variables", expanded=False):
            for i, sel in enumerate(valid_selections, 1):
                boundary_vars, equipment_vars = get_available_variables(sel['dataset'])
                var_type = "Boundary" if sel['variable'] in boundary_vars else "Equipment"
                st.write(f"{i}. **{sel['dataset']}**: {sel['variable']} ({var_type})")
        
        # Create and display the plot
        try:
            fig = create_multi_variable_plot(valid_selections)
            st.plotly_chart(fig, width='stretch')
            
            # Display statistics
            st.subheader("📊 Variable Statistics")
            
            stats_data = []
            for sel in valid_selections:
                boundary_data, equipment_data = load_dataset_data(sel['dataset'])
                
                data_source = None
                if boundary_data is not None and sel['variable'] in boundary_data.columns:
                    data_source = boundary_data
                elif equipment_data is not None and sel['variable'] in equipment_data.columns:
                    data_source = equipment_data
                
                if data_source is not None:
                    var_data = data_source[sel['variable']]
                    stats_data.append({
                        'Dataset': sel['dataset'],
                        'Variable': sel['variable'],
                        'Mean': var_data.mean(),
                        'Std': var_data.std(),
                        'Min': var_data.min(),
                        'Max': var_data.max(),
                        'Count': len(var_data)
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, width='stretch')
            
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    else:
        st.info("👆 Select datasets and variables from the sidebar to start comparing time series data.")
        
        # Show available datasets info
        st.subheader("📁 Available Datasets")
        st.write(f"Found {len(available_datasets)} datasets:")
        
        for i, dataset in enumerate(available_datasets, 1):
            with st.expander(f"{i}. {dataset}", expanded=False):
                boundary_vars, equipment_vars = get_available_variables(dataset)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Boundary Variables ({len(boundary_vars)}):**")
                    if boundary_vars:
                        for var in boundary_vars[:10]:  # Show first 10
                            st.write(f"- {var}")
                        if len(boundary_vars) > 10:
                            st.write(f"... and {len(boundary_vars) - 10} more")
                
                with col2:
                    st.write(f"**Equipment Variables ({len(equipment_vars)}):**")
                    if equipment_vars:
                        for var in equipment_vars[:10]:  # Show first 10
                            st.write(f"- {var}")
                        if len(equipment_vars) > 10:
                            st.write(f"... and {len(equipment_vars) - 10} more")

if __name__ == "__main__":
    run_multi_variable_viewer()