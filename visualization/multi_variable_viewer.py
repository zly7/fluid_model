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
import random
import re
from datetime import datetime, timedelta

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
def get_available_variables(dataset_name: str, filter_pattern: str = None) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Get available variables from a specific dataset with optional filtering.
    
    Returns:
        Tuple of (boundary_variables, equipment_variables, equipment_categorized)
    """
    try:
        boundary_data, equipment_data = load_dataset_data(dataset_name)
        
        # Get boundary variables (exclude TIME)
        boundary_vars = []
        if boundary_data is not None:
            boundary_vars = [col for col in boundary_data.columns if col != 'TIME']
        
        # Get equipment variables (exclude TIME) and categorize by equipment type
        equipment_vars = []
        equipment_categorized = {
            'B (球阀/Ball Valves)': [],
            'C (压缩机/Compressors)': [],
            'H (管段/Pipeline Segments)': [],
            'N (节点/Nodes)': [],
            'P (管道/Pipelines)': [],
            'R (调节阀/Control Valves)': [],
            'T (气源/Gas Sources)': [],
            'E (分输点/Distribution Points)': [],
            'Other': []
        }
        
        if equipment_data is not None:
            equipment_vars = [col for col in equipment_data.columns if col != 'TIME']
            
            # Categorize variables by equipment type based on naming pattern
            for var in equipment_vars:
                if var.startswith('B_'):
                    equipment_categorized['B (球阀/Ball Valves)'].append(var)
                elif var.startswith('C_'):
                    equipment_categorized['C (压缩机/Compressors)'].append(var)
                elif var.startswith('H_'):
                    equipment_categorized['H (管段/Pipeline Segments)'].append(var)
                elif var.startswith('N_'):
                    equipment_categorized['N (节点/Nodes)'].append(var)
                elif var.startswith('P_'):
                    equipment_categorized['P (管道/Pipelines)'].append(var)
                elif var.startswith('R_'):
                    equipment_categorized['R (调节阀/Control Valves)'].append(var)
                elif var.startswith('T_'):
                    equipment_categorized['T (气源/Gas Sources)'].append(var)
                elif var.startswith('E_'):
                    equipment_categorized['E (分输点/Distribution Points)'].append(var)
                else:
                    equipment_categorized['Other'].append(var)
        
        # Apply regex filter if provided
        if filter_pattern:
            try:
                regex_pattern = re.compile(filter_pattern, re.IGNORECASE)
                boundary_vars = [var for var in boundary_vars if regex_pattern.search(var)]
                equipment_vars = [var for var in equipment_vars if regex_pattern.search(var)]
                
                # Filter categorized equipment variables
                for category in equipment_categorized:
                    equipment_categorized[category] = [
                        var for var in equipment_categorized[category] 
                        if regex_pattern.search(var)
                    ]
            except re.error:
                st.warning(f"Invalid regex pattern: {filter_pattern}")
        
        # Sort variables within each category
        boundary_vars.sort()
        equipment_vars.sort()
        for category in equipment_categorized:
            equipment_categorized[category].sort()
        
        return boundary_vars, equipment_vars, equipment_categorized
        
    except Exception as e:
        st.error(f"Failed to get variables for {dataset_name}: {e}")
        return [], [], {}

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
        
        # Load boundary data
        boundary_file = sample_dir / "Boundary.csv"
        boundary_data = None
        if boundary_file.exists():
            boundary_data = pd.read_csv(boundary_file)
            # Parse TIME column properly
            boundary_data['TIME'] = pd.to_datetime(boundary_data['TIME'])
        
        # Load and combine all equipment data
        equipment_files = ['B.csv', 'C.csv', 'H.csv', 'N.csv', 'P.csv', 'R.csv', 'T&E.csv']
        equipment_dfs = []
        
        for eq_file in equipment_files:
            file_path = sample_dir / eq_file
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Parse TIME column properly
                df['TIME'] = pd.to_datetime(df['TIME'])
                equipment_dfs.append(df)
        
        # Combine all equipment data using pd.concat to avoid fragmentation
        equipment_data = None
        if equipment_dfs:
            # Get TIME from first dataframe
            base_time = equipment_dfs[0][['TIME']].copy()
            
            # Collect all non-TIME columns from all equipment files
            all_columns = []
            for df in equipment_dfs:
                # Select all columns except TIME
                non_time_cols = [col for col in df.columns if col != 'TIME']
                if non_time_cols:
                    all_columns.append(df[non_time_cols])
            
            # Concatenate all columns at once to avoid fragmentation warning
            if all_columns:
                equipment_data = pd.concat([base_time] + all_columns, axis=1)
        
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
    
    # Add fixed time window for 2025-01-01 (0:00 to 24:00)
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
            type="date",
            range=["2025-01-01 00:00:00", "2025-01-01 23:59:59"]
        )
    )
    
    return fig

def variable_selection_row(row_id: str, available_datasets: List[str], filter_pattern: str = None):
    """Create a single variable selection row with enhanced categorization."""
    # Initialize session state for this row if not exists
    if 'dataset_selections' not in st.session_state:
        st.session_state.dataset_selections = {}
    if 'variable_selections' not in st.session_state:
        st.session_state.variable_selections = {}
    
    # Get current selections from session state
    current_dataset = st.session_state.dataset_selections.get(row_id, "")
    current_variable = st.session_state.variable_selections.get(row_id, "")
    
    col1, col2, col3, col4 = st.columns([3, 4, 2, 1])
    
    with col1:
        # Calculate index for current dataset selection
        dataset_options = [""] + available_datasets
        dataset_index = 0
        if current_dataset and current_dataset in dataset_options:
            dataset_index = dataset_options.index(current_dataset)
        
        selected_dataset = st.selectbox(
            "数据集",
            options=dataset_options,
            index=dataset_index,
            key=f"dataset_{row_id}",
            label_visibility="collapsed"
        )
        
        # Update session state when selection changes
        if selected_dataset != current_dataset:
            st.session_state.dataset_selections[row_id] = selected_dataset
            # Reset variable selection when dataset changes
            st.session_state.variable_selections[row_id] = ""
    
    with col2:
        variables = []
        equipment_categorized = {}
        if selected_dataset:
            boundary_vars, equipment_vars, equipment_categorized = get_available_variables(
                selected_dataset, filter_pattern
            )
            variables = boundary_vars + equipment_vars
        
        # Create categorized options for better user experience
        if variables:
            # Group variables by category for display
            variable_options = [""]
            
            # Add boundary variables if any
            if boundary_vars:
                variable_options.append("--- 边界条件变量 ---")
                variable_options.extend(boundary_vars)
            
            # Add equipment variables by category if any
            if equipment_vars:
                variable_options.append("--- 设备参数变量 ---")
                for category, vars_in_category in equipment_categorized.items():
                    if vars_in_category:
                        variable_options.append(f"-- {category} --")
                        variable_options.extend(vars_in_category)
        else:
            variable_options = [""]
        
        # Calculate index for current variable selection
        display_options = variable_options
        variable_index = 0
        
        # Find the correct index, handling separators properly
        if current_variable and current_variable in variable_options:
            variable_index = variable_options.index(current_variable)
        elif selected_dataset != current_dataset:
            # If dataset changed, find first selectable option
            for i, option in enumerate(variable_options):
                if option and not option.startswith("-"):
                    variable_index = i
                    break
        
        # Ensure index is within bounds and valid
        if variable_index >= len(display_options):
            variable_index = 0
        
        # Ensure index corresponds to a valid option
        if variable_index < len(display_options) and display_options[variable_index].startswith("-"):
            # If we're pointing to a separator, find next valid option
            for i in range(variable_index + 1, len(display_options)):
                if display_options[i] and not display_options[i].startswith("-"):
                    variable_index = i
                    break
            else:
                # If no valid option found after, search from beginning
                for i in range(len(display_options)):
                    if display_options[i] and not display_options[i].startswith("-"):
                        variable_index = i
                        break
        
        selected_variable = st.selectbox(
            "变量",
            options=display_options,
            index=int(variable_index),
            key=f"variable_{row_id}",
            label_visibility="collapsed",
            disabled=False
        )
        
        # Ensure selected variable is valid (not a separator)
        if selected_variable and selected_variable.startswith("-"):
            selected_variable = ""
        
        # Update session state when selection changes
        if selected_variable != current_variable:
            st.session_state.variable_selections[row_id] = selected_variable
    
    with col3:
        # Show data source type and equipment category
        if selected_dataset and selected_variable:
            boundary_vars, equipment_vars, equipment_categorized = get_available_variables(selected_dataset)
            if selected_variable in boundary_vars:
                st.text("边界条件")
            elif selected_variable in equipment_vars:
                # Determine which equipment category this variable belongs to
                equipment_type = "设备参数"
                for category, vars_in_category in equipment_categorized.items():
                    if selected_variable in vars_in_category:
                        # Extract the short equipment type (first character)
                        equipment_type = category.split()[0]
                        break
                st.text(equipment_type)
    
    with col4:
        remove_clicked = st.button("🗑️", key=f"remove_{row_id}", help="删除此变量")
    
    return {
        'dataset': selected_dataset,
        'variable': selected_variable,
        'remove_clicked': remove_clicked,
        'row_id': row_id
    }

def expand_wildcard_pattern(dataset_name: str, pattern: str) -> List[str]:
    """Expand wildcard pattern to matching variable names."""
    if '*' not in pattern:
        return [pattern] if pattern else []
    
    # Convert wildcard to regex
    regex_pattern = pattern.replace('*', '.*')
    
    try:
        boundary_vars, equipment_vars, _ = get_available_variables(dataset_name, regex_pattern)
        return boundary_vars + equipment_vars
    except Exception as e:
        st.error(f"Error expanding pattern '{pattern}': {e}")
        return []

def add_multiple_variables(dataset_name: str, variables: List[str]):
    """Add multiple variables as new rows."""
    for variable in variables:
        new_row_id = str(uuid.uuid4())
        st.session_state.variable_rows.append(new_row_id)
        st.session_state.dataset_selections[new_row_id] = dataset_name
        st.session_state.variable_selections[new_row_id] = variable

def run_multi_variable_viewer():
    """Main application for multi-variable time series viewer."""
    
    st.set_page_config(
        page_title="Gas Pipeline Multi-Variable Viewer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 气体管网多变量时间序列可视化器")
    st.markdown("""
    用于气体管道网络仿真数据的多变量对比可视化工具。
    支持按设备类型分类（B-球阀、C-压缩机、P-管道等）和正则表达式过滤。
    支持通配符批量添加变量（例如：N_005* 添加所有N_005相关变量）。
    时间窗口固定为2025年1月1日0点到24点。
    """)
    
    # Initialize session state
    if 'variable_rows' not in st.session_state:
        st.session_state.variable_rows = [str(uuid.uuid4())]
    if 'dataset_selections' not in st.session_state:
        st.session_state.dataset_selections = {}
    if 'variable_selections' not in st.session_state:
        st.session_state.variable_selections = {}
    
    # Get available datasets
    available_datasets = get_available_datasets()
    
    if not available_datasets:
        st.error("No datasets found. Please check your data directory structure.")
        return
    
    # Sidebar for variable selection
    st.sidebar.header("变量选择")
    st.sidebar.markdown("选择要比较的数据集和变量：")
    
    # Batch add variables section
    st.sidebar.subheader("📦 批量添加变量")
    st.sidebar.markdown("使用通配符批量添加变量（例如：N_005*）")
    
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        batch_dataset = st.selectbox(
            "批量数据集",
            options=[""] + available_datasets,
            key="batch_dataset",
            help="选择要批量添加变量的数据集"
        )
    
    batch_pattern = st.sidebar.text_input(
        "通配符模式",
        value="",
        key="batch_pattern",
        help="使用 * 作为通配符。例如：N_005* 匹配所有N_005开头的变量",
        placeholder="例如：N_005* 或 B_001* 或 *_p_in"
    )
    
    # Preview matching variables
    if batch_dataset and batch_pattern:
        matching_vars = expand_wildcard_pattern(batch_dataset, batch_pattern)
        if matching_vars:
            st.sidebar.markdown(f"**找到 {len(matching_vars)} 个匹配变量：**")
            preview_vars = matching_vars[:5]  # Show first 5
            for var in preview_vars:
                st.sidebar.markdown(f"- {var}")
            if len(matching_vars) > 5:
                st.sidebar.markdown(f"... 还有 {len(matching_vars) - 5} 个变量")
            
            if st.sidebar.button("➕ 批量添加这些变量", key="batch_add"):
                add_multiple_variables(batch_dataset, matching_vars)
                st.sidebar.success(f"成功添加 {len(matching_vars)} 个变量！")
                st.rerun()
        else:
            st.sidebar.warning("未找到匹配的变量")
    
    st.sidebar.markdown("---")
    
    # Add regex filter input
    st.sidebar.subheader("🔍 变量过滤")
    filter_pattern = st.sidebar.text_input(
        "正则表达式过滤（可选）",
        value="",
        help="使用正则表达式过滤变量名。例如：'p_in|p_out' 匹配所有压力变量，'B_001' 匹配特定球阀变量",
        placeholder="例如：p_in|p_out 或 ^B_ 或 .*temperature.*"
    )
    
    # Add new variable row button
    if st.sidebar.button("➕ 添加变量", key="add_variable"):
        st.session_state.variable_rows.append(str(uuid.uuid4()))
        st.rerun()
    
    # Create variable selection rows
    variable_selections = []
    rows_to_remove = []
    
    for row_id in st.session_state.variable_rows:
        st.sidebar.markdown(f"**变量 {len(variable_selections) + 1}:**")
        
        with st.sidebar.container():
            selection = variable_selection_row(row_id, available_datasets, filter_pattern)
            
            if selection['remove_clicked'] and len(st.session_state.variable_rows) > 1:
                rows_to_remove.append(row_id)
            else:
                variable_selections.append(selection)
        
        st.sidebar.markdown("---")
    
    # Remove rows that were marked for removal
    for row_id in rows_to_remove:
        st.session_state.variable_rows.remove(row_id)
        # Clean up session state for removed rows
        if 'dataset_selections' in st.session_state and row_id in st.session_state.dataset_selections:
            del st.session_state.dataset_selections[row_id]
        if 'variable_selections' in st.session_state and row_id in st.session_state.variable_selections:
            del st.session_state.variable_selections[row_id]
        st.rerun()
    
    # Filter valid selections
    valid_selections = [sel for sel in variable_selections 
                       if sel['dataset'] and sel['variable']]
    
    # Main content area
    if valid_selections:
        st.subheader(f"正在比较 {len(valid_selections)} 个变量")
        
        # Show selected variables info
        with st.expander("📋 已选变量", expanded=False):
            for i, sel in enumerate(valid_selections, 1):
                boundary_vars, equipment_vars, equipment_categorized = get_available_variables(sel['dataset'])
                var_type = "边界条件" if sel['variable'] in boundary_vars else "设备参数"
                
                # Find specific equipment category if it's an equipment variable
                if sel['variable'] in equipment_vars:
                    for category, vars_in_category in equipment_categorized.items():
                        if sel['variable'] in vars_in_category:
                            var_type = category
                            break
                
                st.write(f"{i}. **{sel['dataset']}**: {sel['variable']} ({var_type})")
        
        # Create and display the plot
        try:
            fig = create_multi_variable_plot(valid_selections)
            st.plotly_chart(fig, width='stretch')
            
            # Display statistics
            st.subheader("📊 变量统计")
            
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
                        '数据集': sel['dataset'],
                        '变量': sel['variable'],
                        '均值': var_data.mean(),
                        '标准差': var_data.std(),
                        '最小值': var_data.min(),
                        '最大值': var_data.max(),
                        '数据点': len(var_data)
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, width='stretch')
            
        except Exception as e:
            st.error(f"Error creating plot: {e}")
    
    else:
        st.info("👆 从侧边栏选择数据集和变量开始比较时间序列数据。")
        
        # Show available datasets info
        st.subheader("📁 可用数据集")
        st.write(f"找到 {len(available_datasets)} 个数据集：")
        
        for i, dataset in enumerate(available_datasets, 1):
            with st.expander(f"{i}. {dataset}", expanded=False):
                boundary_vars, equipment_vars, equipment_categorized = get_available_variables(dataset)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**边界条件变量 ({len(boundary_vars)}):**")
                    if boundary_vars:
                        for var in boundary_vars[:10]:  # Show first 10
                            st.write(f"- {var}")
                        if len(boundary_vars) > 10:
                            st.write(f"... 还有 {len(boundary_vars) - 10} 个变量")
                
                with col2:
                    st.write(f"**设备参数变量 ({len(equipment_vars)}):**")
                    for category, vars_in_category in equipment_categorized.items():
                        if vars_in_category:
                            st.write(f"*{category} ({len(vars_in_category)}):*")
                            for var in vars_in_category[:5]:  # Show first 5 per category
                                st.write(f"  - {var}")
                            if len(vars_in_category) > 5:
                                st.write(f"  ... 还有 {len(vars_in_category) - 5} 个变量")

if __name__ == "__main__":
    run_multi_variable_viewer()