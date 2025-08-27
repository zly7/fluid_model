import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_dashboard():
    """创建Streamlit仪表板"""
    
    st.set_page_config(
        page_title="天然气管网数值模拟仪表板",
        page_icon="⛽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⛽ 天然气管网数值模拟仪表板")
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.title("控制面板")
    
    # 文件上传
    st.sidebar.subheader("数据文件")
    uploaded_boundary = st.sidebar.file_uploader(
        "上传边界条件文件 (Boundary.csv)", 
        type=['csv']
    )
    
    uploaded_results = st.sidebar.file_uploader(
        "上传结果文件夹", 
        type=['csv'],
        accept_multiple_files=True
    )
    
    # 可视化选项
    st.sidebar.subheader("可视化选项")
    show_time_series = st.sidebar.checkbox("显示时间序列", value=True)
    show_distribution = st.sidebar.checkbox("显示分布图", value=True)
    show_correlation = st.sidebar.checkbox("显示相关性分析", value=False)
    show_network = st.sidebar.checkbox("显示网络拓扑", value=False)
    
    # 主界面
    if uploaded_boundary is not None:
        display_boundary_analysis(uploaded_boundary)
    
    if uploaded_results:
        display_results_analysis(uploaded_results)
    
    if show_time_series and uploaded_results:
        display_time_series_analysis(uploaded_results)
    
    if show_distribution and uploaded_results:
        display_distribution_analysis(uploaded_results)
    
    if show_correlation and uploaded_results:
        display_correlation_analysis(uploaded_results)
    
    if show_network:
        display_network_topology()


def display_boundary_analysis(uploaded_file):
    """显示边界条件分析"""
    st.subheader("📊 边界条件分析")
    
    try:
        # 读取数据
        boundary_data = pd.read_csv(uploaded_file)
        
        # 基础统计信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("时间步数", len(boundary_data))
        
        with col2:
            st.metric("边界条件数量", len(boundary_data.columns) - 1)
        
        with col3:
            non_zero_cols = (boundary_data.drop(columns=['TIME']) != 0).sum().sum()
            st.metric("非零边界条件", non_zero_cols)
        
        # 数据预览
        st.subheader("数据预览")
        st.dataframe(boundary_data.head(10))
        
        # 边界条件热图
        st.subheader("边界条件热图")
        numeric_data = boundary_data.drop(columns=['TIME'])
        
        fig = go.Figure(data=go.Heatmap(
            z=numeric_data.T,
            x=boundary_data['TIME'],
            y=numeric_data.columns,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="边界条件随时间变化",
            xaxis_title="时间",
            yaxis_title="边界条件",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"读取边界条件文件时出错: {str(e)}")


def display_results_analysis(uploaded_files):
    """显示结果分析"""
    st.subheader("📈 结果分析")
    
    try:
        results_data = {}
        
        # 读取所有上传的文件
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            df = pd.read_csv(uploaded_file)
            results_data[file_name] = df
        
        if not results_data:
            st.warning("请上传结果文件")
            return
        
        # 文件选择器
        selected_file = st.selectbox(
            "选择要分析的文件:",
            list(results_data.keys())
        )
        
        if selected_file:
            df = results_data[selected_file]
            
            # 基础统计
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("样本数量", len(df))
            
            with col2:
                st.metric("变量数量", len(df.columns))
            
            with col3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.metric("平均值", f"{df[numeric_cols].mean().mean():.4f}")
            
            with col4:
                if len(numeric_cols) > 0:
                    st.metric("标准差", f"{df[numeric_cols].std().mean():.4f}")
            
            # 描述性统计
            st.subheader("描述性统计")
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe())
            
    except Exception as e:
        st.error(f"分析结果文件时出错: {str(e)}")


def display_time_series_analysis(uploaded_files):
    """显示时间序列分析"""
    st.subheader("📊 时间序列分析")
    
    try:
        results_data = {}
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            df = pd.read_csv(uploaded_file)
            results_data[file_name] = df
        
        if not results_data:
            return
        
        # 选择文件和变量
        col1, col2 = st.columns(2)
        
        with col1:
            selected_file = st.selectbox(
                "选择文件:",
                list(results_data.keys()),
                key="time_series_file"
            )
        
        with col2:
            if selected_file:
                df = results_data[selected_file]
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                selected_columns = st.multiselect(
                    "选择变量:",
                    numeric_cols,
                    default=numeric_cols[:5]  # 默认选择前5个
                )
        
        if selected_file and selected_columns:
            df = results_data[selected_file]
            
            # 创建时间序列图
            fig = go.Figure()
            
            for col in selected_columns:
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        y=df[col],
                        name=col,
                        mode='lines'
                    ))
            
            fig.update_layout(
                title=f"{selected_file} - 时间序列",
                xaxis_title="时间步",
                yaxis_title="数值",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"时间序列分析出错: {str(e)}")


def display_distribution_analysis(uploaded_files):
    """显示分布分析"""
    st.subheader("📊 数据分布分析")
    
    try:
        results_data = {}
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            df = pd.read_csv(uploaded_file)
            results_data[file_name] = df
        
        if not results_data:
            return
        
        # 选择文件
        selected_file = st.selectbox(
            "选择文件:",
            list(results_data.keys()),
            key="distribution_file"
        )
        
        if selected_file:
            df = results_data[selected_file]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                # 选择变量
                selected_col = st.selectbox(
                    "选择变量:",
                    numeric_cols,
                    key="distribution_col"
                )
                
                if selected_col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 直方图
                        fig_hist = px.histogram(
                            df, 
                            x=selected_col,
                            title=f"{selected_col} 分布直方图",
                            nbins=50
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # 箱线图
                        fig_box = px.box(
                            df,
                            y=selected_col,
                            title=f"{selected_col} 箱线图"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                
                # 多变量分布比较
                if len(numeric_cols) >= 2:
                    st.subheader("多变量分布比较")
                    selected_cols = st.multiselect(
                        "选择要比较的变量:",
                        numeric_cols,
                        default=numeric_cols[:3]
                    )
                    
                    if len(selected_cols) >= 2:
                        # 散点图矩阵
                        fig_scatter = px.scatter_matrix(
                            df[selected_cols],
                            title="散点图矩阵"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            
    except Exception as e:
        st.error(f"分布分析出错: {str(e)}")


def display_correlation_analysis(uploaded_files):
    """显示相关性分析"""
    st.subheader("🔗 相关性分析")
    
    try:
        results_data = {}
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            df = pd.read_csv(uploaded_file)
            results_data[file_name] = df
        
        if not results_data:
            return
        
        # 选择文件
        selected_file = st.selectbox(
            "选择文件:",
            list(results_data.keys()),
            key="correlation_file"
        )
        
        if selected_file:
            df = results_data[selected_file]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # 计算相关性矩阵
                corr_matrix = df[numeric_cols].corr()
                
                # 相关性热图
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title="变量相关性矩阵",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 高相关性变量对
                st.subheader("高相关性变量对")
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # 高相关性阈值
                            high_corr_pairs.append({
                                '变量1': corr_matrix.columns[i],
                                '变量2': corr_matrix.columns[j],
                                '相关系数': corr_val
                            })
                
                if high_corr_pairs:
                    high_corr_df = pd.DataFrame(high_corr_pairs)
                    st.dataframe(high_corr_df)
                else:
                    st.info("没有发现高相关性(>0.7)的变量对")
            
    except Exception as e:
        st.error(f"相关性分析出错: {str(e)}")


def display_network_topology():
    """显示网络拓扑"""
    st.subheader("🌐 管网拓扑图")
    
    # 这里应该根据实际的管网数据来绘制
    # 为了演示，创建一个简单的网络图
    
    try:
        # 创建示例网络数据
        nodes = {
            'T_001': {'x': 0, 'y': 0, 'type': '气源', 'size': 20},
            'N_001': {'x': 1, 'y': 0, 'type': '节点', 'size': 10},
            'N_002': {'x': 2, 'y': 1, 'type': '节点', 'size': 10},
            'N_003': {'x': 2, 'y': -1, 'type': '节点', 'size': 10},
            'E_001': {'x': 3, 'y': 0, 'type': '分输点', 'size': 15},
            'C_001': {'x': 1.5, 'y': 0.5, 'type': '压缩机', 'size': 18}
        }
        
        edges = [
            ('T_001', 'N_001'),
            ('N_001', 'C_001'),
            ('C_001', 'N_002'),
            ('N_001', 'N_003'),
            ('N_002', 'E_001'),
            ('N_003', 'E_001')
        ]
        
        # 创建网络图
        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = nodes[edge[0]]['x'], nodes[edge[0]]['y']
            x1, y1 = nodes[edge[1]]['x'], nodes[edge[1]]['y']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # 绘制边
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 绘制节点
        node_x = [nodes[node]['x'] for node in nodes]
        node_y = [nodes[node]['y'] for node in nodes]
        node_text = list(nodes.keys())
        node_size = [nodes[node]['size'] for node in nodes]
        node_color = []
        
        for node in nodes:
            node_type = nodes[node]['type']
            if node_type == '气源':
                node_color.append('red')
            elif node_type == '分输点':
                node_color.append('blue')
            elif node_type == '压缩机':
                node_color.append('green')
            else:
                node_color.append('gray')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='middle center',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext=[f"{node}: {nodes[node]['type']}" for node in nodes]
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='管网拓扑示例',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[
                               dict(
                                   text="红色: 气源, 蓝色: 分输点, 绿色: 压缩机, 灰色: 节点",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(color="black", size=12)
                               )
                           ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=500
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("这是一个示例网络拓扑图。实际使用时需要根据具体的管网数据进行配置。")
        
    except Exception as e:
        st.error(f"显示网络拓扑时出错: {str(e)}")


if __name__ == "__main__":
    create_dashboard()