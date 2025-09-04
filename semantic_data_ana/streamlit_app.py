"""
新的气源和分输点边界数据可视化应用
基于Boundary.csv数据的专用可视化工具
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import os
import sys

# 添加项目路径到系统路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from boundary_processor import BoundaryDataProcessor


class NewBoundaryVisualizationApp:
    """
    新的边界数据可视化应用类
    专门针对Boundary.csv中的特定气源和分输点数据
    """
    
    def __init__(self, data_root: str):
        """
        初始化应用
        
        Args:
            data_root: 数据根目录
        """
        self.data_root = Path(data_root)
        self.processor = BoundaryDataProcessor()
        
        # 定义颜色配置
        self.gas_source_colors = {
            'T_002:SNQ': '#FF0000',
            'T_003:SNQ': '#00BFFF', 
            'T_004:SNQ': '#1E90FF'
        }
        
        self.distribution_colors = {
            'E_001:SNQ': '#228B22', 'E_002:SNQ': '#FF8C00', 'E_003:SNQ': '#1E90FF',
            'E_004:SNQ': '#FF4500', 'E_005:SNQ': '#00A86B', 'E_006:SNQ': '#8A2BE2',
            'E_007:SNQ': '#00CED1', 'E_008:SNQ': '#DC143C', 'E_009:SNQ': '#191970',
            'E_060:SNQ': '#8B0000', 'E_061:SNQ': '#FF6347', 'E_062:SNQ': '#B22222',
            'E_109:SNQ': '#32CD32'
        }
    
    def get_available_cases(self) -> list:
        """
        获取可用的算例列表
        
        Returns:
            算例名称列表
        """
        return self.processor.find_all_cases(str(self.data_root))
    
    def load_case_data(self, case_name: str) -> pd.DataFrame:
        """
        加载指定算例的数据
        
        Args:
            case_name: 算例名称
            
        Returns:
            数据框
        """
        return self.processor.load_boundary_data(str(self.data_root), case_name)
    
    def create_line_plot(self, df: pd.DataFrame, selected_columns: list, case_name: str) -> go.Figure:
        """
        创建折线图
        
        Args:
            df: 数据框
            selected_columns: 选择的列
            case_name: 算例名称
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        # 添加气源数据
        gas_columns = [col for col in selected_columns if col in self.gas_source_colors]
        for col in gas_columns:
            if col in df.columns:
                color = self.gas_source_colors[col]
                fig.add_trace(go.Scatter(
                    x=df['time_index'],
                    y=df[col],
                    mode='lines+markers',
                    name=f'气源 {col}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间点: %{x}<br>' +
                                 'SNQ值: %{y:.4f}<extra></extra>'
                ))
        
        # 添加分输点数据
        dist_columns = [col for col in selected_columns if col in self.distribution_colors]
        for col in dist_columns:
            if col in df.columns:
                color = self.distribution_colors[col]
                fig.add_trace(go.Scatter(
                    x=df['time_index'],
                    y=df[col],
                    mode='lines+markers',
                    name=f'分输点 {col}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间点: %{x}<br>' +
                                 'SNQ值: %{y:.4f}<extra></extra>'
                ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"{case_name} - 气源和分输点SNQ数据",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="时间点 (30分钟间隔)",
            yaxis_title="SNQ值",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            )
        )
        
        # 设置坐标轴格式
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    def create_comparison_plot(self, df: pd.DataFrame, gas_columns: list, dist_columns: list, case_name: str) -> go.Figure:
        """
        创建气源和分输点的对比图
        
        Args:
            df: 数据框
            gas_columns: 气源列
            dist_columns: 分输点列
            case_name: 算例名称
            
        Returns:
            Plotly子图对象
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{case_name} - 气源(T系列)', f'{case_name} - 分输点(E系列)'),
            vertical_spacing=0.15
        )
        
        # 添加气源数据
        for col in gas_columns:
            if col in df.columns:
                color = self.gas_source_colors.get(col, '#000000')
                fig.add_trace(go.Scatter(
                    x=df['time_index'],
                    y=df[col],
                    mode='lines+markers',
                    name=f'{col}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=True,
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间点: %{x}<br>' +
                                 'SNQ值: %{y:.4f}<extra></extra>'
                ), row=1, col=1)
        
        # 添加分输点数据
        for col in dist_columns:
            if col in df.columns:
                color = self.distribution_colors.get(col, '#808080')
                fig.add_trace(go.Scatter(
                    x=df['time_index'],
                    y=df[col],
                    mode='lines+markers',
                    name=f'{col}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=True,
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间点: %{x}<br>' +
                                 'SNQ值: %{y:.4f}<extra></extra>'
                ), row=2, col=1)
        
        # 更新布局
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            )
        )
        
        # 更新坐标轴
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', title_text="时间点 (30分钟间隔)", row=2, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', title_text="SNQ值")
        
        return fig
    
    def create_statistics_summary(self, df: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
        """
        创建统计摘要
        
        Args:
            df: 数据框
            selected_columns: 选择的列
            
        Returns:
            统计摘要数据框
        """
        numeric_columns = [col for col in selected_columns if col in df.columns and col not in ['TIME', 'time_index', 'case_name', 'case_id']]
        
        if not numeric_columns:
            return pd.DataFrame()
        
        stats = df[numeric_columns].describe().round(4)
        stats = stats.T  # 转置
        stats['变化率(%)'] = ((stats['max'] - stats['min']) / stats['mean'] * 100).round(2)
        
        return stats


def main():
    """主应用函数"""
    st.set_page_config(
        page_title="边界数据可视化 - 气源和分输点SNQ",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ 管网边界数据可视化 - 气源和分输点SNQ")
    st.markdown("---")
    
    # 数据根目录
    data_root = "/home/chbds/zly/gaspipe/fluid_model/data/dataset"
    
    # 初始化应用
    if not Path(data_root).exists():
        st.error(f"数据目录不存在: {data_root}")
        st.info("请检查数据路径设置")
        return
    
    app = NewBoundaryVisualizationApp(data_root)
    
    # 获取可用算例
    available_cases = app.get_available_cases()
    
    if not available_cases:
        st.error("未找到任何可用的Boundary.csv文件")
        st.info("请确保数据目录包含正确的算例文件夹结构")
        return
    
    # 侧边栏控制
    with st.sidebar:
        st.header("📋 控制面板")
        
        # 算例选择
        selected_case = st.selectbox(
            "选择算例",
            available_cases,
            help="选择要可视化的算例"
        )
        
        # 加载数据
        if selected_case:
            df = app.load_case_data(selected_case)
            
            if df is not None:
                # 变量选择
                st.subheader("🔧 变量选择")
                
                # 气源变量
                gas_columns = [col for col in app.processor.gas_sources if col in df.columns]
                
                st.write("**气源变量 (T系列)**")
                # 使用session state来管理气源变量的显示状态
                if 'gas_visibility' not in st.session_state:
                    st.session_state.gas_visibility = {col: True for col in gas_columns}
                
                gas_cols = st.columns(len(gas_columns) if gas_columns else 1)
                for i, col in enumerate(gas_columns):
                    with gas_cols[i % len(gas_cols)]:
                        current_state = st.session_state.gas_visibility.get(col, True)
                        if st.button(f"{'✓' if current_state else '✗'} {col.split(':')[0]}", 
                                   key=f"gas_{col}", 
                                   help=f"点击切换 {col} 的显示状态"):
                            st.session_state.gas_visibility[col] = not current_state
                            st.rerun()
                
                selected_gas = [col for col in gas_columns if st.session_state.gas_visibility.get(col, True)]
                
                # 分输点变量
                dist_columns = [col for col in app.processor.distribution_points if col in df.columns]
                
                st.write("**分输点变量 (E系列)**")
                # 使用session state来管理分输点变量的显示状态
                if 'dist_visibility' not in st.session_state:
                    st.session_state.dist_visibility = {col: i < 8 for i, col in enumerate(dist_columns)}  # 默认显示前8个
                
                # 确保所有当前分输点列都在session state中
                for col in dist_columns:
                    if col not in st.session_state.dist_visibility:
                        st.session_state.dist_visibility[col] = False
                
                # 创建多列布局来放置按钮
                num_cols = 3
                dist_button_cols = st.columns(num_cols)
                for i, col in enumerate(dist_columns):
                    with dist_button_cols[i % num_cols]:
                        current_visible = st.session_state.dist_visibility.get(col, False)
                        if st.button(f"{'✓' if current_visible else '✗'} {col.split(':')[0]}", 
                                   key=f"dist_{col}",
                                   help=f"点击切换 {col} 的显示状态"):
                            st.session_state.dist_visibility[col] = not current_visible
                            st.rerun()
                
                selected_dist = [col for col in dist_columns if st.session_state.dist_visibility.get(col, False)]
                
                # 显示选中变量统计
                st.markdown("---")
                col_left, col_right = st.columns(2)
                with col_left:
                    st.metric("气源选中", f"{len(selected_gas)}/{len(gas_columns)}")
                with col_right:
                    st.metric("分输点选中", f"{len(selected_dist)}/{len(dist_columns)}")
                
                # 可视化选项
                st.subheader("📊 可视化选项")
                viz_type = st.radio(
                    "图表类型",
                    ["统一折线图", "分类对比图"],
                    help="选择可视化类型"
                )
                
                show_stats = st.checkbox("显示统计信息", value=True)
                
                # 数据信息
                st.subheader("📋 数据信息")
                st.metric("数据点数", len(df))
                st.metric("时间跨度", f"{len(df) * 0.5:.1f} 小时")
                if len(df) > 0:
                    st.metric("开始时间", df['TIME'].min().strftime('%Y-%m-%d %H:%M'))
                    st.metric("结束时间", df['TIME'].max().strftime('%Y-%m-%d %H:%M'))
            else:
                st.error("无法加载数据文件")
                return
        else:
            st.info("请选择一个算例")
            return
    
    # 主内容区域
    if selected_case and df is not None:
        # 显示基本信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="选择算例",
                value=selected_case,
            )
        
        with col2:
            available_gas = len([col for col in app.processor.gas_sources if col in df.columns])
            st.metric(
                label="可用气源",
                value=f"{available_gas}/{len(app.processor.gas_sources)}",
            )
        
        with col3:
            available_dist = len([col for col in app.processor.distribution_points if col in df.columns])
            st.metric(
                label="可用分输点",
                value=f"{available_dist}/{len(app.processor.distribution_points)}",
            )
        
        # 数据预览
        with st.expander("📈 数据预览", expanded=False):
            preview_columns = ['TIME', 'time_index'] + selected_gas + selected_dist
            preview_df = df[preview_columns].head(10)
            st.dataframe(preview_df, width='stretch')
        
        # 主要可视化
        all_selected = selected_gas + selected_dist
        
        if all_selected:
            st.subheader(f"📊 {selected_case} - 边界数据可视化")
            
            if viz_type == "统一折线图":
                # 统一折线图
                fig = app.create_line_plot(df, all_selected, selected_case)
                st.plotly_chart(fig, width='stretch')
                
            else:
                # 分类对比图
                fig = app.create_comparison_plot(df, selected_gas, selected_dist, selected_case)
                st.plotly_chart(fig, width='stretch')
            
            # 统计信息
            if show_stats and all_selected:
                st.subheader("📋 统计摘要")
                
                stats_df = app.create_statistics_summary(df, all_selected)
                if not stats_df.empty:
                    st.dataframe(stats_df, width='stretch')
                else:
                    st.info("没有数值数据可以计算统计信息")
        else:
            st.warning("请至少选择一个变量进行可视化")
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            🔬 管网边界数据可视化工具 - 基于Boundary.csv数据
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()