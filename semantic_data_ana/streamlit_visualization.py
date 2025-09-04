"""
气源和分输点边界数据可视化应用
使用Streamlit创建交互式可视化界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import os


class TEVisualizationApp:
    """
    气源和分输点数据可视化应用类
    """
    
    def __init__(self, data_dir: str):
        """
        初始化应用
        
        Args:
            data_dir: 处理后的数据目录
        """
        self.data_dir = Path(data_dir)
        
        # 定义颜色配置
        self.gas_source_colors = {
            'T_002:SNQ': '#FF6B6B',
            'T_003:SNQ': '#4ECDC4', 
            'T_004:SNQ': '#45B7D1'
        }
        
        self.distribution_colors = {
            'E_001:SNQ': '#96CEB4', 'E_002:SNQ': '#FECA57', 'E_003:SNQ': '#48CAE4',
            'E_004:SNQ': '#FF9F43', 'E_005:SNQ': '#10AC84', 'E_006:SNQ': '#5F27CD',
            'E_007:SNQ': '#00D2D3', 'E_008:SNQ': '#FF3838', 'E_009:SNQ': '#2E86AB',
            'E_060:SNQ': '#A23B72', 'E_061:SNQ': '#F18F01', 'E_062:SNQ': '#C73E1D',
            'E_109:SNQ': '#6BCF7F'
        }
    
    def find_available_cases(self):
        """
        查找可用的数据案例
        
        Returns:
            可用案例列表
        """
        if not self.data_dir.exists():
            return []
        
        cases = []
        for file in self.data_dir.glob("*_target_TE.csv"):
            case_name = file.name.replace('_target_TE.csv', '')
            cases.append(case_name)
        
        return sorted(cases)
    
    def load_case_data(self, case_name: str):
        """
        加载指定案例的数据
        
        Args:
            case_name: 案例名称
            
        Returns:
            数据框
        """
        file_path = self.data_dir / f"{case_name}_target_TE.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['TIME'] = pd.to_datetime(df['TIME'])
            return df
        return None
    
    def get_days_from_data(self, df: pd.DataFrame):
        """
        从数据中提取可用的日期
        
        Args:
            df: 数据框
            
        Returns:
            日期列表
        """
        if 'date' in df.columns:
            return sorted(df['date'].unique())
        else:
            return sorted(df['TIME'].dt.date.unique())
    
    def filter_data_by_date(self, df: pd.DataFrame, selected_date: str):
        """
        根据日期过滤数据
        
        Args:
            df: 数据框
            selected_date: 选择的日期
            
        Returns:
            过滤后的数据框
        """
        if 'date' in df.columns:
            return df[df['date'] == selected_date].copy()
        else:
            date_obj = pd.to_datetime(selected_date).date()
            return df[df['TIME'].dt.date == date_obj].copy()
    
    def create_line_plot(self, df: pd.DataFrame, selected_columns: list, title: str):
        """
        创建折线图
        
        Args:
            df: 数据框
            selected_columns: 选择的列
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        # 添加气源数据
        gas_columns = [col for col in selected_columns if col.startswith('T_')]
        for col in gas_columns:
            if col in df.columns:
                color = self.gas_source_colors.get(col, '#000000')
                fig.add_trace(go.Scatter(
                    x=df['TIME'],
                    y=df[col],
                    mode='lines+markers',
                    name=f'气源 {col}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间: %{x}<br>' +
                                 '数值: %{y:.4f}<extra></extra>'
                ))
        
        # 添加分输点数据
        dist_columns = [col for col in selected_columns if col.startswith('E_')]
        for col in dist_columns:
            if col in df.columns:
                color = self.distribution_colors.get(col, '#808080')
                fig.add_trace(go.Scatter(
                    x=df['TIME'],
                    y=df[col],
                    mode='lines+markers',
                    name=f'分输点 {col}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间: %{x}<br>' +
                                 '数值: %{y:.4f}<extra></extra>'
                ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="时间",
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
    
    def create_comparison_plot(self, df: pd.DataFrame, gas_columns: list, dist_columns: list):
        """
        创建气源和分输点的对比图
        
        Args:
            df: 数据框
            gas_columns: 气源列
            dist_columns: 分输点列
            
        Returns:
            Plotly子图对象
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('气源(T系列)', '分输点(E系列)'),
            vertical_spacing=0.15
        )
        
        # 添加气源数据
        for col in gas_columns:
            if col in df.columns:
                color = self.gas_source_colors.get(col, '#000000')
                fig.add_trace(go.Scatter(
                    x=df['TIME'],
                    y=df[col],
                    mode='lines+markers',
                    name=f'{col}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=True,
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间: %{x}<br>' +
                                 '数值: %{y:.4f}<extra></extra>'
                ), row=1, col=1)
        
        # 添加分输点数据
        for col in dist_columns:
            if col in df.columns:
                color = self.distribution_colors.get(col, '#808080')
                fig.add_trace(go.Scatter(
                    x=df['TIME'],
                    y=df[col],
                    mode='lines+markers',
                    name=f'{col}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=True,
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间: %{x}<br>' +
                                 '数值: %{y:.4f}<extra></extra>'
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
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', title_text="时间", row=2, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', title_text="SNQ值")
        
        return fig
    
    def create_statistics_summary(self, df: pd.DataFrame, selected_columns: list):
        """
        创建统计摘要
        
        Args:
            df: 数据框
            selected_columns: 选择的列
            
        Returns:
            统计摘要数据框
        """
        numeric_columns = [col for col in selected_columns if col in df.columns and col not in ['TIME', 'date', 'hour_minute']]
        
        if not numeric_columns:
            return pd.DataFrame()
        
        stats = df[numeric_columns].describe().round(4)
        stats = stats.T  # 转置
        stats['变化率(%)'] = ((stats['max'] - stats['min']) / stats['mean'] * 100).round(2)
        
        return stats


def main():
    """主应用函数"""
    st.set_page_config(
        page_title="气源和分输点边界数据可视化",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔥 气源和分输点边界数据可视化")
    st.markdown("---")
    
    # 初始化应用
    data_dir = "/home/chbds/zly/gaspipe/fluid_model/semantic_data_ana/specific_te_data"
    app = TEVisualizationApp(data_dir)
    
    # 检查数据目录是否存在
    if not Path(data_dir).exists():
        st.error(f"数据目录不存在: {data_dir}")
        st.info("请先运行数据处理脚本生成数据")
        return
    
    # 获取可用案例
    available_cases = app.find_available_cases()
    
    if not available_cases:
        st.error("未找到任何可用的数据文件")
        st.info("请确保数据处理脚本已经运行并生成了数据文件")
        return
    
    # 侧边栏控制
    with st.sidebar:
        st.header("📋 控制面板")
        
        # 案例选择
        selected_case = st.selectbox(
            "选择案例",
            available_cases,
            help="选择要可视化的算例"
        )
        
        # 加载数据
        if selected_case:
            df = app.load_case_data(selected_case)
            
            if df is not None:
                # 日期选择
                available_dates = app.get_days_from_data(df)
                selected_date = st.selectbox(
                    "选择日期",
                    available_dates,
                    help="选择要显示的日期"
                )
                
                # 过滤日期数据
                daily_df = app.filter_data_by_date(df, selected_date)
                
                # 变量选择
                st.subheader("🔧 变量选择")
                
                # 气源变量
                gas_columns = [col for col in df.columns if col.startswith('T_')]
                selected_gas = st.multiselect(
                    "气源变量",
                    gas_columns,
                    default=gas_columns,
                    help="选择要显示的气源变量"
                )
                
                # 分输点变量
                dist_columns = [col for col in df.columns if col.startswith('E_')]
                selected_dist = st.multiselect(
                    "分输点变量", 
                    dist_columns,
                    default=dist_columns[:5],  # 默认选择前5个
                    help="选择要显示的分输点变量"
                )
                
                # 可视化选项
                st.subheader("📊 可视化选项")
                viz_type = st.radio(
                    "图表类型",
                    ["统一折线图", "分类对比图"],
                    help="选择可视化类型"
                )
                
                show_stats = st.checkbox("显示统计信息", value=True)
            else:
                st.error("无法加载数据文件")
                return
        else:
            st.info("请选择一个案例")
            return
    
    # 主内容区域
    if selected_case and df is not None:
        # 显示基本信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="案例",
                value=selected_case,
            )
        
        with col2:
            st.metric(
                label="选择日期",
                value=str(selected_date),
            )
        
        with col3:
            st.metric(
                label="数据点数",
                value=len(daily_df),
            )
        
        # 数据预览
        with st.expander("📈 数据预览", expanded=False):
            st.dataframe(
                daily_df.head(10),
                use_container_width=True
            )
        
        # 主要可视化
        all_selected = selected_gas + selected_dist
        
        if all_selected:
            st.subheader(f"📊 {selected_case} - {selected_date} 数据可视化")
            
            if viz_type == "统一折线图":
                # 统一折线图
                title = f"{selected_case} - {selected_date} 气源和分输点SNQ数据"
                fig = app.create_line_plot(daily_df, all_selected, title)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # 分类对比图
                fig = app.create_comparison_plot(daily_df, selected_gas, selected_dist)
                st.plotly_chart(fig, use_container_width=True)
            
            # 统计信息
            if show_stats and all_selected:
                st.subheader("📋 统计摘要")
                
                stats_df = app.create_statistics_summary(daily_df, all_selected)
                if not stats_df.empty:
                    st.dataframe(
                        stats_df,
                        use_container_width=True
                    )
                else:
                    st.info("没有数值数据可以计算统计信息")
        else:
            st.warning("请至少选择一个变量进行可视化")
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            🔬 流体大模型开发工作 - 气源和分输点边界数据分析工具
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()