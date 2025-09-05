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
        
        # 定义基本颜色调色板
        self.color_palette = [
            '#FF0000', '#00BFFF', '#1E90FF', '#228B22', '#FF8C00', 
            '#FF4500', '#00A86B', '#8A2BE2', '#00CED1', '#DC143C', 
            '#191970', '#8B0000', '#FF6347', '#B22222', '#32CD32',
            '#FF69B4', '#4169E1', '#FF1493', '#00FF7F', '#FFD700',
            '#9370DB', '#20B2AA', '#F4A460', '#DA70D6', '#87CEEB',
            '#FF6347', '#40E0D0', '#EE82EE', '#90EE90', '#FFA500',
            '#CD853F', '#DDA0DD', '#98FB98', '#F0E68C', '#DEB887'
        ]
        
        # 存储动态分配的颜色映射
        self.variable_colors = {}
        
        # 预定义的变量方案 - 更新为支持多种类型
        self.predefined_schemes = {
            "方案1 - 关键阀门+主气源+重点分输": {
                "description": "N_138附近节点的关键设备",
                "variables": ['B_242:FR', 'B_243:FR', 'R_001:ST','R_001:SPD', 'T_003:SNQ', 'E_060:SNQ', 'E_061:SNQ', 'E_062:SNQ']
            },
            "方案2 - 中段压缩机组合": {
                "description": "主要气源T_002、T_003和关键分输点E_109",
                "variables": ['E_108:SNQ', 'E_107:SNQ', 'C_016:ST', 'C_016:SP_out','C_017:ST', 'C_017:SP_out',"B_306:FR"]
            },
            "方案3 - 全部气源": {
                "description": "所有气源的SNQ数据",
                "variables": []  # 将在运行时动态填充
            },
            "方案4 - 核心分输点": {
                "description": "核心分输点E_001到E_005",
                "variables": ['E_001:SNQ', 'E_002:SNQ', 'E_003:SNQ', 'E_004:SNQ', 'E_005:SNQ']
            },
            "方案5 - 重要阀门": {
                "description": "关键阀门B_240-B_245",
                "variables": ['B_240:FR', 'B_241:FR', 'B_242:FR', 'B_243:FR', 'B_244:FR', 'B_245:FR']
            },
            "方案6 - 调节器监控": {
                "description": "前5个调节器的状态和速度",
                "variables": ['R_001:ST', 'R_001:SPD', 'R_002:ST', 'R_002:SPD', 'R_003:ST', 'R_003:SPD']
            }
        }
    
    def get_variable_color(self, variable: str) -> str:
        """
        获取变量的颜色，如果不存在则动态分配
        
        Args:
            variable: 变量名
            
        Returns:
            颜色代码
        """
        if variable not in self.variable_colors:
            # 根据已使用的颜色数量选择新颜色
            color_index = len(self.variable_colors) % len(self.color_palette)
            self.variable_colors[variable] = self.color_palette[color_index]
        
        return self.variable_colors[variable]
    
    def get_variable_display_name(self, variable: str) -> str:
        """
        获取变量的显示名称
        
        Args:
            variable: 变量名
            
        Returns:
            显示名称
        """
        if variable.startswith('T_'):
            return f"气源 {variable}"
        elif variable.startswith('E_'):
            return f"分输点 {variable}"
        elif variable.startswith('B_'):
            return f"阀门 {variable}"
        elif variable.startswith('R_'):
            return f"调节器 {variable}"
        elif variable.startswith('C_'):
            return f"压缩机 {variable}"
        else:
            return f"其他 {variable}"
    
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
        
        # 为所有选择的变量添加轨迹
        for col in selected_columns:
            if col in df.columns:
                color = self.get_variable_color(col)
                display_name = self.get_variable_display_name(col)
                
                fig.add_trace(go.Scatter(
                    x=df['time_index'],
                    y=df[col],
                    mode='lines+markers',
                    name=display_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{col}</b><br>' +
                                 '时间点: %{x}<br>' +
                                 '数值: %{y:.4f}<extra></extra>'
                ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"{case_name} - 边界数据",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="时间点 (30分钟间隔)",
            yaxis_title="数值",
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
    
    def create_comparison_plot(self, df: pd.DataFrame, selected_variables: dict, case_name: str) -> go.Figure:
        """
        创建分类对比图
        
        Args:
            df: 数据框
            selected_variables: 按类别分类的选择变量字典
            case_name: 算例名称
            
        Returns:
            Plotly子图对象
        """
        # 计算需要的子图数量（只包含有数据的类别）
        categories_with_data = {k: v for k, v in selected_variables.items() if v}
        num_subplots = len(categories_with_data)
        
        if num_subplots == 0:
            # 返回空图
            fig = go.Figure()
            fig.update_layout(title="没有选择任何变量")
            return fig
        
        category_titles = {
            'gas_sources': '气源(T系列)',
            'distribution_points': '分输点(E系列)',
            'valves': '阀门(B系列)',
            'regulators': '调节器(R系列)',
            'compressors': '压缩机(C系列)'
        }
        
        subplot_titles = [f'{case_name} - {category_titles.get(cat, cat)}' 
                         for cat in categories_with_data.keys()]
        
        fig = make_subplots(
            rows=num_subplots, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15 / max(1, num_subplots-1) if num_subplots > 1 else 0
        )
        
        # 为每个类别添加数据
        row = 1
        for category, variables in categories_with_data.items():
            for var in variables:
                if var in df.columns:
                    color = self.get_variable_color(var)
                    fig.add_trace(go.Scatter(
                        x=df['time_index'],
                        y=df[var],
                        mode='lines+markers',
                        name=var,
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        showlegend=True,
                        hovertemplate=f'<b>{var}</b><br>' +
                                     '时间点: %{x}<br>' +
                                     '数值: %{y:.4f}<extra></extra>'
                    ), row=row, col=1)
            row += 1
        
        # 更新布局
        fig.update_layout(
            height=400 * num_subplots,
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
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', 
                        title_text="时间点 (30分钟间隔)", row=num_subplots, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', title_text="数值")
        
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
    
    def apply_predefined_scheme(self, scheme_name: str, df: pd.DataFrame):
        """
        应用预定义的变量方案
        
        Args:
            scheme_name: 方案名称
            df: 数据框
            
        Returns:
            dict: 按类别分组的选择变量
        """
        if scheme_name not in self.predefined_schemes:
            return {}
        
        scheme_variables = self.predefined_schemes[scheme_name]['variables']
        
        # 特殊处理"全部气源"方案
        if scheme_name == "方案3 - 全部气源":
            scheme_variables = self.processor.gas_sources
        
        # 按类别分组变量
        categories = self.processor.get_variable_categories()
        selected_vars = {category: [] for category in categories.keys()}
        
        for var in scheme_variables:
            if var in df.columns:
                for category, category_vars in categories.items():
                    if var in category_vars:
                        selected_vars[category].append(var)
                        break
        
        return selected_vars


def main():
    """主应用函数"""
    st.set_page_config(
        page_title="边界数据可视化 - 动态变量支持",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ 管网边界数据可视化 - 动态变量支持")
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
                # 获取变量类别
                categories = app.processor.get_variable_categories()
                
                # 预定义方案选择
                st.subheader("🎯 预定义方案")
                
                scheme_options = ["手动选择"] + list(app.predefined_schemes.keys())
                selected_scheme = st.selectbox(
                    "选择预设方案",
                    scheme_options,
                    help="选择预定义的变量组合方案，或选择'手动选择'自定义变量"
                )
                
                # 显示方案描述
                if selected_scheme != "手动选择":
                    scheme_info = app.predefined_schemes[selected_scheme]
                    # 获取当前方案的变量（处理动态方案）
                    if selected_scheme == "方案3 - 全部气源":
                        display_vars = categories['gas_sources']
                    else:
                        display_vars = scheme_info['variables']
                    
                    st.info(f"**{selected_scheme}**\n\n{scheme_info['description']}\n\n变量: {', '.join(display_vars[:8])}{'...' if len(display_vars) > 8 else ''}")
                    
                    # 应用预定义方案
                    if st.button("应用方案", key="apply_scheme"):
                        scheme_selected = app.apply_predefined_scheme(selected_scheme, df)
                        
                        # 初始化session state
                        for category, variables in categories.items():
                            key = f"{category}_visibility"
                            st.session_state[key] = {
                                var: var in scheme_selected.get(category, []) 
                                for var in variables
                            }
                        st.rerun()
                
                st.markdown("---")
                
                # 动态变量选择部分
                st.subheader("🔧 动态变量选择")
                
                # 初始化session state
                for category, variables in categories.items():
                    key = f"{category}_visibility"
                    if key not in st.session_state:
                        # 默认选择策略：气源全选，分输点选前8个，其他类型选前3个
                        if category == 'gas_sources':
                            st.session_state[key] = {var: True for var in variables}
                        elif category == 'distribution_points':
                            st.session_state[key] = {var: i < 8 for i, var in enumerate(variables)}
                        else:
                            st.session_state[key] = {var: i < 3 for i, var in enumerate(variables)}
                
                # 为每个类别创建选择界面
                category_names = {
                    'gas_sources': '气源变量 (T系列)',
                    'distribution_points': '分输点变量 (E系列)', 
                    'valves': '阀门变量 (B系列)',
                    'regulators': '调节器变量 (R系列)',
                    'compressors': '压缩机变量 (C系列)'
                }
                
                selected_vars_by_category = {}
                
                for category, variables in categories.items():
                    if not variables:  # 跳过空类别
                        continue
                        
                    st.write(f"**{category_names.get(category, category)}**")
                    
                    visibility_key = f"{category}_visibility"
                    
                    # 全选/全不选按钮
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"全选", key=f"select_all_{category}"):
                            st.session_state[visibility_key] = {var: True for var in variables}
                            st.rerun()
                    with col2:
                        if st.button(f"全不选", key=f"deselect_all_{category}"):
                            st.session_state[visibility_key] = {var: False for var in variables}
                            st.rerun()
                    
                    # 变量按钮
                    num_cols = 3 if category in ['valves', 'regulators', 'compressors'] else 2
                    var_cols = st.columns(num_cols)
                    
                    for i, var in enumerate(variables):
                        with var_cols[i % num_cols]:
                            current_state = st.session_state[visibility_key].get(var, False)
                            display_name = var.split(':')[0] if ':' in var else var
                            
                            if st.button(f"{'✓' if current_state else '✗'} {display_name}", 
                                       key=f"{category}_{var}", 
                                       help=f"点击切换 {var} 的显示状态"):
                                st.session_state[visibility_key][var] = not current_state
                                st.rerun()
                    
                    # 收集选中的变量
                    selected_vars_by_category[category] = [
                        var for var in variables 
                        if st.session_state[visibility_key].get(var, False)
                    ]
                    
                    # 显示选中数量
                    st.metric(f"{category_names.get(category, category)}选中", 
                             f"{len(selected_vars_by_category[category])}/{len(variables)}")
                    
                    st.markdown("---")
                
                # 用户自定义变量输入
                st.subheader("➕ 自定义变量")
                
                custom_var = st.text_input(
                    "输入变量名",
                    placeholder="例如: B_244:FR 或 R_005:ST",
                    help="输入新的变量名，必须存在于数据中"
                )
                
                if custom_var and custom_var not in app.processor.all_variables:
                    if custom_var in df.columns:
                        if st.button("添加自定义变量", key="add_custom"):
                            # 确定变量类别
                            if custom_var.startswith('T_'):
                                category = 'gas_sources'
                            elif custom_var.startswith('E_'):
                                category = 'distribution_points'
                            elif custom_var.startswith('B_'):
                                category = 'valves'
                            elif custom_var.startswith('R_'):
                                category = 'regulators'
                            elif custom_var.startswith('C_'):
                                category = 'compressors'
                            else:
                                category = 'gas_sources'  # 默认分类
                            
                            # 添加到处理器
                            getattr(app.processor, category).append(custom_var)
                            app.processor.all_variables.append(custom_var)
                            
                            # 添加到session state
                            visibility_key = f"{category}_visibility"
                            if visibility_key not in st.session_state:
                                st.session_state[visibility_key] = {}
                            st.session_state[visibility_key][custom_var] = True
                            
                            st.success(f"已添加变量 {custom_var} 到 {category}")
                            st.rerun()
                    else:
                        st.error(f"变量 {custom_var} 不存在于数据中")
                elif custom_var:
                    st.warning(f"变量 {custom_var} 已存在")
                
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
                
                total_vars = sum(len(vars) for vars in categories.values())
                total_selected = sum(len(vars) for vars in selected_vars_by_category.values())
                st.metric("变量选择", f"{total_selected}/{total_vars}")
                
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
        # 收集所有选中的变量
        all_selected = []
        for category_vars in selected_vars_by_category.values():
            all_selected.extend(category_vars)
        
        # 显示基本信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="选择算例",
                value=selected_case,
            )
        
        with col2:
            available_vars = sum(len(vars) for vars in categories.values())
            st.metric(
                label="可用变量",
                value=f"{available_vars}",
            )
        
        with col3:
            selected_count = len(all_selected)
            st.metric(
                label="已选变量",
                value=f"{selected_count}",
            )
        
        # 数据预览
        with st.expander("📈 数据预览", expanded=False):
            if all_selected:
                preview_columns = ['TIME', 'time_index'] + all_selected
                preview_df = df[preview_columns].head(10)
                st.dataframe(preview_df, width='stretch')
            else:
                st.info("请选择变量以查看数据预览")
        
        # 主要可视化
        if all_selected:
            st.subheader(f"📊 {selected_case} - 边界数据可视化")
            
            if viz_type == "统一折线图":
                # 统一折线图
                fig = app.create_line_plot(df, all_selected, selected_case)
                st.plotly_chart(fig, width='stretch')
                
            else:
                # 分类对比图
                fig = app.create_comparison_plot(df, selected_vars_by_category, selected_case)
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
            🔬 管网边界数据可视化工具 - 支持动态变量和自定义输入
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()