import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid")
sns.set_palette("husl")


class TrainingPlotter:
    """训练过程可视化"""
    
    def __init__(self, save_dir: str = 'plots'):
        """
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: Optional[List[float]] = None,
                           learning_rates: Optional[List[float]] = None,
                           save_name: str = 'training_curves.png') -> None:
        """绘制训练曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            learning_rates: 学习率列表
            save_name: 保存文件名
        """
        fig_height = 10 if learning_rates else 6
        fig, axes = plt.subplots(2 if learning_rates else 1, 1, 
                               figsize=(10, fig_height))
        
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        # 损失曲线
        epochs = range(1, len(train_losses) + 1)
        
        axes[0].plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        
        if val_losses and len(val_losses) > 0:
            val_epochs = range(1, len(val_losses) + 1)
            axes[0].plot(val_epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        
        axes[0].set_xlabel('轮次 (Epoch)')
        axes[0].set_ylabel('损失 (Loss)')
        axes[0].set_title('训练损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # 使用对数尺度
        
        # 学习率曲线
        if learning_rates:
            axes[1].plot(epochs[:len(learning_rates)], learning_rates, 
                        'g-', label='学习率', linewidth=2)
            axes[1].set_xlabel('轮次 (Epoch)')
            axes[1].set_ylabel('学习率 (Learning Rate)')
            axes[1].set_title('学习率变化曲线')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_component_losses(self,
                            component_losses: Dict[str, List[float]],
                            save_name: str = 'component_losses.png') -> None:
        """绘制各组件损失曲线
        
        Args:
            component_losses: 组件损失字典
            save_name: 保存文件名
        """
        n_components = len(component_losses)
        if n_components == 0:
            return
        
        cols = min(3, n_components)
        rows = (n_components + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        colors = sns.color_palette("husl", n_components)
        
        for i, (component_name, losses) in enumerate(component_losses.items()):
            if i >= len(axes):
                break
                
            epochs = range(1, len(losses) + 1)
            axes[i].plot(epochs, losses, color=colors[i], linewidth=2)
            axes[i].set_xlabel('轮次 (Epoch)')
            axes[i].set_ylabel('损失值')
            axes[i].set_title(f'{component_name} 损失')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')
        
        # 隐藏多余的子图
        for i in range(len(component_losses), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gradient_norms(self,
                          gradient_norms: List[float],
                          save_name: str = 'gradient_norms.png') -> None:
        """绘制梯度范数变化
        
        Args:
            gradient_norms: 梯度范数列表
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(gradient_norms) + 1)
        ax.plot(epochs, gradient_norms, 'purple', linewidth=2)
        
        ax.set_xlabel('轮次 (Epoch)')
        ax.set_ylabel('梯度范数')
        ax.set_title('梯度范数变化')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()


class ResultsPlotter:
    """结果可视化"""
    
    def __init__(self, save_dir: str = 'plots'):
        """
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_time_series_comparison(self,
                                  predictions: Dict[str, np.ndarray],
                                  ground_truth: Dict[str, np.ndarray],
                                  time_points: Optional[List[str]] = None,
                                  save_name: str = 'time_series_comparison.png') -> None:
        """绘制时间序列对比图
        
        Args:
            predictions: 预测结果字典
            ground_truth: 真实值字典
            time_points: 时间点列表
            save_name: 保存文件名
        """
        n_outputs = len(predictions)
        if n_outputs == 0:
            return
        
        cols = min(2, n_outputs)
        rows = (n_outputs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 4*rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, output_name in enumerate(predictions.keys()):
            if i >= len(axes):
                break
            
            pred = predictions[output_name]
            true = ground_truth.get(output_name, None)
            
            # 如果是2D数组，取第一行或平均值
            if pred.ndim == 2:
                if pred.shape[0] == 1:
                    pred = pred[0]
                else:
                    pred = np.mean(pred, axis=0)
            
            if true is not None:
                if true.ndim == 2:
                    if true.shape[0] == 1:
                        true = true[0]
                    else:
                        true = np.mean(true, axis=0)
            
            # 创建时间轴
            if time_points is None:
                x_axis = range(len(pred))
                x_label = '时间步'
            else:
                x_axis = time_points[:len(pred)]
                x_label = '时间'
            
            ax = axes[i]
            
            # 绘制预测值
            ax.plot(x_axis, pred, 'b-', label='预测值', linewidth=2, alpha=0.8)
            
            # 绘制真实值
            if true is not None:
                ax.plot(x_axis, true, 'r--', label='真实值', linewidth=2, alpha=0.8)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel('数值')
            ax.set_title(f'{output_name} 时间序列对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 如果是时间点，旋转x轴标签
            if time_points is not None:
                ax.tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for i in range(len(predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_network_topology(self,
                            node_data: pd.DataFrame,
                            edge_data: pd.DataFrame,
                            save_name: str = 'network_topology.png') -> None:
        """绘制管网拓扑图
        
        Args:
            node_data: 节点数据
            edge_data: 边数据
            save_name: 保存文件名
        """
        try:
            import networkx as nx
        except ImportError:
            print("networkx is required for network topology visualization")
            return
        
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for _, row in node_data.iterrows():
            G.add_node(row['node_id'], 
                      pos=(row.get('x', 0), row.get('y', 0)),
                      node_type=row.get('type', 'N'))
        
        # 添加边
        for _, row in edge_data.iterrows():
            G.add_edge(row['from_node'], row['to_node'],
                      edge_type=row.get('type', 'P'))
        
        # 获取位置
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 根据节点类型设置颜色
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'N')
            if node_type == 'T':  # 气源
                node_colors.append('red')
                node_sizes.append(300)
            elif node_type == 'E':  # 分输点
                node_colors.append('blue')
                node_sizes.append(200)
            elif node_type == 'C':  # 压缩机
                node_colors.append('green')
                node_sizes.append(250)
            else:  # 普通节点
                node_colors.append('gray')
                node_sizes.append(100)
        
        # 绘制网络
        nx.draw(G, pos, ax=ax,
               node_color=node_colors,
               node_size=node_sizes,
               edge_color='black',
               width=1,
               alpha=0.7)
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='气源 (T)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=8, label='分输点 (E)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=9, label='压缩机 (C)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=6, label='节点 (N)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title('天然气管网拓扑图')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pressure_distribution(self,
                                 pressure_data: np.ndarray,
                                 node_positions: Optional[Dict] = None,
                                 save_name: str = 'pressure_distribution.png') -> None:
        """绘制压力分布图
        
        Args:
            pressure_data: 压力数据
            node_positions: 节点位置字典
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if node_positions:
            # 创建散点图
            positions = list(node_positions.values())
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            scatter = ax.scatter(x_coords, y_coords, 
                               c=pressure_data[:len(positions)], 
                               cmap='viridis', 
                               s=100, 
                               alpha=0.8)
            
            plt.colorbar(scatter, ax=ax, label='压力 (MPa)')
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('Y 坐标')
        else:
            # 简单的条形图
            ax.bar(range(len(pressure_data)), pressure_data)
            ax.set_xlabel('节点索引')
            ax.set_ylabel('压力 (MPa)')
        
        ax.set_title('管网压力分布')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_flow_distribution(self,
                             flow_data: np.ndarray,
                             save_name: str = 'flow_distribution.png') -> None:
        """绘制流量分布直方图
        
        Args:
            flow_data: 流量数据
            save_name: 保存文件名
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 流量分布直方图
        ax1.hist(flow_data.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('流量 (m³/h)')
        ax1.set_ylabel('频次')
        ax1.set_title('流量分布直方图')
        ax1.grid(True, alpha=0.3)
        
        # 流量箱线图
        ax2.boxplot(flow_data.flatten())
        ax2.set_ylabel('流量 (m³/h)')
        ax2.set_title('流量分布箱线图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self,
                              data_dict: Dict[str, np.ndarray],
                              save_name: str = 'correlation_matrix.png') -> None:
        """绘制相关性矩阵热图
        
        Args:
            data_dict: 数据字典
            save_name: 保存文件名
        """
        # 创建DataFrame
        df_data = {}
        for name, data in data_dict.items():
            if data.ndim == 1:
                df_data[name] = data
            else:
                # 对于2D数据，计算每列的平均值
                for i in range(data.shape[1]):
                    df_data[f'{name}_col_{i}'] = data[:, i]
        
        df = pd.DataFrame(df_data)
        
        # 计算相关性矩阵
        correlation_matrix = df.corr()
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   ax=ax,
                   fmt='.2f')
        
        ax.set_title('变量相关性矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self,
                               predictions: Dict[str, np.ndarray],
                               ground_truth: Dict[str, np.ndarray],
                               metrics: Dict[str, Dict[str, float]],
                               save_name: str = 'summary_dashboard.png') -> None:
        """创建总结仪表板
        
        Args:
            predictions: 预测结果
            ground_truth: 真实值
            metrics: 评估指标
            save_name: 保存文件名
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig)
        
        # 1. 整体指标条形图
        ax1 = fig.add_subplot(gs[0, :2])
        metric_names = ['MSE', 'MAE', 'R²']
        overall_mse = np.mean([m['mse'] for m in metrics.values()])
        overall_mae = np.mean([m['mae'] for m in metrics.values()])
        overall_r2 = np.mean([m['r2_score'] for m in metrics.values()])
        
        values = [overall_mse, overall_mae, overall_r2]
        bars = ax1.bar(metric_names, values, color=['red', 'orange', 'green'])
        ax1.set_title('整体性能指标')
        ax1.set_ylabel('指标值')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 2. 各组件R²对比
        ax2 = fig.add_subplot(gs[0, 2:])
        component_names = list(metrics.keys())
        r2_scores = [metrics[name]['r2_score'] for name in component_names]
        
        ax2.barh(component_names, r2_scores, color=sns.color_palette("viridis", len(component_names)))
        ax2.set_title('各组件 R² 分数')
        ax2.set_xlabel('R² 分数')
        
        # 3-4. 预测vs真实值散点图（选择前两个组件）
        for i, component_name in enumerate(list(predictions.keys())[:2]):
            ax = fig.add_subplot(gs[1, i*2:(i+1)*2])
            
            pred = predictions[component_name].flatten()
            true = ground_truth[component_name].flatten()
            
            # 随机采样减少点数
            if len(pred) > 5000:
                indices = np.random.choice(len(pred), 5000, replace=False)
                pred = pred[indices]
                true = true[indices]
            
            ax.scatter(true, pred, alpha=0.5, s=1)
            
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(f'{component_name}')
            
            # 添加R²
            r2 = metrics[component_name]['r2_score']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. 误差分布（选择一个组件）
        ax5 = fig.add_subplot(gs[2, :2])
        first_component = list(predictions.keys())[0]
        pred = predictions[first_component].flatten()
        true = ground_truth[first_component].flatten()
        errors = pred - true
        
        ax5.hist(errors, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', label='零误差线')
        ax5.set_xlabel('预测误差')
        ax5.set_ylabel('频次')
        ax5.set_title(f'{first_component} 误差分布')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 指标热图
        ax6 = fig.add_subplot(gs[2, 2:])
        metric_matrix = []
        metric_labels = ['MSE', 'MAE', 'R²', 'MAPE']
        
        for component in component_names:
            row = [
                metrics[component]['mse'],
                metrics[component]['mae'], 
                metrics[component]['r2_score'],
                metrics[component]['mape']
            ]
            metric_matrix.append(row)
        
        metric_matrix = np.array(metric_matrix)
        
        # 标准化每个指标列
        for j in range(metric_matrix.shape[1]):
            col = metric_matrix[:, j]
            metric_matrix[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-8)
        
        sns.heatmap(metric_matrix, 
                   xticklabels=metric_labels,
                   yticklabels=component_names,
                   annot=True,
                   cmap='RdYlGn_r',
                   ax=ax6,
                   fmt='.3f')
        ax6.set_title('标准化指标热图')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), 
                   dpi=300, bbox_inches='tight')
        plt.close()