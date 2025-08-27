#!/usr/bin/env python3
"""
天然气管网数值模拟主程序
数据与机理融合的天然气管网数值模拟

使用方法:
    python main.py --mode train --config config/default.yaml
    python main.py --mode test --model_path checkpoints/best_model.pth
    python main.py --mode dashboard
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入模块
from models.utils import set_random_seed, get_device, count_parameters
from models.neural_network import FluidNet, PhysicsInformedNet, AttentionFluidNet
from data.loader import create_data_loaders
from training.trainer import Trainer
from training.loss import FluidLoss, PhysicsLoss, CombinedLoss
from training.optimizer import create_optimizer, create_scheduler
from inference.predictor import FluidPredictor
from inference.evaluator import ModelEvaluator
from visualization.plotter import TrainingPlotter, ResultsPlotter
from visualization.dashboard import create_dashboard

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fluid_model.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"已加载配置文件: {config_path}")
    return config


def get_default_config() -> dict:
    """获取默认配置"""
    return {
        'data': {
            'data_dir': 'data',
            'batch_size': 8,
            'num_workers': 4,
            'validation_split': 0.1
        },
        'model': {
            'type': 'FluidNet',  # FluidNet, PhysicsInformedNet, AttentionFluidNet
            'hidden_dims': [512, 256, 128],
            'dropout_rate': 0.1,
            'activation': 'relu'
        },
        'training': {
            'epochs': 100,
            'optimizer': {
                'type': 'adamw',
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'cosine',
                'T_max': 100
            },
            'loss': {
                'type': 'combined',  # fluid, physics, combined
                'physics_weight': 0.1
            },
            'early_stopping': {
                'patience': 15,
                'min_delta': 1e-6
            }
        },
        'paths': {
            'checkpoints_dir': 'checkpoints',
            'logs_dir': 'logs',
            'results_dir': 'results'
        },
        'device': 'auto',  # auto, cpu, cuda
        'random_seed': 42
    }


def create_model(config: dict, input_dim: int, output_dims: dict) -> object:
    """创建模型"""
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == 'FluidNet':
        model = FluidNet(
            input_dim=input_dim,
            output_dims=output_dims,
            hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            activation=model_config.get('activation', 'relu')
        )
    elif model_type == 'PhysicsInformedNet':
        model = PhysicsInformedNet(
            input_dim=input_dim,
            output_dims=output_dims,
            hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            activation=model_config.get('activation', 'tanh'),
            physics_weight=model_config.get('physics_weight', 0.1)
        )
    elif model_type == 'AttentionFluidNet':
        model = AttentionFluidNet(
            input_dim=input_dim,
            output_dims=output_dims,
            hidden_dim=model_config.get('hidden_dim', 256),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 4),
            dropout_rate=model_config.get('dropout_rate', 0.1)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    logger.info(f"创建了 {model_type} 模型")
    return model


def create_loss_function(config: dict) -> object:
    """创建损失函数"""
    loss_config = config['training']['loss']
    loss_type = loss_config['type']
    
    if loss_type == 'fluid':
        return FluidLoss(
            output_weights=loss_config.get('output_weights', None),
            loss_type=loss_config.get('loss_type', 'mse')
        )
    elif loss_type == 'physics':
        return PhysicsLoss(
            physics_weight=loss_config.get('physics_weight', 1.0)
        )
    elif loss_type == 'combined':
        data_loss = FluidLoss(
            output_weights=loss_config.get('output_weights', None),
            loss_type=loss_config.get('loss_type', 'mse')
        )
        physics_loss = PhysicsLoss(
            physics_weight=1.0
        )
        return CombinedLoss(
            data_loss=data_loss,
            physics_loss=physics_loss,
            physics_weight=loss_config.get('physics_weight', 0.1)
        )
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")


def train_model(config: dict):
    """训练模型"""
    logger.info("开始训练模型...")
    
    # 设置随机种子
    set_random_seed(config['random_seed'])
    
    # 获取设备
    if config['device'] == 'auto':
        device = get_device()
    else:
        device = config['device']
    
    # 创建数据加载器
    data_config = config['data']
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_config['data_dir'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        validation_split=data_config['validation_split']
    )
    
    # 获取输入输出维度
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    
    # 从数据集获取输出维度
    output_dims = train_loader.dataset.get_output_dims()
    
    logger.info(f"输入维度: {input_dim}")
    logger.info(f"输出维度: {output_dims}")
    
    # 创建模型
    model = create_model(config, input_dim, output_dims)
    count_parameters(model)
    
    # 创建损失函数
    criterion = create_loss_function(config)
    
    # 创建优化器和调度器
    optimizer_config = config['training']['optimizer']
    optimizer = create_optimizer(
        model=model,
        optimizer_type=optimizer_config['type'],
        learning_rate=optimizer_config['learning_rate'],
        weight_decay=optimizer_config['weight_decay']
    )
    
    scheduler_config = config['training']['scheduler']
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_config['type'],
        **{k: v for k, v in scheduler_config.items() if k != 'type'}
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config['paths']['checkpoints_dir'],
        patience=config['training']['early_stopping']['patience']
    )
    
    # 开始训练
    history = trainer.train(
        epochs=config['training']['epochs'],
        log_interval=10,
        save_interval=20
    )
    
    # 绘制训练曲线
    plotter = TrainingPlotter(save_dir=config['paths']['logs_dir'])
    plotter.plot_training_curves(
        train_losses=history['train_losses'],
        val_losses=history['val_losses'],
        learning_rates=history.get('learning_rates', None)
    )
    
    logger.info("训练完成!")


def test_model(config: dict, model_path: str):
    """测试模型"""
    logger.info("开始测试模型...")
    
    # 设置随机种子
    set_random_seed(config['random_seed'])
    
    # 获取设备
    if config['device'] == 'auto':
        device = get_device()
    else:
        device = config['device']
    
    # 创建数据加载器
    data_config = config['data']
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_config['data_dir'],
        batch_size=1,  # 测试时使用批次大小为1
        num_workers=data_config['num_workers'],
        validation_split=0
    )
    
    # 获取输入输出维度
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    output_dims = train_loader.dataset.get_output_dims()
    
    # 创建模型
    model = create_model(config, input_dim, output_dims)
    
    # 创建预测器
    predictor = FluidPredictor(
        model=model,
        model_path=model_path,
        device=device
    )
    
    # 进行预测
    test_data_dir = os.path.join(data_config['data_dir'], 'dataset', 'test')
    output_dir = config['paths']['results_dir']
    
    predictor.predict_from_files(test_data_dir, output_dir)
    
    logger.info(f"预测结果已保存到: {output_dir}")
    
    # 如果有验证数据，进行评估
    val_data_dir = os.path.join(data_config['data_dir'], 'dataset', 'train')
    if os.path.exists(val_data_dir):
        logger.info("开始评估模型性能...")
        
        evaluator = ModelEvaluator(output_dir='evaluation_results')
        
        # 选择部分训练数据作为验证
        val_cases = [f"第{i:03d}个算例" for i in range(240, 264)]  # 使用最后24个训练案例作为验证
        
        # 创建临时验证目录
        temp_val_dir = 'temp_validation'
        os.makedirs(temp_val_dir, exist_ok=True)
        
        for case in val_cases:
            case_dir = os.path.join(val_data_dir, case)
            if os.path.exists(case_dir):
                import shutil
                shutil.copytree(case_dir, os.path.join(temp_val_dir, case), 
                              dirs_exist_ok=True)
        
        # 对验证数据进行预测
        predictor.predict_from_files(temp_val_dir, 'temp_predictions')
        
        # 评估预测结果
        metrics = evaluator.create_comprehensive_report(
            predictions_dir='temp_predictions',
            ground_truth_dir=temp_val_dir
        )
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_val_dir)
        shutil.rmtree('temp_predictions')
        
        logger.info("评估完成!")


def run_dashboard():
    """运行可视化仪表板"""
    logger.info("启动可视化仪表板...")
    
    try:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", str(project_root / "visualization" / "dashboard.py")]
        stcli.main()
    except ImportError:
        logger.error("Streamlit未安装，无法启动仪表板")
        logger.info("请安装Streamlit: pip install streamlit")
    except Exception as e:
        logger.error(f"启动仪表板失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="天然气管网数值模拟")
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'test', 'dashboard'],
                       help='运行模式: train/test/dashboard')
    
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='配置文件路径')
    
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/best_model.pth',
                       help='模型文件路径 (测试模式)')
    
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据目录路径')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录路径')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='计算设备')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖命令行参数
    if args.data_dir != 'data':
        config['data']['data_dir'] = args.data_dir
    if args.output_dir != 'results':
        config['paths']['results_dir'] = args.output_dir
    if args.device != 'auto':
        config['device'] = args.device
    if args.seed != 42:
        config['random_seed'] = args.seed
    
    # 创建必要的目录
    os.makedirs(config['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    # 根据模式执行相应功能
    if args.mode == 'train':
        train_model(config)
    
    elif args.mode == 'test':
        if not os.path.exists(args.model_path):
            logger.error(f"模型文件不存在: {args.model_path}")
            sys.exit(1)
        test_model(config, args.model_path)
    
    elif args.mode == 'dashboard':
        run_dashboard()


if __name__ == "__main__":
    main()