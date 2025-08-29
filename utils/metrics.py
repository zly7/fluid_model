import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_mse(predictions: Union[torch.Tensor, np.ndarray], 
                 targets: Union[torch.Tensor, np.ndarray],
                 reduce_mean: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        reduce_mean: Whether to average over all dimensions
        
    Returns:
        MSE value(s)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    mse = (predictions - targets) ** 2
    
    if reduce_mean:
        return np.mean(mse)
    else:
        return np.mean(mse, axis=(0, 1)) if mse.ndim > 2 else np.mean(mse, axis=0)

def calculate_mae(predictions: Union[torch.Tensor, np.ndarray],
                 targets: Union[torch.Tensor, np.ndarray], 
                 reduce_mean: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        reduce_mean: Whether to average over all dimensions
        
    Returns:
        MAE value(s)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    mae = np.abs(predictions - targets)
    
    if reduce_mean:
        return np.mean(mae)
    else:
        return np.mean(mae, axis=(0, 1)) if mae.ndim > 2 else np.mean(mae, axis=0)

def calculate_mape(predictions: Union[torch.Tensor, np.ndarray],
                  targets: Union[torch.Tensor, np.ndarray],
                  epsilon: float = 1e-8,
                  reduce_mean: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        epsilon: Small value to avoid division by zero
        reduce_mean: Whether to average over all dimensions
        
    Returns:
        MAPE value(s)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Avoid division by zero
    targets_safe = np.where(np.abs(targets) < epsilon, epsilon, targets)
    
    mape = np.abs((predictions - targets) / targets_safe) * 100
    
    if reduce_mean:
        return np.mean(mape)
    else:
        return np.mean(mape, axis=(0, 1)) if mape.ndim > 2 else np.mean(mape, axis=0)

def calculate_rmse(predictions: Union[torch.Tensor, np.ndarray],
                  targets: Union[torch.Tensor, np.ndarray],
                  reduce_mean: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        reduce_mean: Whether to average over all dimensions
        
    Returns:
        RMSE value(s)
    """
    mse = calculate_mse(predictions, targets, reduce_mean)
    
    if isinstance(mse, np.ndarray):
        return np.sqrt(mse)
    else:
        return np.sqrt(mse)

def calculate_r2_score(predictions: Union[torch.Tensor, np.ndarray],
                      targets: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        R² score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Flatten for sklearn
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(predictions_flat) | np.isnan(targets_flat))
    predictions_clean = predictions_flat[mask]
    targets_clean = targets_flat[mask]
    
    if len(predictions_clean) == 0:
        return 0.0
    
    return r2_score(targets_clean, predictions_clean)

class EquipmentMetrics:
    """
    Calculate metrics for specific equipment types.
    """
    
    def __init__(self, equipment_dims: Dict[str, int]):
        """
        Initialize equipment metrics calculator.
        
        Args:
            equipment_dims: Dictionary mapping equipment type to dimension count
        """
        self.equipment_dims = equipment_dims
        self._calculate_indices()
    
    def _calculate_indices(self):
        """Calculate start/end indices for each equipment type."""
        self.equipment_indices = {}
        start_idx = 0
        
        # Follow consistent ordering
        equipment_order = ['B', 'C', 'H', 'N', 'P', 'R', 'T&E']
        
        for eq_type in equipment_order:
            if eq_type in self.equipment_dims:
                end_idx = start_idx + self.equipment_dims[eq_type]
                self.equipment_indices[eq_type] = (start_idx, end_idx)
                start_idx = end_idx
    
    def calculate_equipment_mse(self, predictions: Union[torch.Tensor, np.ndarray],
                               targets: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        Calculate MSE for each equipment type.
        
        Args:
            predictions: Model predictions (batch_size, seq_len, total_dims)
            targets: Ground truth targets (batch_size, seq_len, total_dims)
            
        Returns:
            Dictionary mapping equipment type to MSE
        """
        equipment_mse = {}
        
        for eq_type, (start_idx, end_idx) in self.equipment_indices.items():
            pred_eq = predictions[..., start_idx:end_idx]
            target_eq = targets[..., start_idx:end_idx]
            
            equipment_mse[eq_type] = calculate_mse(pred_eq, target_eq, reduce_mean=True)
        
        # Overall MSE
        equipment_mse['overall'] = calculate_mse(predictions, targets, reduce_mean=True)
        
        return equipment_mse
    
    def calculate_equipment_mae(self, predictions: Union[torch.Tensor, np.ndarray],
                               targets: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        Calculate MAE for each equipment type.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary mapping equipment type to MAE
        """
        equipment_mae = {}
        
        for eq_type, (start_idx, end_idx) in self.equipment_indices.items():
            pred_eq = predictions[..., start_idx:end_idx]
            target_eq = targets[..., start_idx:end_idx]
            
            equipment_mae[eq_type] = calculate_mae(pred_eq, target_eq, reduce_mean=True)
        
        # Overall MAE
        equipment_mae['overall'] = calculate_mae(predictions, targets, reduce_mean=True)
        
        return equipment_mae
    
    def calculate_equipment_r2(self, predictions: Union[torch.Tensor, np.ndarray],
                              targets: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        Calculate R² for each equipment type.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary mapping equipment type to R²
        """
        equipment_r2 = {}
        
        for eq_type, (start_idx, end_idx) in self.equipment_indices.items():
            pred_eq = predictions[..., start_idx:end_idx]
            target_eq = targets[..., start_idx:end_idx]
            
            equipment_r2[eq_type] = calculate_r2_score(pred_eq, target_eq)
        
        # Overall R²
        equipment_r2['overall'] = calculate_r2_score(predictions, targets)
        
        return equipment_r2

class FluidDynamicsEvaluator:
    """
    Comprehensive evaluator for fluid dynamics predictions.
    
    Provides detailed evaluation including:
    - Standard regression metrics (MSE, MAE, RMSE, R²)
    - Equipment-specific metrics
    - Temporal analysis
    - Physical consistency checks
    """
    
    def __init__(self, equipment_dims: Dict[str, int]):
        """
        Initialize evaluator.
        
        Args:
            equipment_dims: Equipment type dimensions
        """
        self.equipment_dims = equipment_dims
        self.equipment_metrics = EquipmentMetrics(equipment_dims)
    
    def evaluate_predictions(self, 
                           predictions: Union[torch.Tensor, np.ndarray],
                           targets: Union[torch.Tensor, np.ndarray],
                           return_detailed: bool = True) -> Dict:
        """
        Comprehensive evaluation of predictions.
        
        Args:
            predictions: Model predictions (batch_size, seq_len, total_dims)
            targets: Ground truth targets (batch_size, seq_len, total_dims)
            return_detailed: Whether to return detailed equipment-wise metrics
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Overall metrics
        results['overall'] = {
            'mse': calculate_mse(predictions, targets),
            'mae': calculate_mae(predictions, targets),
            'rmse': calculate_rmse(predictions, targets),
            'mape': calculate_mape(predictions, targets),
            'r2': calculate_r2_score(predictions, targets)
        }
        
        # Equipment-specific metrics
        if return_detailed:
            results['equipment_mse'] = self.equipment_metrics.calculate_equipment_mse(predictions, targets)
            results['equipment_mae'] = self.equipment_metrics.calculate_equipment_mae(predictions, targets)
            results['equipment_r2'] = self.equipment_metrics.calculate_equipment_r2(predictions, targets)
        
        # Temporal analysis
        results['temporal'] = self._temporal_analysis(predictions, targets)
        
        # Physical consistency
        results['physical'] = self._physical_consistency_check(predictions)
        
        return results
    
    def _temporal_analysis(self, predictions: Union[torch.Tensor, np.ndarray],
                          targets: Union[torch.Tensor, np.ndarray]) -> Dict:
        """
        Analyze temporal properties of predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Temporal analysis results
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        temporal_results = {}
        
        if predictions.ndim >= 2 and predictions.shape[1] > 1:
            # Temporal smoothness (variance in temporal differences)
            pred_diff = np.diff(predictions, axis=1)
            target_diff = np.diff(targets, axis=1)
            
            temporal_results['smoothness_error'] = np.mean((pred_diff - target_diff) ** 2)
            temporal_results['prediction_smoothness'] = np.mean(np.var(pred_diff, axis=1))
            temporal_results['target_smoothness'] = np.mean(np.var(target_diff, axis=1))
            
            # Early vs late prediction accuracy
            seq_len = predictions.shape[1]
            early_len = seq_len // 3
            late_start = 2 * seq_len // 3
            
            early_mse = calculate_mse(predictions[:, :early_len], targets[:, :early_len])
            late_mse = calculate_mse(predictions[:, late_start:], targets[:, late_start:])
            
            temporal_results['early_mse'] = early_mse
            temporal_results['late_mse'] = late_mse
            temporal_results['mse_degradation'] = late_mse - early_mse
        
        return temporal_results
    
    def _physical_consistency_check(self, predictions: Union[torch.Tensor, np.ndarray]) -> Dict:
        """
        Check physical consistency of predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Physical consistency metrics
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        physical_results = {}
        
        # Basic physical constraints
        physical_results['negative_values_ratio'] = np.mean(predictions < 0)
        physical_results['extreme_values_ratio'] = np.mean(np.abs(predictions) > 1000)  # Assuming normalized data
        
        # Temporal consistency (no sudden jumps)
        if predictions.ndim >= 2 and predictions.shape[1] > 1:
            temporal_diff = np.diff(predictions, axis=1)
            
            # Define "sudden jump" as change > 3 standard deviations
            std_threshold = 3 * np.std(temporal_diff)
            sudden_jumps = np.mean(np.abs(temporal_diff) > std_threshold)
            
            physical_results['sudden_jumps_ratio'] = sudden_jumps
        
        return physical_results
    
    def compare_models(self, 
                      predictions_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
                      targets: Union[torch.Tensor, np.ndarray]) -> Dict:
        """
        Compare multiple model predictions.
        
        Args:
            predictions_dict: Dictionary mapping model name to predictions
            targets: Ground truth targets
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for model_name, predictions in predictions_dict.items():
            comparison_results[model_name] = self.evaluate_predictions(
                predictions, targets, return_detailed=False
            )
        
        # Ranking
        model_names = list(predictions_dict.keys())
        overall_mse = [comparison_results[name]['overall']['mse'] for name in model_names]
        
        # Sort by MSE (ascending)
        sorted_indices = np.argsort(overall_mse)
        comparison_results['ranking'] = {
            'by_mse': [model_names[i] for i in sorted_indices],
            'mse_values': [overall_mse[i] for i in sorted_indices]
        }
        
        return comparison_results
    
    def generate_report(self, 
                       predictions: Union[torch.Tensor, np.ndarray],
                       targets: Union[torch.Tensor, np.ndarray],
                       model_name: str = "FluidDynamicsTransformer") -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model_name: Name of the model being evaluated
            
        Returns:
            Formatted evaluation report
        """
        results = self.evaluate_predictions(predictions, targets, return_detailed=True)
        
        report = f"=== {model_name} Evaluation Report ===\n\n"
        
        # Overall metrics
        overall = results['overall']
        report += "Overall Performance:\n"
        report += f"  MSE:  {overall['mse']:.6f}\n"
        report += f"  MAE:  {overall['mae']:.6f}\n"
        report += f"  RMSE: {overall['rmse']:.6f}\n"
        report += f"  MAPE: {overall['mape']:.2f}%\n"
        report += f"  R²:   {overall['r2']:.4f}\n\n"
        
        # Equipment-specific metrics
        if 'equipment_mse' in results:
            report += "Equipment-Specific MSE:\n"
            for eq_type, mse in results['equipment_mse'].items():
                if eq_type != 'overall':
                    report += f"  {eq_type}: {mse:.6f}\n"
            report += "\n"
        
        # Temporal analysis
        if results['temporal']:
            temporal = results['temporal']
            report += "Temporal Analysis:\n"
            if 'early_mse' in temporal:
                report += f"  Early MSE:        {temporal['early_mse']:.6f}\n"
                report += f"  Late MSE:         {temporal['late_mse']:.6f}\n"
                report += f"  MSE Degradation:  {temporal['mse_degradation']:.6f}\n"
            report += "\n"
        
        # Physical consistency
        if results['physical']:
            physical = results['physical']
            report += "Physical Consistency:\n"
            report += f"  Negative values:  {physical['negative_values_ratio']:.2%}\n"
            report += f"  Extreme values:   {physical['extreme_values_ratio']:.2%}\n"
            if 'sudden_jumps_ratio' in physical:
                report += f"  Sudden jumps:     {physical['sudden_jumps_ratio']:.2%}\n"
        
        return report