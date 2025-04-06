"""
Specialized analyzer for machine learning model architectures.

This module provides tools for analyzing and profiling ML models from
frameworks like PyTorch and TensorFlow, estimating theoretical complexity,
and providing optimization recommendations.
"""

import logging
import time
from typing import Dict, List, Any, Union, Tuple, Optional
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        FVCORE_AVAILABLE = True
    except ImportError:
        FVCORE_AVAILABLE = False
        logger.warning("fvcore not available. FLOP counting will be limited.")
        
    try:
        import thop
        THOP_AVAILABLE = True
    except ImportError:
        THOP_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False
    FVCORE_AVAILABLE = False
    THOP_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Library to store historical profiling data for training recommendations
PROFILE_HISTORY_DIR = os.path.join(os.path.dirname(__file__), 'profile_history')
os.makedirs(PROFILE_HISTORY_DIR, exist_ok=True)

class ModelAnalyzer:
    """
    Specialized analyzer for machine learning models that provides detailed
    analysis of complexity, memory usage, and performance.
    """
    
    def __init__(self):
        """Initialize the model analyzer"""
        self.last_analysis = None
        self.framework = None
        self.optimization_history = self._load_optimization_history()
    
    def _load_optimization_history(self) -> Dict[str, Any]:
        """Load historical optimization data if available"""
        history_file = os.path.join(PROFILE_HISTORY_DIR, 'optimization_history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load optimization history: {e}")
        return {}
    
    def _save_profile_data(self, profile_data: Dict[str, Any], model_name: str):
        """Save profile data to history for future recommendation training"""
        filename = f"{model_name}_{int(time.time())}.json"
        filepath = os.path.join(PROFILE_HISTORY_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save profile data: {e}")
    
    def analyze_model(self, model: Any, input_shape: Optional[Tuple] = None, 
                      framework: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a machine learning model to determine its complexity, 
        memory usage, and performance characteristics.
        
        Args:
            model: Model object to analyze (PyTorch or TensorFlow)
            input_shape: Optional shape for input data
            framework: Optional framework name ('pytorch', 'tensorflow') if detection fails
            
        Returns:
            Dictionary with analysis results
        """
        # Auto-detect framework if not specified
        if framework:
            self.framework = framework.lower()
        else:
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                self.framework = 'pytorch'
            elif TF_AVAILABLE and hasattr(model, 'layers'):
                self.framework = 'tensorflow'
            else:
                return {'error': 'Unsupported model type or required libraries not available'}
        
        # Analyze based on framework
        if self.framework == 'pytorch':
            analysis = self._analyze_pytorch_model(model, input_shape)
        elif self.framework == 'tensorflow':
            analysis = self._analyze_tensorflow_model(model, input_shape)
        else:
            return {'error': f"Unsupported framework: {self.framework}"}
        
        # Add recommendations based on model architecture
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        # Save analysis for future reference
        self.last_analysis = analysis
        
        # Save profile data for training recommendation system
        if 'name' in analysis:
            self._save_profile_data(analysis, analysis['name'])
        
        return analysis
    
    def _analyze_pytorch_model(self, model: Any, input_shape: Optional[Tuple] = None) -> Dict[str, Any]:
        """Analyze PyTorch model architecture and performance"""
        model_info = {
            'type': 'pytorch',
            'name': model.__class__.__name__,
            'layers': [],
            'param_count': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'layer_stats': {},
            'memory_estimation': {},
            'performance': {}
        }
        
        # Identify architecture type
        if any('Conv' in m.__class__.__name__ for m in model.modules()):
            model_info['architecture_type'] = 'CNN'
        elif any('LSTM' in m.__class__.__name__ or 'GRU' in m.__class__.__name__ for m in model.modules()):
            model_info['architecture_type'] = 'RNN'
        elif any('MultiheadAttention' in m.__class__.__name__ or 'Transformer' in m.__class__.__name__ 
                for m in model.modules()):
            model_info['architecture_type'] = 'Transformer'
        else:
            model_info['architecture_type'] = 'MLP'
        
        # Create a dummy input
        if input_shape is None:
            if model_info['architecture_type'] == 'CNN':
                input_shape = (1, 3, 224, 224)  # Assume standard image input
            elif model_info['architecture_type'] == 'RNN':
                input_shape = (1, 128, 512)  # Assume (batch, seq_len, features)
            elif model_info['architecture_type'] == 'Transformer':
                input_shape = (1, 128, 768)  # Assume (batch, seq_len, embed_dim)
            else:
                input_shape = (1, 512)  # Assume (batch, features)
        
        dummy_input = torch.zeros(*input_shape)
        
        # Collect layer information with memory and parameter details
        for name, module in model.named_modules():
            if name == '':  # Skip the top-level module
                continue
                
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
                'params': sum(p.numel() for p in module.parameters()),
                'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'param_size_mb': sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 * 1024)
            }
            model_info['layers'].append(layer_info)
            
            # Group by layer type for statistics
            layer_type = module.__class__.__name__
            if layer_type not in model_info['layer_stats']:
                model_info['layer_stats'][layer_type] = {
                    'count': 0,
                    'total_params': 0,
                    'total_size_mb': 0
                }
            
            model_info['layer_stats'][layer_type]['count'] += 1
            model_info['layer_stats'][layer_type]['total_params'] += layer_info['params']
            model_info['layer_stats'][layer_type]['total_size_mb'] += layer_info['param_size_mb']
        
        # Estimate activation memory (if possible)
        try:
            # Hook to capture output sizes
            output_sizes = []
            
            def hook_fn(module, input, output):
                output_sizes.append({
                    'module': module.__class__.__name__,
                    'output_size': output.size(),
                    'memory_mb': output.nelement() * output.element_size() / (1024 * 1024)
                })
            
            hooks = []
            for name, module in model.named_modules():
                if not list(module.children()):  # Only leaf modules
                    hooks.append(module.register_forward_hook(hook_fn))
            
            # Forward pass to collect activation sizes
            with torch.no_grad():
                model.eval()
                _ = model(dummy_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Calculate total activation memory
            total_activation_mb = sum(size_info['memory_mb'] for size_info in output_sizes)
            model_info['memory_estimation']['activations_mb'] = total_activation_mb
            model_info['memory_estimation']['parameters_mb'] = sum(layer['param_size_mb'] for layer in model_info['layers'])
            model_info['memory_estimation']['total_mb'] = total_activation_mb + model_info['memory_estimation']['parameters_mb']
            model_info['memory_estimation']['activation_details'] = output_sizes
            
        except Exception as e:
            logger.warning(f"Error estimating activation memory: {e}")
        
        # Estimate FLOPs and performance
        if FVCORE_AVAILABLE:
            try:
                flops = FlopCountAnalysis(model, dummy_input)
                flops_count = flops.total()
                model_info['performance']['flops'] = flops_count
                model_info['performance']['flops_by_operator'] = flops.by_operator()
                model_info['performance']['flops_by_module'] = flops.by_module()
                
                # Theoretical estimation of throughput
                # Assuming ~10 TFLOPS for a modern GPU (very rough estimate)
                theoretical_tflops = 10  # 10 trillion FLOP/s
                model_info['performance']['theoretical_batches_per_second'] = theoretical_tflops * 1e12 / flops_count
                
            except Exception as e:
                logger.warning(f"Error estimating FLOPs: {e}")
        
        elif THOP_AVAILABLE:
            try:
                macs, params = thop.profile(model, inputs=(dummy_input,))
                model_info['performance']['macs'] = macs
                model_info['performance']['flops'] = macs * 2  # Approximate FLOPs as 2 * MACs
            except Exception as e:
                logger.warning(f"Error estimating MACs: {e}")
        
        # Check for architecture-specific properties
        if model_info['architecture_type'] == 'Transformer':
            # Look for attention mechanism details
            attention_heads = 0
            embed_dim = 0
            
            for m in model.modules():
                if 'MultiheadAttention' in m.__class__.__name__:
                    if hasattr(m, 'num_heads'):
                        attention_heads = max(attention_heads, m.num_heads)
                    if hasattr(m, 'embed_dim'):
                        embed_dim = max(embed_dim, m.embed_dim)
            
            if attention_heads > 0:
                model_info['transformer_details'] = {
                    'attention_heads': attention_heads,
                    'embedding_dimension': embed_dim
                }
        
        return model_info
    
    def _analyze_tensorflow_model(self, model: Any, input_shape: Optional[Tuple] = None) -> Dict[str, Any]:
        """Analyze TensorFlow model architecture and performance"""
        model_info = {
            'type': 'tensorflow',
            'name': model.__class__.__name__,
            'layers': [],
            'param_count': model.count_params(),
            'trainable_params': sum(w.numpy().size for w in model.trainable_weights),
            'layer_stats': {},
            'memory_estimation': {},
            'performance': {}
        }
        
        # Identify architecture type
        has_conv = False
        has_recurrent = False
        has_transformer = False
        
        for layer in model.layers:
            layer_class = layer.__class__.__name__
            if 'Conv' in layer_class:
                has_conv = True
            elif any(rnn_type in layer_class for rnn_type in ['LSTM', 'GRU', 'RNN', 'SimpleRNN']):
                has_recurrent = True
            elif any(tf_type in layer_class for tf_type in ['Attention', 'MultiHead', 'Transformer']):
                has_transformer = True
        
        if has_transformer:
            model_info['architecture_type'] = 'Transformer'
        elif has_recurrent:
            model_info['architecture_type'] = 'RNN'
        elif has_conv:
            model_info['architecture_type'] = 'CNN'
        else:
            model_info['architecture_type'] = 'MLP'
        
        # Create input shape if not provided
        if input_shape is None:
            input_shape = model.input_shape[1:]
            if model.input_shape[0] is None:  # Batch dimension
                if model_info['architecture_type'] == 'CNN':
                    input_shape = (1,) + tuple(s if s is not None else 224 for s in input_shape)
                elif model_info['architecture_type'] == 'RNN':
                    input_shape = (1,) + tuple(s if s is not None else (128 if i == 0 else 512) 
                                             for i, s in enumerate(input_shape))
                else:
                    input_shape = (1,) + tuple(s if s is not None else 512 for s in input_shape)
        
        # Collect layer information
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'params': layer.count_params(),
                'trainable_params': sum(w.numpy().size for w in layer.trainable_weights) 
                                  if hasattr(layer, 'trainable_weights') else 0,
                'output_shape': str(layer.output_shape),
                'param_size_mb': sum(w.numpy().size * w.dtype.size for w in layer.weights) / (1024 * 1024) 
                               if hasattr(layer, 'weights') and layer.weights else 0
            }
            model_info['layers'].append(layer_info)
            
            # Group by layer type for statistics
            layer_type = layer.__class__.__name__
            if layer_type not in model_info['layer_stats']:
                model_info['layer_stats'][layer_type] = {
                    'count': 0,
                    'total_params': 0,
                    'total_size_mb': 0
                }
            
            model_info['layer_stats'][layer_type]['count'] += 1
            model_info['layer_stats'][layer_type]['total_params'] += layer_info['params']
            model_info['layer_stats'][layer_type]['total_size_mb'] += layer_info['param_size_mb']
            
        # Memory estimation
        model_info['memory_estimation']['parameters_mb'] = sum(layer['param_size_mb'] for layer in model_info['layers'])
        
        # Estimate FLOPs using TensorFlow's profiler if available
        if TF_AVAILABLE and hasattr(model, 'input') and hasattr(tf, 'compat'):
            try:
                # Create a concrete function
                import numpy as np
                dummy_input = np.zeros(input_shape)
                concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec(input_shape))
                
                # Get the graph
                graph = concrete_func.graph
                
                # Rough FLOP estimation based on operation counts (simplified)
                op_counts = {}
                flops = 0
                
                for op in graph.get_operations():
                    op_type = op.type
                    if op_type not in op_counts:
                        op_counts[op_type] = 0
                    op_counts[op_type] += 1
                    
                    # Approximate FLOP counting (very simplified)
                    if op_type in ['MatMul', 'BatchMatMul']:
                        # Estimate matrix multiply FLOPs
                        shapes = [i.shape for i in op.inputs]
                        if len(shapes) >= 2 and len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
                            # For matmul(A, B), FLOPs ~= 2 * prod(A.shape[:-1]) * A.shape[-1] * B.shape[-1]
                            a_elems = 1
                            for dim in shapes[0][:-1]:
                                if dim is not None:
                                    a_elems *= dim
                            
                            if shapes[0][-1] is not None and shapes[1][-1] is not None:
                                flops += 2 * a_elems * shapes[0][-1] * shapes[1][-1]
                    
                    elif op_type in ['Conv2D']:
                        # Estimate convolution FLOPs
                        # This is a simplification; actual FLOPs depend on stride, padding, etc.
                        shapes = [i.shape for i in op.inputs]
                        if len(shapes) >= 2:
                            # input and filter
                            in_shape, filt_shape = shapes[0], shapes[1]
                            if all(dim is not None for shape in [in_shape, filt_shape] for dim in shape):
                                # FLOPs ~= 2 * batch * out_h * out_w * in_c * filt_h * filt_w * out_c
                                # Approximate as 2 * prod(in_shape) * prod(filt_shape) / in_shape[-1]
                                in_elems = 1
                                for dim in in_shape:
                                    in_elems *= dim
                                    
                                filt_elems = 1
                                for dim in filt_shape:
                                    filt_elems *= dim
                                    
                                flops += 2 * in_elems * filt_elems / in_shape[-1]
                
                model_info['performance']['estimated_flops'] = flops
                model_info['performance']['op_counts'] = op_counts
                
            except Exception as e:
                logger.warning(f"Error estimating TensorFlow FLOPs: {e}")
        
        return model_info
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on model analysis"""
        recommendations = []
        
        # Check model size and suggest quantization if large
        if analysis.get('param_count', 0) > 10_000_000:  # > 10M parameters
            recommendations.append({
                'type': 'optimization',
                'priority': 'high',
                'description': 'Model is large (>10M parameters). Consider quantization.',
                'details': 'Quantizing from float32 to int8 can reduce model size by 75% with minimal accuracy loss.'
            })
        
        # Check for architecture-specific recommendations
        arch_type = analysis.get('architecture_type', '')
        
        if arch_type == 'Transformer':
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'description': 'Consider optimizing transformer inference',
                'details': 'For inference, try using ONNX Runtime, TorchScript or TensorRT for up to 5x speedup.'
            })
            
            if analysis.get('transformer_details', {}).get('attention_heads', 0) > 8:
                recommendations.append({
                    'type': 'architecture',
                    'priority': 'medium',
                    'description': 'Large number of attention heads detected',
                    'details': 'Consider pruning attention heads. Research shows many heads can be removed with minimal impact.'
                })
        
        elif arch_type == 'CNN':
            # Check if using batch normalization
            has_batchnorm = any('BatchNorm' in layer.get('type', '') for layer in analysis.get('layers', []))
            if not has_batchnorm:
                recommendations.append({
                    'type': 'architecture',
                    'priority': 'medium',
                    'description': 'No batch normalization layers detected',
                    'details': 'Adding BatchNorm layers can improve training stability and convergence.'
                })
        
        # Check memory requirements
        if analysis.get('memory_estimation', {}).get('total_mb', 0) > 1024:  # > 1GB
            recommendations.append({
                'type': 'memory',
                'priority': 'high',
                'description': 'High memory usage detected (>1GB)',
                'details': 'Consider gradient checkpointing, mixed precision training, or model sharding.'
            })
        
        # Check framework-specific optimizations
        if analysis.get('type') == 'pytorch':
            recommendations.append({
                'type': 'deployment',
                'priority': 'low',
                'description': 'Consider TorchScript for deployment',
                'details': 'Converting to TorchScript can improve performance and portability.'
            })
        
        elif analysis.get('type') == 'tensorflow':
            recommendations.append({
                'type': 'deployment',
                'priority': 'low',
                'description': 'Consider TF-TRT for deployment',
                'details': 'TensorFlow with TensorRT integration can significantly improve inference performance.'
            })
        
        return recommendations
    
    def generate_report(self, analysis: Optional[Dict[str, Any]] = None, 
                       format: str = 'markdown') -> str:
        """
        Generate a formatted report from analysis results
        
        Args:
            analysis: Analysis results (uses last analysis if None)
            format: Output format ('text', 'markdown', 'html')
            
        Returns:
            Formatted report string
        """
        if analysis is None:
            if self.last_analysis is None:
                return "No analysis available. Run analyze_model first."
            analysis = self.last_analysis
        
        if format == 'markdown':
            return self._generate_markdown_report(analysis)
        elif format == 'html':
            return self._generate_html_report(analysis)
        else:  # text
            return self._generate_text_report(analysis)
    
    def _generate_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a markdown formatted report"""
        report = [f"# Model Analysis: {analysis.get('name', 'Unknown Model')}"]
        
        # General information
        report.append("## General Information")
        report.append(f"- **Architecture Type:** {analysis.get('architecture_type', 'Unknown')}")
        report.append(f"- **Framework:** {analysis.get('type', 'Unknown')}")
        report.append(f"- **Parameters:** {analysis.get('param_count', 0):,}")
        report.append(f"- **Trainable Parameters:** {analysis.get('trainable_params', 0):,}")
        
        # Performance metrics
        report.append("\n## Performance Metrics")
        if 'performance' in analysis:
            perf = analysis['performance']
            if 'flops' in perf:
                report.append(f"- **Estimated FLOPs:** {perf['flops']:,}")
            if 'theoretical_batches_per_second' in perf:
                report.append(f"- **Theoretical Throughput:** {perf['theoretical_batches_per_second']:.2f} batches/second")
        
        # Memory usage
        if 'memory_estimation' in analysis:
            mem = analysis['memory_estimation']
            report.append("\n## Memory Usage")
            report.append(f"- **Parameters Memory:** {mem.get('parameters_mb', 0):.2f} MB")
            if 'activations_mb' in mem:
                report.append(f"- **Activations Memory:** {mem['activations_mb']:.2f} MB")
            if 'total_mb' in mem:
                report.append(f"- **Total Memory:** {mem['total_mb']:.2f} MB")
        
        # Layer summary
        report.append("\n## Layer Summary")
        report.append("| Layer Type | Count | Parameters | Memory (MB) |")
        report.append("|------------|-------|-----------|------------|")
        
        if 'layer_stats' in analysis:
            for layer_type, stats in analysis['layer_stats'].items():
                report.append(f"| {layer_type} | {stats['count']} | {stats['total_params']:,} | "
                            f"{stats['total_size_mb']:.2f} |")
        
        # Recommendations
        if 'recommendations' in analysis and analysis['recommendations']:
            report.append("\n## Optimization Recommendations")
            
            for i, rec in enumerate(analysis['recommendations']):
                priority_marker = "🔴" if rec['priority'] == 'high' else "🟠" if rec['priority'] == 'medium' else "🟢"
                report.append(f"\n### {i+1}. {priority_marker} {rec['description']}")
                report.append(f"{rec['details']}")
        
        return "\n".join(report)
    
    def _generate_text_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a plain text report"""
        # Similar to markdown but without formatting
        # ...
        return "Text report"  # Placeholder
    
    def _generate_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate an HTML report"""
        # HTML report with potential visualizations
        # ...
        return "<html>...</html>"  # Placeholder
    
    def export_report(self, output_path: str, analysis: Optional[Dict[str, Any]] = None, 
                     format: str = 'markdown') -> str:
        """
        Export analysis report to a file
        
        Args:
            output_path: Path to save the report
            analysis: Analysis results (uses last analysis if None)
            format: Format to export ('markdown', 'html', 'json')
            
        Returns:
            Path to the saved file
        """
        if analysis is None:
            if self.last_analysis is None:
                raise ValueError("No analysis available. Run analyze_model first.")
            analysis = self.last_analysis
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
        else:
            report = self.generate_report(analysis, format)
            with open(output_path, 'w') as f:
                f.write(report)
        
        return output_path
