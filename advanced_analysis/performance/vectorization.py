"""
CPU vectorization analysis and optimization.

This module provides tools for analyzing Python code to detect opportunities
for vectorization and parallelization, with a focus on CPU-bound operations
in machine learning workloads.
"""

import ast
import textwrap
import re
import logging
from typing import Dict, List, Set, Optional, Any, Union

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Some vectorization analysis features will be limited.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. CPU utilization monitoring will be limited.")


class CPUVectorizationAnalyzer:
    """
    Static analysis class to detect naive loops that should be vectorized
    or parallelized on CPU for AI/ML code.
    """
    def __init__(self, code_str: str):
        """
        Initialize with code to analyze.
        
        Args:
            code_str: Python code as a string
        """
        self.original_code = code_str
        # attempt to parse AST
        try:
            self.tree = ast.parse(textwrap.dedent(code_str))
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            self.tree = None

    def analyze_code(self) -> Dict[str, Any]:
        """
        Analyze code for vectorization opportunities.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.tree:
            return {
                "error": "Could not parse code",
                "naive_loops": [],
                "recommendations": []
            }
            
        # Find naive loops that could be vectorized
        naive_loops = self._find_naive_loops()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(naive_loops)
        
        return {
            "naive_loops": naive_loops,
            "recommendations": recommendations
        }
        
    def _find_naive_loops(self) -> List[Dict[str, Any]]:
        """
        Find loops that could be vectorized.
        
        Returns:
            List of dictionaries describing naive loops
        """
        naive_loops = []
        
        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.loops = []
                self.current_loop_depth = 0
                self.loop_stack = []
                self.numpy_imported = False
                self.pandas_imported = False
                self.import_aliases = {
                    'numpy': set(['np', 'numpy']),
                    'pandas': set(['pd', 'pandas'])
                }
                
            def visit_Import(self, node):
                # Track imports of vectorization libraries
                for alias in node.names:
                    if alias.name == 'numpy':
                        self.numpy_imported = True
                        if alias.asname:
                            self.import_aliases['numpy'].add(alias.asname)
                    elif alias.name == 'pandas':
                        self.pandas_imported = True
                        if alias.asname:
                            self.import_aliases['pandas'].add(alias.asname)
                self.generic_visit(node)
                
            def visit_ImportFrom(self, node):
                # Track imports from vectorization libraries
                if node.module == 'numpy':
                    self.numpy_imported = True
                elif node.module == 'pandas':
                    self.pandas_imported = True
                self.generic_visit(node)
                
            def visit_For(self, node):
                # Track loop nesting depth
                self.current_loop_depth += 1
                self.loop_stack.append(node)
                
                # Check if this is a naive loop that could be vectorized
                loop_info = {
                    'type': 'for',
                    'line': node.lineno,
                    'depth': self.current_loop_depth,
                    'vectorizable': False,
                    'reason': None,
                    'code': ast.unparse(node) if hasattr(ast, 'unparse') else None
                }
                
                # Check for arithmetic operations in loop body
                arithmetic_ops = self._count_arithmetic_ops(node)
                if arithmetic_ops > 0:
                    loop_info['vectorizable'] = True
                    loop_info['reason'] = f"Contains {arithmetic_ops} arithmetic operations that could be vectorized"
                
                # Check for list/dict comprehensions that could be vectorized
                comprehensions = self._find_comprehensions(node)
                if comprehensions:
                    loop_info['vectorizable'] = True
                    loop_info['reason'] = f"Contains {len(comprehensions)} comprehensions that could be vectorized"
                
                # Check for iterating over numpy arrays or pandas DataFrames with Python loops
                if self._is_iterating_over_numpy_or_pandas(node):
                    loop_info['vectorizable'] = True
                    loop_info['reason'] = "Iterating over numpy array or pandas DataFrame with Python loop"
                
                if loop_info['vectorizable']:
                    self.loops.append(loop_info)
                
                # Visit loop body
                self.generic_visit(node)
                
                # Restore state
                self.loop_stack.pop()
                self.current_loop_depth -= 1
                
            def _count_arithmetic_ops(self, node):
                """Count arithmetic operations in a node"""
                class ArithmeticCounter(ast.NodeVisitor):
                    def __init__(self):
                        self.count = 0
                        
                    def visit_BinOp(self, node):
                        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                            self.count += 1
                        self.generic_visit(node)
                        
                counter = ArithmeticCounter()
                counter.visit(node)
                return counter.count
                
            def _find_comprehensions(self, node):
                """Find list/dict comprehensions in a node"""
                class ComprehensionFinder(ast.NodeVisitor):
                    def __init__(self):
                        self.comprehensions = []
                        
                    def visit_ListComp(self, node):
                        self.comprehensions.append(('list_comp', node.lineno))
                        self.generic_visit(node)
                        
                    def visit_DictComp(self, node):
                        self.comprehensions.append(('dict_comp', node.lineno))
                        self.generic_visit(node)
                        
                finder = ComprehensionFinder()
                finder.visit(node)
                return finder.comprehensions
                
            def _is_iterating_over_numpy_or_pandas(self, node):
                """Check if loop is iterating over numpy array or pandas DataFrame"""
                if isinstance(node.iter, ast.Name):
                    # Simple case: for i in array
                    return False  # Can't determine type from just the name
                elif isinstance(node.iter, ast.Attribute):
                    # Check for pandas iterrows/itertuples
                    if node.iter.attr in ('iterrows', 'itertuples', 'items'):
                        return True
                elif isinstance(node.iter, ast.Call):
                    # Check for range() calls that could be replaced with numpy operations
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        return True
                return False
        
        # Run the visitor
        visitor = LoopVisitor()
        visitor.visit(self.tree)
        
        return visitor.loops
        
    def _generate_recommendations(self, naive_loops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate recommendations for vectorization.
        
        Args:
            naive_loops: List of dictionaries describing naive loops
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Check if numpy is imported
        numpy_imported = False
        pandas_imported = False
        
        class ImportChecker(ast.NodeVisitor):
            def __init__(self):
                self.numpy_imported = False
                self.pandas_imported = False
                
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name == 'numpy':
                        self.numpy_imported = True
                    elif alias.name == 'pandas':
                        self.pandas_imported = True
                self.generic_visit(node)
                
            def visit_ImportFrom(self, node):
                if node.module == 'numpy':
                    self.numpy_imported = True
                elif node.module == 'pandas':
                    self.pandas_imported = True
                self.generic_visit(node)
        
        if self.tree:
            checker = ImportChecker()
            checker.visit(self.tree)
            numpy_imported = checker.numpy_imported
            pandas_imported = checker.pandas_imported
        
        # Recommend numpy for vectorization if not already imported
        if naive_loops and not numpy_imported:
            recommendations.append({
                "type": "import",
                "severity": "medium",
                "message": "Consider importing NumPy for vectorized operations",
                "details": "NumPy provides efficient vectorized operations that can replace loops.",
                "code_example": "import numpy as np"
            })
        
        # Generate specific recommendations for each naive loop
        for loop in naive_loops:
            if loop['vectorizable']:
                if 'arithmetic' in loop.get('reason', '').lower():
                    recommendations.append({
                        "type": "vectorization",
                        "severity": "medium",
                        "message": f"Vectorize arithmetic operations in loop at line {loop['line']}",
                        "details": "Replace loop with NumPy vectorized operations for better performance.",
                        "code_example": """
# Instead of:
result = []
for i in range(len(data)):
    result.append(data[i] * 2 + 1)

# Use:
import numpy as np
result = data * 2 + 1
"""
                    })
                elif 'comprehension' in loop.get('reason', '').lower():
                    recommendations.append({
                        "type": "vectorization",
                        "severity": "low",
                        "message": f"Consider replacing comprehension at line {loop['line']} with NumPy operations",
                        "details": "NumPy operations can be faster than list comprehensions for numerical data.",
                        "code_example": """
# Instead of:
result = [x * 2 for x in data]

# Use:
import numpy as np
result = np.array(data) * 2
"""
                    })
                elif 'iterating over numpy' in loop.get('reason', '').lower() or 'pandas' in loop.get('reason', '').lower():
                    recommendations.append({
                        "type": "vectorization",
                        "severity": "high",
                        "message": f"Avoid iterating over NumPy arrays or Pandas DataFrames at line {loop['line']}",
                        "details": "Iterating over NumPy arrays or Pandas DataFrames with Python loops is inefficient.",
                        "code_example": """
# Instead of:
for i, row in df.iterrows():
    result[i] = row['a'] * 2

# Use:
df['result'] = df['a'] * 2

# Or for NumPy:
# Instead of:
for i in range(len(arr)):
    result[i] = arr[i] * 2

# Use:
result = arr * 2
"""
                    })
        
        # Check for nested loops that could be vectorized
        nested_loops = [loop for loop in naive_loops if loop['depth'] > 1]
        if nested_loops:
            recommendations.append({
                "type": "vectorization",
                "severity": "high",
                "message": f"Nested loops detected at lines {', '.join(str(loop['line']) for loop in nested_loops)}",
                "details": "Nested loops can often be replaced with vectorized operations for better performance.",
                "code_example": """
# Instead of:
result = []
for i in range(len(data)):
    row = []
    for j in range(len(data[i])):
        row.append(data[i][j] * 2)
    result.append(row)

# Use:
import numpy as np
result = np.array(data) * 2
"""
            })
        
        return recommendations


class CPUUtilizationMonitor:
    """
    Monitor CPU utilization during function execution.
    """
    def __init__(self):
        """Initialize the CPU utilization monitor."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available. CPU utilization monitoring will be limited.")
            
    def monitor_function(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Monitor CPU utilization during function execution.
        
        Args:
            func: Function to monitor
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary with monitoring results
        """
        if not PSUTIL_AVAILABLE:
            return {
                "error": "psutil not available",
                "cpu_utilization": None,
                "execution_time": None
            }
            
        import time
        import threading
        
        # Initialize monitoring data
        monitoring_data = {
            "cpu_percent": [],
            "num_threads": [],
            "sampling_times": []
        }
        
        # Flag to control monitoring thread
        stop_monitoring = threading.Event()
        
        # Monitoring thread function
        def monitor_cpu():
            process = psutil.Process()
            while not stop_monitoring.is_set():
                monitoring_data["cpu_percent"].append(process.cpu_percent(interval=0.1))
                monitoring_data["num_threads"].append(process.num_threads())
                monitoring_data["sampling_times"].append(time.time())
                time.sleep(0.1)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Execute function and measure time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            error = str(e)
        finally:
            end_time = time.time()
            
        # Stop monitoring thread
        stop_monitoring.set()
        monitor_thread.join(timeout=1.0)
        
        # Calculate statistics
        execution_time = end_time - start_time
        avg_cpu_percent = sum(monitoring_data["cpu_percent"]) / len(monitoring_data["cpu_percent"]) if monitoring_data["cpu_percent"] else 0
        max_cpu_percent = max(monitoring_data["cpu_percent"]) if monitoring_data["cpu_percent"] else 0
        avg_num_threads = sum(monitoring_data["num_threads"]) / len(monitoring_data["num_threads"]) if monitoring_data["num_threads"] else 0
        max_num_threads = max(monitoring_data["num_threads"]) if monitoring_data["num_threads"] else 0
        
        # Prepare results
        monitoring_results = {
            "success": success,
            "execution_time": execution_time,
            "cpu_utilization": {
                "average": avg_cpu_percent,
                "maximum": max_cpu_percent,
                "samples": monitoring_data["cpu_percent"]
            },
            "threading": {
                "average_threads": avg_num_threads,
                "maximum_threads": max_num_threads,
                "samples": monitoring_data["num_threads"]
            },
            "sampling_times": monitoring_data["sampling_times"]
        }
        
        if not success:
            monitoring_results["error"] = error
        
        return monitoring_results, result if success else None

"""
Vectorization efficiency analysis for neural networks.

This module provides tools to analyze how efficiently models use vectorized
operations and tensor cores, and identifies opportunities for optimization.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import platform

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.utils.benchmark as benchmark
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch-specific features will be disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Some features will be disabled.")

@dataclass
class VectorOp:
    """Information about a vectorizable operation"""
    name: str  # Name of the operation
    shape: Tuple  # Shape of tensors involved
    is_vectorized: bool  # Whether operation uses vectorized implementation
    efficiency: float  # Efficiency of vectorization (0-1)
    execution_time: float  # Execution time in ms
    tensor_core_eligible: bool = False  # Whether op could use tensor cores
    tensor_core_utilized: bool = False  # Whether op actually used tensor cores


@dataclass
class VectorizationResult:
    """Results of vectorization analysis"""
    operations: List[VectorOp]  # List of analyzed operations
    overall_efficiency: float  # Overall vectorization efficiency (0-1)
    optimization_candidates: List[Dict[str, Any]]  # Operations that could be optimized
    recommendations: List[Dict[str, Any]]  # Recommendations for better vectorization
    execution_time: float  # Total execution time in ms
    tensor_core_usage: float  # Percentage of eligible ops using tensor cores


class VectorizationAnalyzer:
    """
    Analyzes neural network models for vectorization efficiency and
    provides recommendations for optimization
    """
    
    def __init__(self, model=None, framework="pytorch"):
        self.model = model
        self.framework = framework.lower()
        self.device_info = self._get_device_info()
        self.simd_width = self._detect_simd_width()
        
    def _get_device_info(self):
        """Get hardware capability information"""
        device_info = {
            "architecture": platform.machine(),
            "system": platform.system(),
            "has_avx": False,
            "has_avx2": False,
            "has_avx512": False,
            "tensor_cores": False
        }
        
        # Check for CPU vectorization capabilities
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    if "avx512" in cpuinfo:
                        device_info["has_avx512"] = True
                    if "avx2" in cpuinfo:
                        device_info["has_avx2"] = True
                    if "avx" in cpuinfo:
                        device_info["has_avx"] = True
            except:
                pass
        
        # Check for GPU features if PyTorch is available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.cuda.current_device()
            device_info["gpu_name"] = torch.cuda.get_device_name(device)
            device_info["compute_capability"] = torch.cuda.get_device_capability(device)
            
            # Check if tensor cores are available (compute capability 7.0+)
            if device_info["compute_capability"][0] >= 7:
                device_info["tensor_cores"] = True
                
                # Ampere (SM80) and later also have sparse tensor cores
                if device_info["compute_capability"][0] >= 8:
                    device_info["sparse_tensor_cores"] = True
        
        return device_info
    
    def _detect_simd_width(self):
        """Detect SIMD width based on CPU architecture"""
        if self.device_info.get("has_avx512"):
            return 512
        elif self.device_info.get("has_avx2"):
            return 256
        elif self.device_info.get("has_avx"):
            return 256
        else:
            return 128  # SSE
    
    def analyze(self, input_data=None, trace_path=None):
        """
        Analyze model for vectorization efficiency
        
        Args:
            input_data: Input data for the model (for dynamic analysis)
            trace_path: Path to a saved PyTorch trace (for static analysis)
            
        Returns:
            VectorizationResult with detailed analysis
        """
        if self.framework == "pytorch":
            return self._analyze_pytorch(input_data, trace_path)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _analyze_pytorch(self, input_data=None, trace_path=None):
        """Analyze PyTorch model for vectorization efficiency"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for vectorization analysis")
        
        operations = []
        tensor_core_eligible = 0
        tensor_core_utilized = 0
        total_execution_time = 0
        
        # Determine if input is on GPU
        device = "cpu"
        if self.model and next(self.model.parameters(), None) is not None:
            device = next(self.model.parameters()).device
        
        if input_data is not None:
            # Move input to appropriate device
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.to(device)
            elif isinstance(input_data, (list, tuple)):
                input_data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in input_data]
            
            # Set model to eval mode and perform analysis
            if self.model:
                self.model.eval()
                
                # Collect operations using PyTorch profiler
                with torch.no_grad():
                    # Warm-up run
                    self.model(input_data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    
                    # Profiling run
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA if device.type == "cuda" else None
                        ],
                        record_shapes=True
                    ) as prof:
                        start_time = time.time()
                        output = self.model(input_data)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        total_execution_time = (time.time() - start_time) * 1000  # ms
                
                # Analyze profile results
                for event in prof.key_averages():
                    if event.self_cpu_time_total < 1:  # Skip insignificant ops
                        continue
                    
                    op_name = event.key
                    shapes = event.input_shapes if hasattr(event, 'input_shapes') else []
                    shapes = [shape for shape in shapes if shape]  # Filter out None shapes
                    
                    # Determine if operation is vectorized
                    is_vectorized = False
                    efficiency = 0.0
                    
                    # Check for common vectorized operations
                    if any(x in op_name for x in ['matmul', 'conv', 'gemm', 'addmm', 'linear']):
                        is_vectorized = True
                        # Estimate efficiency based on shapes
                        efficiency = self._estimate_vectorization_efficiency(op_name, shapes)
                    
                    # Check for tensor core eligibility
                    tensor_core_eligible_op = False
                    tensor_core_utilized_op = False
                    
                    if device.type == "cuda" and self.device_info.get("tensor_cores", False):
                        tensor_core_eligible_op = self._is_tensor_core_eligible(op_name, shapes)
                        if tensor_core_eligible_op:
                            tensor_core_eligible += 1
                            tensor_core_utilized_op = self._is_tensor_core_utilized(op_name, event)
                            if tensor_core_utilized_op:
                                tensor_core_utilized += 1
                    
                    operations.append(VectorOp(
                        name=op_name,
                        shape=tuple(shapes[0]) if shapes else (),
                        is_vectorized=is_vectorized,
                        efficiency=efficiency,
                        execution_time=event.self_cpu_time_total / 1000,  # Convert to ms
                        tensor_core_eligible=tensor_core_eligible_op,
                        tensor_core_utilized=tensor_core_utilized_op
                    ))
        
        # Calculate overall efficiency
        vectorized_ops = [op for op in operations if op.is_vectorized]
        if vectorized_ops:
            overall_efficiency = sum(op.efficiency * op.execution_time for op in vectorized_ops) / \
                               sum(op.execution_time for op in vectorized_ops)
        else:
            overall_efficiency = 0.0
        
        # Find operations that could be optimized
        optimization_candidates = []
        
        for op in operations:
            if op.is_vectorized and op.efficiency < 0.7:
                # This op is vectorized but not efficiently
                optimization_candidates.append({
                    "operation": op.name,
                    "shape": op.shape,
                    "current_efficiency": op.efficiency,
                    "reason": "Inefficient vectorization due to suboptimal tensor shapes"
                })
            elif op.tensor_core_eligible and not op.tensor_core_utilized:
                # This op could use tensor cores but doesn't
                optimization_candidates.append({
                    "operation": op.name,
                    "shape": op.shape,
                    "current_efficiency": op.efficiency,
                    "reason": "Operation is eligible for Tensor Cores but doesn't use them"
                })
        
        # Generate recommendations
        recommendations = []
        
        # CPU-specific recommendations
        if device == "cpu":
            if self.device_info.get("has_avx512") and overall_efficiency < 0.7:
                recommendations.append({
                    "type": "cpu_vectorization",
                    "suggestion": "Optimize for AVX-512",
                    "details": [
                        "Your CPU supports AVX-512 instructions but vectorization efficiency is low.",
                        "Ensure tensor dimensions are multiples of 16 for float32 operations.",
                        "Consider using torch.backends.mkldnn to improve performance."
                    ],
                    "code_example": "# Enable MKL-DNN optimizations\n"
                                    "torch.backends.mkldnn.enabled = True\n\n"
                                    "# Reshape tensors to be more vectorization-friendly\n"
                                    "# Example: pad dimensions to multiples of 16\n"
                                    "batch_size = (batch_size + 15) // 16 * 16"
                })
            elif self.device_info.get("has_avx2") and overall_efficiency < 0.7:
                recommendations.append({
                    "type": "cpu_vectorization",
                    "suggestion": "Optimize for AVX2",
                    "details": [
                        "Your CPU supports AVX2 instructions but vectorization efficiency is low.",
                        "Ensure tensor dimensions are multiples of 8 for float32 operations.",
                        "Use contiguous memory layout by calling tensor.contiguous() before operations."
                    ],
                    "code_example": "# Ensure contiguous memory layout\n"
                                    "tensor = tensor.contiguous()\n\n"
                                    "# Reshape tensors to be vectorization-friendly\n"
                                    "# Example: pad dimensions to multiples of 8\n"
                                    "hidden_dim = (hidden_dim + 7) // 8 * 8"
                })
                
        # GPU-specific recommendations
        if device.type == "cuda":
            # Tensor Core recommendations
            if self.device_info.get("tensor_cores", False):
                if tensor_core_eligible == 0:
                    recommendations.append({
                        "type": "tensor_cores",
                        "suggestion": "Reshape operations to utilize Tensor Cores",
                        "details": [
                            "Your GPU supports Tensor Cores but no operations are currently eligible.",
                            "Reshape matrix dimensions to multiples of 8 for FP16/BF16 or 16 for INT8.",
                            "Consider using Automatic Mixed Precision (AMP) for training."
                        ],
                        "code_example": "# Enable Automatic Mixed Precision\n"
                                        "from torch.cuda.amp import autocast, GradScaler\n\n"
                                        "scaler = GradScaler()\n"
                                        "with autocast():\n"
                                        "    output = model(input)\n"
                                        "    loss = loss_fn(output, target)\n\n"
                                        "scaler.scale(loss).backward()\n"
                                        "scaler.step(optimizer)\n"
                                        "scaler.update()"
                    })
                elif tensor_core_utilized < tensor_core_eligible:
                    tc_usage = tensor_core_utilized / tensor_core_eligible if tensor_core_eligible > 0 else 0
                    recommendations.append({
                        "type": "tensor_cores",
                        "suggestion": f"Improve Tensor Core utilization (currently {tc_usage:.1%})",
                        "details": [
                            f"Only {tensor_core_utilized} of {tensor_core_eligible} eligible operations use Tensor Cores.",
                            "Ensure you're using PyTorch with CUDA 10.2+ and cuDNN 7.6.5+",
                            "Force operations to use FP16/BF16 or enable AMP.",
                            "For inference, consider TensorRT which optimizes for Tensor Cores."
                        ],
                        "code_example": "# Check if your operations use TF32 (on Ampere GPUs)\n"
                                        "print(f\"MatMul TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}\")\n"
                                        "print(f\"cuDNN TF32 enabled: {torch.backends.cudnn.allow_tf32}\")\n\n"
                                        "# Enable TF32 if supported\n"
                                        "torch.backends.cuda.matmul.allow_tf32 = True\n"
                                        "torch.backends.cudnn.allow_tf32 = True"
                    })
            
            # Memory coalescing recommendations
            if optimization_candidates and any("suboptimal tensor shapes" in cand["reason"] for cand in optimization_candidates):
                recommendations.append({
                    "type": "memory_coalescing",
                    "suggestion": "Improve memory access patterns",
                    "details": [
                        "Some operations have poor memory coalescing, likely due to suboptimal tensor shapes or strides.",
                        "Ensure contiguous memory layout before compute-intensive operations.",
                        "Align tensor dimensions with warp size (32 threads).",
                        "Consider using channels_last memory format for CNNs."
                    ],
                    "code_example": "# Convert to channels_last format for CNNs\n"
                                    "model = model.to(memory_format=torch.channels_last)\n"
                                    "input = input.to(memory_format=torch.channels_last)\n\n"
                                    "# Ensure contiguous memory layout\n"
                                    "if not tensor.is_contiguous():\n"
                                    "    tensor = tensor.contiguous()"
                })
        
        # Add general recommendations for improving vectorization
        recommendations.append({
            "type": "general",
            "suggestion": "General vectorization improvements",
            "details": [
                "Avoid small, fragmented operations that can't benefit from vectorization",
                "Batch smaller operations together when possible",
                "Use torch.nn.functional.linear instead of custom matrix operations",
                "Consider using optimized libraries like FBGEMM for quantized operations"
            ],
            "code_example": "# Instead of many small matrix multiplications in a loop:\n"
                            "# for i in range(100):\n"
                            "#     c = torch.matmul(a[i], b[i])\n\n"
                            "# Batch them into a single operation:\n"
                            "c = torch.bmm(a, b)  # batch matrix multiply"
        })
        
        # Calculate tensor core usage
        tensor_core_usage = tensor_core_utilized / tensor_core_eligible if tensor_core_eligible > 0 else 0.0
        
        return VectorizationResult(
            operations=operations,
            overall_efficiency=overall_efficiency,
            optimization_candidates=optimization_candidates,
            recommendations=recommendations,
            execution_time=total_execution_time,
            tensor_core_usage=tensor_core_usage
        )
    
    def _estimate_vectorization_efficiency(self, op_name, shapes):
        """Estimate vectorization efficiency based on operation and tensor shapes"""
        if not shapes:
            return 0.0
            
        efficiency = 0.8  # Default assumption for recognized vectorized ops
        
        # Check for suboptimal sizes for vectorization
        if 'cpu' in str(self.device_info.get('architecture', '')):
            # For CPU, check against SIMD width
            simd_elements = self.simd_width // 32  # Assuming float32, get elements per vector
            
            for shape in shapes:
                if not shape:
                    continue
                # Check if innermost dimensions are SIMD-friendly
                innermost_dim = shape[-1]
                if innermost_dim % simd_elements != 0:
                    # Not aligned with SIMD width
                    efficiency *= (innermost_dim // simd_elements * simd_elements) / innermost_dim
        
        # For GPU operations
        elif any(x in op_name.lower() for x in ['cuda', 'gpu']):
            # Check for memory coalescing (multiples of 32 bytes)
            for shape in shapes:
                if not shape:
                    continue
                # For GPU, check last dimension for memory coalescing
                if shape[-1] % 8 != 0:  # Assuming float32, 8 elements = 32 bytes
                    efficiency *= 0.7  # Penalty for non-coalesced access
                    
                # Check if shape would lead to bank conflicts
                if any(dim % 32 == 0 and dim % 64 != 0 for dim in shape):
                    efficiency *= 0.9  # Penalty for potential bank conflicts
        
        # Special case for matrix multiplications
        if any(x in op_name.lower() for x in ['matmul', 'gemm', 'bmm', 'addmm']):
            # Check for efficient dimensions for matrix multiplications
            matrix_dims = []
            for shape in shapes:
                if len(shape) >= 2:
                    matrix_dims.extend([shape[-2], shape[-1]])
            
            # Penalize for non-multiples of 16/32/64 in matrix dimensions
            for dim in matrix_dims:
                if dim % 64 == 0:
                    pass  # Ideal
                elif dim % 32 == 0:
                    efficiency *= 0.95
                elif dim % 16 == 0:
                    efficiency *= 0.9
                elif dim % 8 == 0:
                    efficiency *= 0.85
                else:
                    efficiency *= 0.7
        
        return max(0.1, min(1.0, efficiency))  # Clamp between 0.1 and 1.0
    
    def _is_tensor_core_eligible(self, op_name, shapes):
        """Check if an operation is eligible for Tensor Cores"""
        if not self.device_info.get("tensor_cores", False):
            return False
        
        # Check for operations that can use Tensor Cores
        tensor_core_ops = ['matmul', 'conv', 'linear', 'addmm', 'bmm', 'gemm']
        if not any(op in op_name.lower() for op in tensor_core_ops):
            return False
            
        # Check shapes for Tensor Core compatibility
        # For Volta: dimensions must be multiples of 8
        # For Ampere/Turing: dimensions should ideally be multiples of 8 for best performance
        
        # Extract matrix dimensions from shapes
        matrix_dims = []
        for shape in shapes:
            if not shape or len(shape) < 2:
                continue
                
            if 'conv' in op_name.lower():
                # For convolutions, look at input channels, output channels, and batch size
                if len(shape) == 4:  # NCHW format
                    matrix_dims.extend([shape[0], shape[1]])
            else:
                # For matrix multiplications, look at rows and columns
                matrix_dims.extend(shape[-2:])
        
        # Check if dimensions are compatible with Tensor Cores
        tensor_core_compatible = False
        if matrix_dims:
            # For Ampere (compute capability 8.x), TF32 requires multiples of 4
            if self.device_info.get("compute_capability", (0, 0))[0] >= 8:
                tensor_core_compatible = all(dim % 4 == 0 for dim in matrix_dims)
            # For Volta/Turing, FP16 requires multiples of 8
            else:
                tensor_core_compatible = all(dim % 8 == 0 for dim in matrix_dims)
        
        return tensor_core_compatible
    
    def _is_tensor_core_utilized(self, op_name, event):
        """Check if an operation actually used Tensor Cores based on profiling info"""
        # This is challenging to determine without special profiling tools
        # As a heuristic, we'll check for certain keywords in the kernel name
        tc_keywords = ['hmma', 'tensor', 'tc', 'tgv', 'tf32', 'tfma']
        
        if hasattr(event, 'device_self_cuda_kernel_names'):
            kernel_names = event.device_self_cuda_kernel_names
            return any(any(kw in kernel_name.lower() for kw in tc_keywords) for kernel_name in kernel_names)
            
        # If we can't determine directly, use a heuristic based on performance
        if self._is_tensor_core_eligible(op_name, event.input_shapes if hasattr(event, 'input_shapes') else []):
            # If eligible, assume it's used with 70% probability (educated guess)
            return True
            
        return False
    
    def benchmark_vectorization(self, functions, input_sizes=None, dtypes=None, iterations=100):
        """
        Benchmark vectorization performance for different function implementations
        
        Args:
            functions: Dict mapping names to functions
            input_sizes: List of input sizes to test
            dtypes: List of dtypes to test
            iterations: Number of benchmark iterations
            
        Returns:
            Dict with benchmark results
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for benchmarking")
            
        if input_sizes is None:
            # Default progressive sizes
            input_sizes = [
                (32, 32),
                (64, 64),
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
                # Include one non-power-of-2 size to test alignment effects
                (766, 766)
            ]
            
        if dtypes is None:
            dtypes = [torch.float32]
            if torch.cuda.is_available():
                dtypes.append(torch.float16)
                
                # Add TF32 if on Ampere
                if torch.cuda.get_device_capability(0)[0] >= 8:
                    dtypes.append("tf32")  # Special marker for TF32 mode
        
        results = {}
        
        for dtype in dtypes:
            dtype_results = {}
            
            # Special handling for TF32
            if dtype == "tf32" and torch.cuda.is_available():
                original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
                original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                current_dtype = torch.float32  # TF32 uses float32 tensors with internal TF32 compute
            else:
                current_dtype = dtype
            
            for size in input_sizes:
                size_results = {}
                
                # Create input tensors
                if len(size) == 2:
                    a = torch.randn(*size, dtype=current_dtype)
                    b = torch.randn(*size, dtype=current_dtype)
                elif len(size) == 3:
                    a = torch.randn(*size, dtype=current_dtype)
                    b = torch.randn(*size, dtype=current_dtype)
                else:
                    a = torch.randn(size, dtype=current_dtype)
                    b = torch.randn(size, dtype=current_dtype)
                
                if torch.cuda.is_available():
                    a = a.cuda()
                    b = b.cuda()
                
                # Benchmark each function
                for name, func in functions.items():
                    try:
                        # Use PyTorch's benchmark utility
                        timer = benchmark.Timer(
                            stmt=f"func(a, b)",
                            globals={'func': func, 'a': a, 'b': b}
                        )
                        
                        # Run benchmark
                        benchmark_result = timer.timeit(iterations)
                        mean_time = benchmark_result.mean * 1e6  # Convert to microseconds
                        
                        size_results[name] = {
                            "mean_time_us": mean_time,
                            "std_time_us": benchmark_result.median * 1e6,
                            "iterations": iterations
                        }
                    except Exception as e:
                        size_results[name] = {
                            "error": str(e)
                        }
                
                # Find fastest implementation for this size
                valid_times = [(name, result["mean_time_us"]) 
                              for name, result in size_results.items() 
                              if "error" not in result]
                
                if valid_times:
                    fastest_name, fastest_time = min(valid_times, key=lambda x: x[1])
                    
                    # Calculate speedups relative to fastest
                    for name in size_results:
                        if "error" not in size_results[name]:
                            size_results[name]["speedup_vs_fastest"] = \
                                fastest_time / size_results[name]["mean_time_us"]
                
                # Store results
                size_key = "x".join(str(s) for s in size)
                dtype_results[size_key] = size_results
            
            # Restore original TF32 settings if needed
            if dtype == "tf32" and torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
                torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
            
            # Store results for this dtype
            dtype_name = str(dtype).replace("torch.", "")
            results[dtype_name] = dtype_results
        
        # Add summary and recommendations
        results["summary"] = self._generate_vectorization_recommendations(results, functions)
        
        return results
    
    def _generate_vectorization_recommendations(self, results, functions):
        """Generate recommendations based on vectorization benchmark results"""
        summary = {
            "fastest_implementation": {},
            "largest_speedup": 0,
            "recommendations": []
        }
        
        # Find the overall fastest implementation for each dtype
        for dtype, dtype_results in results.items():
            if dtype == "summary":
                continue
                
            implementation_times = {}
            
            # Aggregate times across sizes
            for size, size_results in dtype_results.items():
                for impl, impl_result in size_results.items():
                    if "error" not in impl_result:
                        if impl not in implementation_times:
                            implementation_times[impl] = []
                        implementation_times[impl].append(impl_result["mean_time_us"])
            
            # Calculate average time for each implementation
            avg_times = {impl: sum(times) / len(times) for impl, times in implementation_times.items()}
            
            if avg_times:
                # Find fastest implementation
                fastest_impl = min(avg_times.items(), key=lambda x: x[1])[0]
                summary["fastest_implementation"][dtype] = fastest_impl
                
                # Calculate overall speedups
                fastest_time = avg_times[fastest_impl]
                for impl, time in avg_times.items():
                    speedup = fastest_time / time
                    if speedup > summary["largest_speedup"]:
                        summary["largest_speedup"] = speedup
        
        # Generate recommendations
        if summary["largest_speedup"] < 1.1:
            summary["recommendations"].append({
                "recommendation": "Current implementations are similarly performant",
                "explanation": "There's less than 10% difference between implementations. Focus optimization efforts elsewhere."
            })
        else:
            # Identify best implementations by dtype
            for dtype, fastest in summary["fastest_implementation"].items():
                alternative_impls = [name for name in functions.keys() if name != fastest]
                if alternative_impls:
                    summary["recommendations"].append({
                        "recommendation": f"Use {fastest} for {dtype} operations",
                        "explanation": f"This implementation outperforms {', '.join(alternative_impls)} for {dtype} operations"
                    })
        
        # Add hardware-specific recommendations
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 7:  # Volta or newer
                summary["recommendations"].append({
                    "recommendation": "Use Tensor Cores for matrix operations",
                    "explanation": "Your GPU supports Tensor Cores. For matrix operations, ensure dimensions are multiples of 8 (FP16) or 4 (TF32)."
                })
                
                if cc[0] >= 8:  # Ampere or newer
                    summary["recommendations"].append({
                        "recommendation": "Enable TF32 for FP32 operations",
                        "explanation": "For Ampere and newer GPUs, TF32 provides substantial speedup with minimal precision loss.",
                        "code_example": "torch.backends.cuda.matmul.allow_tf32 = True\ntorch.backends.cudnn.allow_tf32 = True"
                    })
                    
        # Add vectorization recommendations based on hardware
        if self.device_info.get("has_avx512"):
            summary["recommendations"].append({
                "recommendation": "Optimize for AVX-512",
                "explanation": "Your CPU supports AVX-512. Ensure tensor dimensions are multiples of 16 for optimal vectorization."
            })
        elif self.device_info.get("has_avx2"):
            summary["recommendations"].append({
                "recommendation": "Optimize for AVX2",
                "explanation": "Your CPU supports AVX2. Ensure tensor dimensions are multiples of 8 for optimal vectorization."
            })
        
        return summary
