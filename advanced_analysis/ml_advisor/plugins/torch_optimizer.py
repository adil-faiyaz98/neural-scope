"""
PyTorch optimization suggestions plugin
"""

import uuid
import re
from ..inefficiency_detection import RegexRule, Severity
from ..suggestions import SuggestionPlugin, CodeSuggestion, SuggestionJustification, SuggestionSource


class TorchOptimizationPlugin(SuggestionPlugin):
    """Plugin for PyTorch-specific optimizations."""
    name = "torch_optimizer"
    description = "Suggests PyTorch-specific performance optimizations"
    version = "0.1.0"
    
    def __init__(self):
        super().__init__()
        self.rules = [
            RegexRule(
                name="torch_contiguous_op",
                description="Using .contiguous() unnecessarily can waste memory",
                suggestion="Only use .contiguous() when needed for specific operations",
                severity=Severity.MEDIUM,
                code_example="""
# Instead of:
out = tensor.transpose(0, 1).contiguous().view(-1, 10)

# Use:
out = tensor.transpose(0, 1).view(-1, 10)  # Only use contiguous when needed
                """,
                pattern=r"\.transpose\([^)]+\)\.contiguous\(\)"
            ),
            RegexRule(
                name="cpu_gpu_transfers",
                description="Frequent CPU-GPU transfers can cause performance bottlenecks",
                suggestion="Minimize transfers between CPU and GPU memory",
                severity=Severity.HIGH,
                code_example="""
# Instead of:
for i, data in enumerate(dataloader):
    inputs = data.to('cuda')
    # process on GPU
    result = model(inputs)
    result_cpu = result.cpu().numpy()  # Transferring back to CPU each iteration
    # process on CPU
    
# Use:
results = []
for i, data in enumerate(dataloader):
    inputs = data.to('cuda')
    # process on GPU
    result = model(inputs)
    results.append(result)  # Keep on GPU

# Transfer once at the end
results_cpu = [r.cpu().numpy() for r in results]
                """,
                pattern=r"\.to\(['\"]cuda['\"]\).*\.cpu\(\)"
            )
        ]
    
    def get_rules(self):
        return self.rules
        
    def analyze(self, code: str, context: Dict) -> List[CodeSuggestion]:
        suggestions = []
        
        # Check for specific PyTorch patterns
        if "torch" in context.get("imports", []):
            # Check for suboptimal DataLoader configuration
            if "DataLoader(" in code and not re.search(r"num_workers\s*=\s*[1-9]", code):
                justification = SuggestionJustification(
                    reason="DataLoader performance can be significantly improved with proper worker configuration",
                    source=SuggestionSource.PATTERN,
                    metrics={"impact": "moderate", "estimated_speedup": "2-4x"},
                    confidence=0.8
                )
                
                suggestions.append(CodeSuggestion(
                    id=str(uuid.uuid4()),
                    title="Optimize DataLoader Workers",
                    description="Set num_workers in DataLoader for faster data loading based on CPU cores",
                    code_example="""
# Instead of:
loader = DataLoader(dataset, batch_size=32)

# Use:
import multiprocessing as mp
num_workers = mp.cpu_count()  # Use CPU count or a fraction of it
loader = DataLoader(dataset, batch_size=32, num_workers=num_workers)
                    """,
                    severity="medium",
                    justification=justification
                ))
            
            # Check for use of .item() in loops
            if re.search(r"for.*:.*\.item\(\)", code):
                justification = SuggestionJustification(
                    reason="Using .item() in loops causes synchronization overhead between CPU and GPU",
                    source=SuggestionSource.PROFILER,
                    metrics={"slowdown": "up to 10x in large loops"},
                    evidence="Frequent sync points block the CUDA stream pipeline",
                    confidence=0.95
                )
                
                suggestions.append(CodeSuggestion(
                    id=str(uuid.uuid4()),
                    title="Avoid Item() In Loops",
                    description="Calling .item() inside loops creates GPU-CPU synchronization overhead",
                    code_example="""
# Instead of:
running_loss = 0
for inputs, labels in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    running_loss += loss.item()  # Forces sync in each iteration
    
# Use:
losses = []
for inputs, labels in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    losses.append(loss)
    
# Only sync once at the end
running_loss = torch.stack(losses).sum().item()
                    """,
                    severity="high",
                    justification=justification
                ))
                
        return suggestions


def register_plugin():
    """Register this plugin with the suggestion system."""
    return {
        "class": TorchOptimizationPlugin,
        "instance": TorchOptimizationPlugin()
    }
