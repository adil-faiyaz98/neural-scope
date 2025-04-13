"""
ML code inefficiency detection module.

This module provides tools for detecting common inefficiencies in machine learning code,
including inefficient data processing, suboptimal algorithm usage, and performance bottlenecks.
"""

import ast
import re
import logging
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Union, Callable, Tuple, ClassVar
import importlib.metadata


logger = logging.getLogger(__name__)

# Version information
__version__ = '0.2.0'  # Semantic versioning

# Rule versioning schema
RULES_VERSION = '0.2.0'

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DetectionRule:
    """Base class for inefficiency detection rules."""
    name: str
    description: str
    suggestion: str
    severity: Severity
    code_example: str
    version: str = RULES_VERSION
    enabled: bool = True
    
    def check(self, node: ast.AST, context: Dict[str, Any]) -> bool:
        """Check if the rule applies to the given node."""
        raise NotImplementedError("Each rule must implement this method")

@dataclass
class RegexRule(DetectionRule):
    """Rule based on regex pattern matching."""
    pattern: str = ""
    
    def check(self, code_str: str, context: Dict[str, Any]) -> bool:
        return bool(re.search(self.pattern, code_str, re.MULTILINE))

@dataclass
class ASTRule(DetectionRule):
    """Rule based on AST pattern matching."""
    def check(self, node: ast.AST, context: Dict[str, Any]) -> bool:
        return False

@dataclass
class PandasIterrowsRule(ASTRule):
    """Detects inefficient use of pandas iterrows()."""
    def check(self, node: ast.AST, context: Dict[str, Any]) -> bool:
        if not isinstance(node, ast.For):
            return False
            
        # Check if the iteration target is a call to iterrows()
        if not isinstance(node.iter, ast.Call):
            return False
            
        # Check if the call is to a method named iterrows
        if not hasattr(node.iter, 'func') or not isinstance(node.iter.func, ast.Attribute):
            return False
            
        return node.iter.func.attr == 'iterrows'

@dataclass
class NestedLoopRule(ASTRule):
    """Detects inefficient nested loops."""
    def check(self, node: ast.AST, context: Dict[str, Any]) -> bool:
        if not isinstance(node, ast.For):
            return False
            
        # Check if there's another For loop in the body
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.For) and child_node != node:
                # Check if the inner loop is within the body of the outer loop
                for outer_body_node in node.body:
                    if child_node in ast.walk(outer_body_node):
                        # Check context to see if we're likely operating on large data
                        if context.get('dataframe_var_names', set()) or context.get('array_var_names', set()):
                            return True
        return False

@dataclass
class Config:
    """Configuration for the inefficiency detector."""
    enabled_rules: Set[str] = field(default_factory=set)
    disabled_rules: Set[str] = field(default_factory=set)
    min_severity: Severity = Severity.LOW
    detect_context: bool = True
    max_suggestions: int = 10
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a config from a dictionary."""
        return cls(
            enabled_rules=set(config_dict.get('enabled_rules', [])),
            disabled_rules=set(config_dict.get('disabled_rules', [])),
            min_severity=Severity(config_dict.get('min_severity', 'low')),
            detect_context=config_dict.get('detect_context', True),
            max_suggestions=config_dict.get('max_suggestions', 10)
        )
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """Load configuration from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
                return cls.from_dict(config_dict)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load config from {file_path}: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary."""
        return {
            'enabled_rules': list(self.enabled_rules),
            'disabled_rules': list(self.disabled_rules),
            'min_severity': self.min_severity.value,
            'detect_context': self.detect_context,
            'max_suggestions': self.max_suggestions
        }
    
    def to_file(self, file_path: str) -> None:
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Define Rule Registry
class RuleRegistry:
    """Registry for all available detection rules."""
    _rules: ClassVar[Dict[str, DetectionRule]] = {}
    
    @classmethod
    def register(cls, rule: DetectionRule) -> None:
        """Register a detection rule."""
        cls._rules[rule.name] = rule
    
    @classmethod
    def get_rule(cls, name: str) -> Optional[DetectionRule]:
        """Get a rule by name."""
        return cls._rules.get(name)
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, DetectionRule]:
        """Get all registered rules."""
        return cls._rules.copy()
    
    @classmethod
    def load_rules_from_file(cls, file_path: str) -> None:
        """Load rules from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                rules_dict = json.load(f)
                for name, rule_dict in rules_dict.items():
                    if rule_dict.get('type') == 'regex':
                        cls.register(RegexRule(
                            name=name,
                            description=rule_dict.get('description', ''),
                            suggestion=rule_dict.get('suggestion', ''),
                            severity=Severity(rule_dict.get('severity', 'medium')),
                            code_example=rule_dict.get('code_example', ''),
                            pattern=rule_dict.get('pattern', ''),
                            version=rule_dict.get('version', RULES_VERSION),
                            enabled=rule_dict.get('enabled', True)
                        ))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load rules from {file_path}: {e}")

# Initialize rule registry with built-in rules
# Common inefficiency patterns as ASTRules
INEFFICIENCY_PATTERNS = {
    "pandas_iterrows": {
        "description": "Using pandas iterrows() is inefficient for large dataframes",
        "suggestion": "Use vectorized operations or apply() instead of iterrows()",
        "severity": "high",
        "code_example": """
# Instead of:
for index, row in df.iterrows():
    result[index] = row['a'] * 2

# Use:
df['result'] = df['a'] * 2
"""
    },
    "nested_loops": {
        "description": "Inefficient nested loops on large data structures",
        "suggestion": "Consider vectorizing operations or using numpy/pandas operations",
        "severity": "high",
        "code_example": """
# Instead of:
for i in range(len(data)):
    for j in range(len(data)):
        result[i][j] = data[i] * data[j]

# Use:
result = np.outer(data, data)
"""
    }
}

pandas_iterrows_rule = PandasIterrowsRule(
    name="pandas_iterrows",
    description="Using pandas iterrows() is inefficient for large dataframes",
    suggestion="Use vectorized operations or apply() instead of iterrows()",
    severity=Severity.HIGH,
    code_example="""
# Instead of:
for index, row in df.iterrows():
    result[index] = row['a'] * 2

# Use:
df['result'] = df['a'] * 2
"""
)

nested_loop_rule = NestedLoopRule(
    name="nested_loops",
    description="Nested loops can lead to O(n²) complexity",
    suggestion="Use vectorized operations or more efficient algorithms",
    severity=Severity.HIGH,
    code_example="""
# Instead of:
result = []
for i in range(len(data)):
    for j in range(len(data[i])):
        result.append(data[i][j] * 2)

# Use:
import numpy as np
result = np.array(data) * 2
"""
)

# Register AST-based rules
RuleRegistry.register(pandas_iterrows_rule)
RuleRegistry.register(nested_loop_rule)

# Convert existing regex patterns to RegexRules and register them
for name, info in INEFFICIENCY_PATTERNS.items():
    if name not in ["pandas_iterrows", "nested_loops"]:  # Skip those we've converted to AST rules
        RuleRegistry.register(RegexRule(
            name=name,
            description=info["description"],
            suggestion=info["suggestion"],
            severity=Severity(info["severity"]),
            code_example=info["code_example"],
            pattern=info["pattern"]
        ))

class ContextDetector(ast.NodeVisitor):
    """Detects context information from code to improve detection accuracy."""
    
    def __init__(self):
        self.context = {
            'imports': set(),
            'dataframe_var_names': set(),
            'array_var_names': set(),
            'tensor_var_names': set(),
            'loop_vars': set(),
            'function_calls': {},
            'class_definitions': set(),
            'batch_sizes': {},
        }
    
    def visit_Import(self, node):
        for name in node.names:
            self.context['imports'].add(name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.context['imports'].add(node.module)
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Detect dataframe and array assignments
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == 'DataFrame':
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.context['dataframe_var_names'].add(target.id)
            elif isinstance(func, ast.Attribute):
                if func.attr == 'DataFrame' and isinstance(func.value, ast.Name) and func.value.id == 'pd':
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.context['dataframe_var_names'].add(target.id)
                elif func.attr == 'array' and isinstance(func.value, ast.Name) and func.value.id == 'np':
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.context['array_var_names'].add(target.id)
                elif func.attr == 'tensor' and isinstance(func.value, ast.Name) and func.value.id in ('torch', 'tf'):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.context['tensor_var_names'].add(target.id)
            
            # Detect batch_size assignments
            if isinstance(func, ast.Name) and func.id == 'DataLoader':
                for kw in node.value.keywords:
                    if kw.arg == 'batch_size' and isinstance(kw.value, ast.Constant):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.context['batch_sizes'][target.id] = kw.value.value
        
        self.generic_visit(node)
    
    def visit_For(self, node):
        if isinstance(node.target, ast.Name):
            self.context['loop_vars'].add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.context['loop_vars'].add(elt.id)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self.context['function_calls']:
                self.context['function_calls'][func_name] = 0
            self.context['function_calls'][func_name] += 1
        elif isinstance(node.func, ast.Attribute):
            if hasattr(node.func, 'attr'):
                method_name = node.func.attr
                if method_name not in self.context['function_calls']:
                    self.context['function_calls'][method_name] = 0
                self.context['function_calls'][method_name] += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.context['class_definitions'].add(node.name)
        self.generic_visit(node)

class InefficiencyDetector:
    """
    Detects common inefficiencies in ML code using AST and regex analysis.
    """
    
    def __init__(self, code_str: str, config: Optional[Config] = None):
        """
        Initialize the inefficiency detector with code to analyze.
        
        Args:
            code_str: Python code as a string
            config: Configuration options
        """
        self.code_str = code_str
        self.config = config or Config()
        self.context = {}
        try:
            self.tree = ast.parse(code_str)
            # Extract context information
            if self.config.detect_context:
                context_detector = ContextDetector()
                context_detector.visit(self.tree)
                self.context = context_detector.context
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            self.tree = None

    def detect_inefficiencies(self) -> List[Dict[str, Any]]:
        """
        Detect inefficiencies in the code.
        
        Returns:
            List of dictionaries with inefficiency information
        """
        inefficiencies = []
        
        if not self.tree:
            return inefficiencies
            
        # Get enabled rules based on configuration
        enabled_rules = {}
        all_rules = RuleRegistry.get_all_rules()
        
        for name, rule in all_rules.items():
            # Skip rules that don't meet the minimum severity threshold
            if Severity[rule.severity.name].value < Severity[self.config.min_severity.name].value:
                continue
                
            # Determine if the rule is enabled
            explicitly_enabled = name in self.config.enabled_rules
            explicitly_disabled = name in self.config.disabled_rules
            default_enabled = rule.enabled
            
            if explicitly_enabled or (default_enabled and not explicitly_disabled):
                enabled_rules[name] = rule
        
        # Apply AST-based rules
        for name, rule in enabled_rules.items():
            if isinstance(rule, ASTRule):
                for node in ast.walk(self.tree):
                    if rule.check(node, self.context):
                        inefficiencies.append({
                            "name": name,
                            "description": rule.description,
                            "suggestion": rule.suggestion,
                            "severity": rule.severity.value,
                            "code_example": rule.code_example,
                            "rule_version": rule.version
                        })
                        break  # Only report each rule violation once
            elif isinstance(rule, RegexRule):
                if rule.check(self.code_str, self.context):
                    inefficiencies.append({
                        "name": name,
                        "description": rule.description,
                        "suggestion": rule.suggestion,
                        "severity": rule.severity.value,
                        "code_example": rule.code_example,
                        "rule_version": rule.version
                    })
                
        # Limit the number of suggestions based on configuration
        return inefficiencies[:self.config.max_suggestions]
        
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions for detected inefficiencies.
        
        Returns:
            List of dictionaries with optimization suggestions
        """
        inefficiencies = self.detect_inefficiencies()
        suggestions = []
        
        for inefficiency in inefficiencies:
            suggestions.append({
                "type": "inefficiency",
                "severity": inefficiency["severity"],
                "message": inefficiency["description"],
                "details": inefficiency["suggestion"],
                "code_example": inefficiency["code_example"],
                "rule_version": inefficiency.get("rule_version", RULES_VERSION)
            })
                
        return suggestions
        
    def analyze_code(self) -> Dict[str, Any]:
        """
        Analyze code to detect inefficiencies and provide optimization suggestions.
        
        Returns:
            Dictionary with analysis results
        """
        inefficiencies = self.detect_inefficiencies()
        suggestions = self.get_optimization_suggestions()
        auto_fixes = self.generate_auto_fixes()
        
        return {
            "version": __version__,
            "rules_version": RULES_VERSION,
            "detected_inefficiencies": inefficiencies,
            "optimization_suggestions": suggestions,
            "detected_context": self.context,
            "auto_fixes": auto_fixes,
            "config": self.config.to_dict()
        }

    @staticmethod
    def get_version_info() -> Dict[str, str]:
        """
        Get version information for the inefficiency detector.
        
        Returns:
            Dictionary with version information
        """
        dependencies = {
            "python": ".".join(map(str, __import__('sys').version_info[:3])),
            "ast": ast.__version__ if hasattr(ast, "__version__") else "built-in"
        }
        
        # Add versions for common ML libraries if available
        for lib in ["numpy", "pandas", "torch", "tensorflow"]:
            try:
                dependencies[lib] = importlib.metadata.version(lib)
            except importlib.metadata.PackageNotFoundError:
                pass
                
        return {
            "version": __version__,
            "rules_version": RULES_VERSION,
            "dependencies": dependencies
        }

    @classmethod
    def from_config_file(cls, code_str: str, config_path: str) -> 'InefficiencyDetector':
        """
        Create an inefficiency detector from a config file.
        
        Args:
            code_str: Python code as a string
            config_path: Path to the config file
            
        Returns:
            InefficiencyDetector instance
        """
        config = Config.from_file(config_path)
        return cls(code_str, config)

# Auto-fix hooks functionality
@dataclass
class AutoFix:
    """Base class for auto-fix implementations that can rewrite code."""
    name: str
    description: str
    
    def can_fix(self, node: ast.AST, context: Dict[str, Any]) -> bool:
        """Check if this auto-fix can handle the given node."""
        return False
        
    def generate_fix(self, node: ast.AST, source_code: str) -> Optional[str]:
        """Generate fixed code for the given node."""
        return None

@dataclass
class DataLoaderFixHook(AutoFix):
    """Auto-fixes DataLoader recreation in loops."""
    
    def can_fix(self, node: ast.AST, context: Dict[str, Any]) -> bool:
        if not isinstance(node, ast.For):
            return False
            
        # Check if there's a DataLoader creation in the loop body
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == 'DataLoader':
                if child not in ast.walk(node.iter):  # Make sure it's not part of the iterator
                    return True
        return False
        
    def generate_fix(self, node: ast.For, source_code: str) -> Optional[str]:
        """Move DataLoader creation before the loop."""
        import astor
        
        # Find DataLoader calls in the body
        dataloader_assignments = []
        for child in node.body:
            if isinstance(child, ast.Assign):
                if isinstance(child.value, ast.Call) and isinstance(child.value.func, ast.Name) and child.value.func.id == 'DataLoader':
                    dataloader_assignments.append(child)
        
        if not dataloader_assignments:
            return None
            
        # Get the original source lines
        original_source = source_code.splitlines()
        
        # Extract the for loop lines
        loop_start_line = node.lineno - 1  # 0-indexed
        loop_end_line = max(n.end_lineno if hasattr(n, 'end_lineno') else n.lineno for n in ast.walk(node)) - 1
        
        # Generate the fixed code
        fixed_code = []
        
        # Add the DataLoader assignments before the loop
        for assignment in dataloader_assignments:
            fixed_code.append(astor.to_source(assignment).strip())
            
        # Add the loop without the DataLoader assignments
        loop_code = original_source[loop_start_line:loop_end_line+1]
        for assignment in dataloader_assignments:
            if hasattr(assignment, 'lineno'):
                assign_line = assignment.lineno - 1
                loop_code[assign_line - loop_start_line] = "# " + loop_code[assign_line - loop_start_line] + " (moved above)"
                
        fixed_code.extend(loop_code)
        
        return "\n".join(fixed_code)

class InefficiencyDetector:
    # ...existing code...
    
    def __init__(self, code_str: str, config: Optional[Config] = None):
        # ...existing code...
        self.auto_fixes = []
        self.register_default_auto_fixes()
        
    def register_default_auto_fixes(self) -> None:
        """Register default auto-fix hooks."""
        self.register_auto_fix(DataLoaderFixHook(
            name="dataloader_fix",
            description="Move DataLoader creation outside of loops"
        ))
        
    def register_auto_fix(self, auto_fix: AutoFix) -> None:
        """Register an auto-fix hook."""
        self.auto_fixes.append(auto_fix)
        
    def generate_auto_fixes(self) -> Dict[str, str]:
        """Generate automatic fixes for detected inefficiencies."""
        fixes = {}
        
        if not self.tree:
            return fixes
            
        for node in ast.walk(self.tree):
            for auto_fix in self.auto_fixes:
                if auto_fix.can_fix(node, self.context):
                    fixed_code = auto_fix.generate_fix(node, self.code_str)
                    if fixed_code:
                        fixes[auto_fix.name] = fixed_code
                        
        return fixes
    
    def analyze_code(self) -> Dict[str, Any]:
        """Analyze code to detect inefficiencies and provide optimization suggestions."""
        inefficiencies = self.detect_inefficiencies()
        suggestions = self.get_optimization_suggestions()
        auto_fixes = self.generate_auto_fixes()
        
        return {
            "version": __version__,
            "rules_version": RULES_VERSION,
            "detected_inefficiencies": inefficiencies,
            "optimization_suggestions": suggestions,
            "detected_context": self.context,
            "auto_fixes": auto_fixes,
            "config": self.config.to_dict()
        }
