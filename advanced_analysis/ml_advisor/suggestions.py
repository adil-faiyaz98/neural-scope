"""
ML-based optimization suggestions for machine learning models.

This module uses machine learning to analyze models and provide targeted
optimization recommendations based on model architecture, execution patterns,
and hardware utilization.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import hashlib
import json
import traceback
from functools import lru_cache

logger = logging.getLogger(__name__)

# Version tracking for suggestions database
SUGGESTIONS_VERSION = "1.0.0"

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch-specific features will be disabled.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. TensorFlow-specific features will be disabled.")

@dataclass
class OptimizationSuggestion:
    """Represents a specific optimization suggestion"""
    category: str  # Category of optimization (e.g., "memory", "computation", "architecture")
    priority: int  # Priority level (1-5, with 5 being highest)
    description: str  # Human-readable description
    estimated_impact: str  # Estimated impact (e.g., "20% speedup", "50% memory reduction")
    code_example: str  # Example code implementing the suggestion
    applicable_frameworks: List[str]  # Frameworks this suggestion applies to
    confidence_score: float = 0.8  # Confidence in the suggestion (0.0-1.0)
    verification_status: str = "unverified"  # Status: "verified", "unverified", "experimental"

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to dictionary for serialization"""
        return {
            "category": self.category,
            "priority": self.priority,
            "description": self.description,
            "estimated_impact": self.estimated_impact,
            "code_example": self.code_example,
            "applicable_frameworks": self.applicable_frameworks,
            "confidence_score": self.confidence_score,
            "verification_status": self.verification_status
        }


class MLAdvisor:
    """
    ML-based advisor for model optimization that provides intelligent,
    targeted recommendations based on model analysis.
    """
    
    def __init__(self, model=None, framework=None, cache_results=True):
        self.model = model
        self.cache_results = cache_results
        
        # Auto-detect framework if not specified
        if framework is None and model is not None:
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                framework = "pytorch"
            elif TF_AVAILABLE and isinstance(model, tf.Module):
                framework = "tensorflow"
            else:
                framework = "unknown"
                logger.warning(f"Unknown model type: {type(model)}. Some features may be limited.")
                
        self.framework = framework
        self.analysis_results = {}
        self._analysis_start_time = None
        self._analysis_duration = None
        self._analysis_cache = {}
        
    @lru_cache(maxsize=32)
    def _get_model_fingerprint(self, model):
        """Generate a fingerprint for model caching purposes"""
        try:
            if self.framework == "pytorch" and TORCH_AVAILABLE:
                param_string = str([p.shape for p in model.parameters()])
                return hashlib.md5(param_string.encode()).hexdigest()
            elif self.framework == "tensorflow" and TF_AVAILABLE:
                return hashlib.md5(str(model.get_config()).encode()).hexdigest()
            else:
                return hashlib.md5(str(id(model)).encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate model fingerprint: {e}")
            return str(id(model))
        
    def analyze_model(self, model=None):
        """
        Analyze model architecture and generate optimization suggestions
        
        Args:
            model: Model to analyze (uses model from initialization if None)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if model is not None:
                self.model = model
                
            if self.model is None:
                raise ValueError("No model provided for analysis")
            
            # Check cache if enabled
            if self.cache_results:
                fingerprint = self._get_model_fingerprint(self.model)
                if fingerprint in self._analysis_cache:
                    logger.info("Using cached analysis results")
                    return self._analysis_cache[fingerprint]
                
            self._analysis_start_time = time.time()
                
            if self.framework == "pytorch":
                result = self._analyze_pytorch_model()
            elif self.framework == "tensorflow":
                result = self._analyze_tensorflow_model()
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
            
            # Update analysis timing
            self._analysis_duration = time.time() - self._analysis_start_time
            result["analysis_metadata"] = {
                "duration_seconds": self._analysis_duration,
                "timestamp": time.time(),
                "suggestions_version": SUGGESTIONS_VERSION
            }
            
            # Cache results
            if self.cache_results:
                self._analysis_cache[fingerprint] = result
                
            return result
            
        except Exception as e:
            error_details = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            logger.error(f"Model analysis failed: {e}", exc_info=True)
            self.analysis_results = {"error": error_details, "partial_results": self.analysis_results}
            return self.analysis_results
            
    def _analyze_pytorch_model(self):
        """Analyze PyTorch model architecture"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorch model analysis")
            
        # Implementation would go here
        # This would include:
        # - Analyzing model architecture
        # - Identifying optimization opportunities
        # - Generating suggestions
        
        # Placeholder for demonstration
        # TODO: Replace with actual analysis logic
        suggestions = [
            OptimizationSuggestion(
                category="memory",
                priority=5,
                description="Use checkpoint to reduce memory usage in large models",
                estimated_impact="30-50% memory reduction",
                code_example="from torch.utils.checkpoint import checkpoint\n\ndef forward(self, x):\n    x = checkpoint(self.block1, x)\n    return x",
                applicable_frameworks=["pytorch"]
            ),
            OptimizationSuggestion(
                category="computation",
                priority=4,
                description="Use torch.compile for faster execution",
                estimated_impact="20-30% speedup",
                code_example="model = torch.compile(model)",
                applicable_frameworks=["pytorch"]
            ),
            OptimizationSuggestion(
                category="architecture",
                priority=3,
                description="Replace ReLU with SiLU/Swish for better accuracy",
                estimated_impact="1-2% accuracy improvement",
                code_example="# Replace\n# self.act = nn.ReLU()\n# With\nself.act = nn.SiLU()",
                applicable_frameworks=["pytorch", "tensorflow"]
            )
        ]
        
        self.analysis_results = {
            "model_type": "pytorch",
            "layer_count": sum(1 for _ in self.model.modules()),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "suggestions": suggestions
        }
        
        return self.analysis_results
        
    def _analyze_tensorflow_model(self):
        """Analyze TensorFlow model architecture"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TensorFlow model analysis")
            
        # Implementation would go here
        # TODO: Replace with actual analysis logic
        
        # Placeholder for demonstration
        suggestions = [
            OptimizationSuggestion(
                category="memory",
                priority=5,
                description="Use mixed precision training",
                estimated_impact="40-60% memory reduction, 2-3x speedup",
                code_example="policy = tf.keras.mixed_precision.Policy('mixed_float16')\ntf.keras.mixed_precision.set_global_policy(policy)",
                applicable_frameworks=["tensorflow"],
                confidence_score=0.95,
                verification_status="verified"
            ),
            OptimizationSuggestion(
                category="computation",
                priority=4,
                description="Enable XLA compilation",
                estimated_impact="20-50% speedup",
                code_example="model = tf.function(model, jit_compile=True)",
                applicable_frameworks=["tensorflow"]
            ),
            OptimizationSuggestion(
                category="architecture",
                priority=3,
                description="Use tf.keras.layers.Embedding with mask_zero=True for variable length sequences",
                estimated_impact="Improved handling of variable length inputs",
                code_example="embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)",
                applicable_frameworks=["tensorflow"]
            )
        ]
        
        self.analysis_results = {
            "model_type": "tensorflow",
            "layer_count": len(self.model.layers) if hasattr(self.model, 'layers') else 0,
            "parameter_count": self.model.count_params() if hasattr(self.model, 'count_params') else 0,
            "suggestions": suggestions
        }
        
        return self.analysis_results
        
    def get_suggestions(self, category=None, min_priority=None, min_confidence=0.0):
        """
        Get optimization suggestions filtered by category and priority
        
        Args:
            category: Filter by category (e.g., "memory", "computation")
            min_priority: Minimum priority level (1-5)
            min_confidence: Minimum confidence score (0.0-1.0)
            
        Returns:
            List of OptimizationSuggestion objects
        """
        if not self.analysis_results or "suggestions" not in self.analysis_results:
            self.analyze_model()
            
        suggestions = self.analysis_results.get("suggestions", [])
        
        if category:
            suggestions = [s for s in suggestions if s.category == category]
            
        if min_priority:
            suggestions = [s for s in suggestions if s.priority >= min_priority]
            
        if min_confidence:
            suggestions = [s for s in suggestions if s.confidence_score >= min_confidence]
            
        return suggestions
        
    def generate_optimization_report(self):
        """
        Generate a comprehensive optimization report
        
        Returns:
            Dictionary with detailed optimization report
        """
        if not self.analysis_results:
            self.analyze_model()
            
        # Group suggestions by category
        suggestions_by_category = {}
        for suggestion in self.analysis_results.get("suggestions", []):
            if suggestion.category not in suggestions_by_category:
                suggestions_by_category[suggestion.category] = []
            suggestions_by_category[suggestion.category].append(suggestion)
            
        # Sort suggestions by priority within each category
        for category in suggestions_by_category:
            suggestions_by_category[category].sort(key=lambda x: x.priority, reverse=True)
            
        return {
            "model_summary": {
                "framework": self.framework,
                "layer_count": self.analysis_results.get("layer_count", 0),
                "parameter_count": self.analysis_results.get("parameter_count", 0)
            },
            "optimization_suggestions": suggestions_by_category,
            "high_priority_suggestions": [s for s in self.analysis_results.get("suggestions", []) if s.priority >= 4],
            "estimated_impact": "Implementing high-priority suggestions could result in 30-50% performance improvement"
        }

"""
ML Optimization Suggestion System

This module provides enhanced suggestions for ML code optimization,
including justifications, feedback collection, and AI-assisted fixes.
"""

import json
import os
import sqlite3
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Union, Callable, Type
from pathlib import Path
import importlib.util

# Configure logging
logger = logging.getLogger(__name__)

# Try to import AI providers if available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Local imports
from .inefficiency_detection import DetectionRule, RuleRegistry, RULES_VERSION

# Constants
SUGGESTION_VERSION = '0.1.0'
DEFAULT_DB_PATH = os.path.expanduser('~/.neural_scope/profiling_db.sqlite')


class SuggestionSource(Enum):
    """Source of the optimization suggestion."""
    RULE = "rule"
    PROFILER = "profiler"
    PATTERN = "pattern"
    AI = "ai"
    MANUAL = "manual"


class SuggestionFeedback(Enum):
    """User feedback on suggestions."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    IMPLEMENTED = "implemented"


@dataclass
class SuggestionJustification:
    """Justification for a code optimization suggestion."""
    reason: str
    source: SuggestionSource
    metrics: Dict[str, Any] = field(default_factory=dict)
    evidence: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert justification to a dictionary."""
        return {
            "reason": self.reason,
            "source": self.source.value,
            "metrics": self.metrics,
            "evidence": self.evidence,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuggestionJustification':
        """Create justification from a dictionary."""
        return cls(
            reason=data.get("reason", ""),
            source=SuggestionSource(data.get("source", "rule")),
            metrics=data.get("metrics", {}),
            evidence=data.get("evidence", ""),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class CodeSuggestion:
    """Enhanced code optimization suggestion with justification."""
    id: str
    title: str
    description: str
    code_example: str
    severity: str
    justification: SuggestionJustification
    rule_name: Optional[str] = None
    rule_version: str = RULES_VERSION
    feedback: SuggestionFeedback = SuggestionFeedback.NEUTRAL
    timestamp: float = field(default_factory=time.time)
    ai_explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to a dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "code_example": self.code_example,
            "severity": self.severity,
            "justification": self.justification.to_dict(),
            "rule_name": self.rule_name,
            "rule_version": self.rule_version,
            "feedback": self.feedback.value,
            "timestamp": self.timestamp,
            "ai_explanation": self.ai_explanation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSuggestion':
        """Create suggestion from a dictionary."""
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            code_example=data.get("code_example", ""),
            severity=data.get("severity", "medium"),
            justification=SuggestionJustification.from_dict(data.get("justification", {})),
            rule_name=data.get("rule_name"),
            rule_version=data.get("rule_version", RULES_VERSION),
            feedback=SuggestionFeedback(data.get("feedback", "neutral")),
            timestamp=data.get("timestamp", time.time()),
            ai_explanation=data.get("ai_explanation")
        )
    
    def record_feedback(self, feedback: SuggestionFeedback) -> None:
        """Record user feedback on this suggestion."""
        self.feedback = feedback
        # Update feedback in database
        ProfilingDatabase.get_instance().update_suggestion_feedback(self.id, feedback)


class ProfilingDatabase:
    """Database for persisting profiling data and suggestion feedback."""
    _instance = None
    
    @classmethod
    def get_instance(cls, db_path: str = DEFAULT_DB_PATH) -> 'ProfilingDatabase':
        """Get singleton instance of the database."""
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize the database connection."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema if not exists."""
        cursor = self.conn.cursor()
        
        # Create suggestions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS suggestions (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            code_example TEXT,
            severity TEXT,
            justification TEXT,
            rule_name TEXT,
            rule_version TEXT,
            feedback TEXT,
            timestamp REAL,
            ai_explanation TEXT
        )
        ''')
        
        # Create model_stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_stats (
            model_class TEXT PRIMARY KEY,
            total_instances INTEGER,
            inefficiencies_detected INTEGER,
            suggestions_implemented INTEGER,
            performance_metrics TEXT,
            last_updated REAL
        )
        ''')
        
        # Create feedback_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_history (
            suggestion_id TEXT,
            feedback TEXT,
            timestamp REAL,
            PRIMARY KEY (suggestion_id, timestamp)
        )
        ''')
        
        self.conn.commit()
    
    def store_suggestion(self, suggestion: CodeSuggestion) -> None:
        """Store a suggestion in the database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO suggestions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                suggestion.id,
                suggestion.title,
                suggestion.description,
                suggestion.code_example,
                suggestion.severity,
                json.dumps(suggestion.justification.to_dict()),
                suggestion.rule_name,
                suggestion.rule_version,
                suggestion.feedback.value,
                suggestion.timestamp,
                suggestion.ai_explanation
            )
        )
        self.conn.commit()
    
    def get_suggestion(self, suggestion_id: str) -> Optional[CodeSuggestion]:
        """Retrieve a suggestion from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM suggestions WHERE id = ?", (suggestion_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        return CodeSuggestion(
            id=row[0],
            title=row[1],
            description=row[2],
            code_example=row[3],
            severity=row[4],
            justification=SuggestionJustification.from_dict(json.loads(row[5])),
            rule_name=row[6],
            rule_version=row[7],
            feedback=SuggestionFeedback(row[8]),
            timestamp=row[9],
            ai_explanation=row[10]
        )
    
    def update_suggestion_feedback(self, suggestion_id: str, feedback: SuggestionFeedback) -> None:
        """Update feedback for a suggestion and record in history."""
        cursor = self.conn.cursor()
        
        # Update main suggestion record
        cursor.execute(
            "UPDATE suggestions SET feedback = ? WHERE id = ?",
            (feedback.value, suggestion_id)
        )
        
        # Add to feedback history
        cursor.execute(
            "INSERT INTO feedback_history VALUES (?, ?, ?)",
            (suggestion_id, feedback.value, time.time())
        )
        
        self.conn.commit()
    
    def update_model_stats(self, model_class: str, stats: Dict[str, Any]) -> None:
        """Update statistics for a model class."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO model_stats VALUES (?, ?, ?, ?, ?, ?)",
            (
                model_class,
                stats.get("total_instances", 0),
                stats.get("inefficiencies_detected", 0),
                stats.get("suggestions_implemented", 0),
                json.dumps(stats.get("performance_metrics", {})),
                time.time()
            )
        )
        self.conn.commit()
    
    def get_model_stats(self, model_class: str) -> Dict[str, Any]:
        """Get statistics for a model class."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM model_stats WHERE model_class = ?", (model_class,))
        row = cursor.fetchone()
        
        if not row:
            return {"model_class": model_class}
            
        return {
            "model_class": row[0],
            "total_instances": row[1],
            "inefficiencies_detected": row[2],
            "suggestions_implemented": row[3],
            "performance_metrics": json.loads(row[4]),
            "last_updated": row[5]
        }
    
    def get_all_model_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all model classes."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM model_stats")
        stats = []
        
        for row in cursor.fetchall():
            stats.append({
                "model_class": row[0],
                "total_instances": row[1],
                "inefficiencies_detected": row[2],
                "suggestions_implemented": row[3],
                "performance_metrics": json.loads(row[4]),
                "last_updated": row[5]
            })
            
        return stats
        
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


# Plugin base class
class SuggestionPlugin:
    """Base class for suggestion plugins."""
    name = "base_plugin"
    description = "Base suggestion plugin"
    version = "0.1.0"
    
    def __init__(self):
        """Initialize the plugin."""
        pass
    
    def analyze(self, code: str, context: Dict[str, Any]) -> List[CodeSuggestion]:
        """Analyze code and generate suggestions."""
        return []
    
    def get_rules(self) -> List[DetectionRule]:
        """Get rules provided by this plugin."""
        return []
    
    @classmethod
    def get_metadata(cls) -> Dict[str, str]:
        """Get plugin metadata."""
        return {
            "name": cls.name,
            "description": cls.description,
            "version": cls.version
        }


class SuggestionManager:
    """Manager for generating and tracking code optimization suggestions."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize the suggestion manager."""
        self.db = ProfilingDatabase.get_instance(db_path)
        self.plugins = {}
        self.load_plugins()
        
    def load_plugins(self) -> None:
        """Load suggestion plugins from the plugins directory."""
        plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
        if not os.path.exists(plugins_dir):
            os.makedirs(plugins_dir, exist_ok=True)
            
        for file_name in os.listdir(plugins_dir):
            if file_name.endswith('.py') and not file_name.startswith('_'):
                try:
                    module_name = file_name[:-3]
                    module_path = os.path.join(plugins_dir, file_name)
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    if hasattr(module, 'register_plugin'):
                        plugin_info = module.register_plugin()
                        self.plugins[module_name] = plugin_info
                        logger.info(f"Loaded plugin: {module_name}")
                except Exception as e:
                    logger.error(f"Error loading plugin {file_name}: {e}")
    
    def create_suggestion_from_rule(self, rule: DetectionRule) -> CodeSuggestion:
        """Create a suggestion from a detection rule."""
        justification = SuggestionJustification(
            reason=f"Rule-based detection: {rule.name}",
            source=SuggestionSource.RULE,
            confidence=1.0
        )
        
        return CodeSuggestion(
            id=str(uuid.uuid4()),
            title=rule.name.replace('_', ' ').title(),
            description=rule.description,
            code_example=rule.code_example,
            severity=rule.severity.value,
            justification=justification,
            rule_name=rule.name,
            rule_version=rule.version
        )
    
    def get_or_create_suggestion(self, rule_name: str) -> Optional[CodeSuggestion]:
        """Get or create a suggestion based on a rule name."""
        rule = RuleRegistry.get_rule(rule_name)
        if not rule:
            return None
            
        return self.create_suggestion_from_rule(rule)
    
    def add_profiler_justification(self, suggestion: CodeSuggestion, profiler_data: Dict[str, Any]) -> None:
        """Add profiler data to justify the suggestion."""
        suggestion.justification.source = SuggestionSource.PROFILER
        suggestion.justification.metrics = profiler_data
        
        # Extract key metrics to enhance the justification
        if "time_impact" in profiler_data:
            suggestion.justification.reason += f"\nProfiling evidence: {profiler_data['time_impact']:.1f}x slower execution"
        
        if "memory_impact" in profiler_data:
            suggestion.justification.reason += f"\nMemory impact: {profiler_data['memory_impact']:.1f}x higher memory usage"
            
        suggestion.justification.evidence = str(profiler_data)
        self.db.store_suggestion(suggestion)
    
    def record_feedback(self, suggestion_id: str, feedback: SuggestionFeedback) -> None:
        """Record user feedback for a suggestion."""
        suggestion = self.db.get_suggestion(suggestion_id)
        if suggestion:
            suggestion.record_feedback(feedback)
    
    def get_ai_explanation(self, suggestion: CodeSuggestion) -> Optional[str]:
        """Get AI explanation for a suggestion using OpenAI or Hugging Face."""
        if suggestion.ai_explanation:
            return suggestion.ai_explanation
            
        try:
            explanation = None
            
            # Try OpenAI first if available
            if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
                explanation = self._get_openai_explanation(suggestion)
                
            # Fall back to Hugging Face if available
            elif TRANSFORMERS_AVAILABLE:
                explanation = self._get_huggingface_explanation(suggestion)
                
            if explanation:
                suggestion.ai_explanation = explanation
                self.db.store_suggestion(suggestion)
                return explanation
                
        except Exception as e:
            logger.error(f"Error getting AI explanation: {e}")
            
        return None
    
    def _get_openai_explanation(self, suggestion: CodeSuggestion) -> Optional[str]:
        """Get explanation from OpenAI API."""
        try:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            prompt = f"""
            Explain why the following ML code inefficiency is problematic and how to fix it:
            
            Inefficiency: {suggestion.title}
            Description: {suggestion.description}
            
            Bad code example:
            {suggestion.code_example.split('# Use:')[0] if '# Use:' in suggestion.code_example else suggestion.code_example}
            
            Provide a detailed explanation of:
            1. Why this is inefficient
            2. What performance impact it has
            3. How the suggested solution improves performance
            4. Any potential trade-offs or considerations
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a machine learning code optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with OpenAI explanation: {e}")
            return None
    
    def _get_huggingface_explanation(self, suggestion: CodeSuggestion) -> Optional[str]:
        """Get explanation from Hugging Face model."""
        try:
            model_name = "distilgpt2"  # Using a simple model as example
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            prompt = f"Explain why this ML code inefficiency is problematic: {suggestion.title}. {suggestion.description}"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=200)
            
            explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the generated text
            if prompt in explanation:
                explanation = explanation[len(prompt):].strip()
                
            return explanation
        except Exception as e:
            logger.error(f"Error with Hugging Face explanation: {e}")
            return None
    
    def generate_suggestions_from_code(self, code: str, context: Dict[str, Any]) -> List[CodeSuggestion]:
        """Generate suggestions from code using all available plugins."""
        suggestions = []
        
        # Run each plugin's analysis
        for plugin_name, plugin_info in self.plugins.items():
            if "instance" in plugin_info:
                plugin_instance = plugin_info["instance"]
                plugin_suggestions = plugin_instance.analyze(code, context)
                suggestions.extend(plugin_suggestions)
        
        # Store all suggestions in the database
        for suggestion in suggestions:
            self.db.store_suggestion(suggestion)
            
        return suggestions

# Example usage of the system
def create_suggestion_with_profiling(code: str):
    from .inefficiency_detection import InefficiencyDetector
    
    # Analyze code for inefficiencies
    detector = InefficiencyDetector(code)
    results = detector.analyze_code()
    
    # Create suggestion manager
    manager = SuggestionManager()
    
    # Generate enhanced suggestions
    enhanced_suggestions = []
    
    # Get detected inefficiencies from the analysis
    for inefficiency in results.get("detected_inefficiencies", []):
        suggestion = manager.get_or_create_suggestion(inefficiency["name"])
        if suggestion:
            # Add simulated profiling data
            profiler_data = {
                "time_impact": 2.5,  # 2.5x slower than optimized version
                "memory_impact": 1.8,  # 1.8x more memory usage
                "function_name": "affected_function",
                "line_hotspots": [42, 57, 63],
                "summary": "This pattern causes significant performance degradation"
            }
            
            manager.add_profiler_justification(suggestion, profiler_data)
            
            # Get AI-powered explanation and fix
            ai_explanation = manager.get_ai_explanation(suggestion)
            if ai_explanation:
                suggestion.ai_explanation = ai_explanation
            
            enhanced_suggestions.append(suggestion)
    
    return enhanced_suggestions
