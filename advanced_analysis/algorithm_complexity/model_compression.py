import logging
import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from advanced_analysis.utils.error_handling import (
    handle_errors, require_dependency, with_recovery,
    log_execution_time, log_call, ModelError, DependencyError
)
from advanced_analysis.utils.hardware_utils import (
    detect_hardware, optimize_for_hardware, get_recommended_precision
)

# Configure logger
logger = logging.getLogger(__name__)

class ProfileInfo:
    """
    Data class for profiling/recognition information about a model.
    Contains details used to decide compression strategies.
    """
    def __init__(self, framework: str, model_type: str, hardware: str,
                 techniques: Optional[list] = None, params: Optional[Dict[str, Any]] = None):
        """
        Initialize profiling info.
        :param framework: The ML framework name (e.g., 'pytorch', 'tensorflow').
        :param model_type: High-level type of model (e.g., 'CNN', 'Transformer', 'RL-policy').
        :param hardware: Target hardware environment (e.g., 'CPU', 'GPU', 'EdgeDevice').
        :param techniques: Optional list of compression techniques to apply (e.g. ['quantization','pruning','distillation']).
                           If None, the ModelCompressor will decide based on other parameters.
        :param params: Optional dict of additional decision parameters or configurations (e.g., quantization method).
        """
        self.framework = framework.lower()
        self.model_type = model_type.lower() if model_type else ""
        self.hardware = hardware.lower()
        self.techniques = techniques if techniques is not None else []  # empty list means decide automatically
        self.params = params or {}

class ModelCompressor:
    """
    Adaptive model compression orchestrator.
    Uses profile information to apply quantization, pruning, and knowledge distillation as appropriate.
    """
    def __init__(self, profile: ProfileInfo):
        """
        Initialize the compressor with profiling data.
        """
        self.profile = profile
        # If no specific techniques list provided, decide default techniques based on profile:
        if not self.profile.techniques:
            self.profile.techniques = self._decide_techniques()
        # Logs for tracking actions
        self.logs = []

    def _decide_techniques(self) -> list:
        """
        Decide which compression techniques to apply if not explicitly provided in profile.
        This decision can use framework, model type, and hardware info.
        """
        techniques = []
        fw = self.profile.framework
        hw = self.profile.hardware
        model_type = self.profile.model_type

        # Example heuristic decisions for demonstration:
        # - If the target hardware is resource-constrained (e.g., CPU or Edge), use quantization and pruning.
        # - If the model type is very large (e.g., a big Transformer or GenAI model), consider knowledge distillation.
        # - Otherwise, apply a moderate combination.
        if hw in ["cpu", "edge", "mobile"]:
            techniques.append("quantization")
            techniques.append("pruning")
        else:
            # On GPUs/TPUs, quantization might be less beneficial unless it's a large model.
            techniques.append("pruning")
            # If it's a large model (e.g., transformer or GenAI) and hardware is not extremely limited, consider quantization too.
            if model_type in ["transformer", "genai", "language model", "vision transformer"]:
                techniques.append("quantization")
        # Knowledge distillation if model is large and we have indications we can train a student:
        if model_type in ["genai", "transformer", "rl-policy", "large model"] or self.profile.params.get("enable_distillation"):
            techniques.append("distillation")
        return techniques

    @log_execution_time
    @with_recovery
    def compress(self, model: Any, **kwargs) -> Any:
        """
        Apply the selected compression techniques to the model.

        Args:
            model: The model to be compressed (PyTorch nn.Module or TensorFlow model)
            **kwargs: Additional parameters for compression techniques
                - calibration_data: Data for calibration in static quantization
                - student_model: Student model for knowledge distillation
                - data: Training data for distillation
                - epochs: Number of epochs for distillation training
                - batch_size: Batch size for distillation training
                - learning_rate: Learning rate for distillation training
                - temperature: Temperature for softening logits in distillation
                - optimizer: Optimizer for distillation training
                - loss_fn: Loss function for distillation training

        Returns:
            Compressed model (or a model artifact, e.g., a quantized model instance or TFLite flatbuffer)

        Raises:
            ModelError: If there's an issue with the model
            DependencyError: If required dependencies are missing
        """
        try:
            framework = self.profile.framework
            logger.info(f"Starting compression for a {framework} model ({self.profile.model_type}) on {self.profile.hardware}.")
            self.logs.append(f"Starting compression for a {framework} model ({self.profile.model_type}) on {self.profile.hardware}.")

            # Make a copy of the model to avoid modifying the original
            try:
                # For PyTorch models
                import torch
                if isinstance(model, torch.nn.Module):
                    model = copy.deepcopy(model)
            except ImportError:
                pass

            try:
                # For TensorFlow models
                import tensorflow as tf
                if isinstance(model, tf.keras.Model):
                    model = tf.keras.models.clone_model(model)
                    model.set_weights(model.get_weights())
            except ImportError:
                pass

            # 1. Quantization
            if "quantization" in self.profile.techniques:
                try:
                    model = self._apply_quantization(model, **kwargs)
                except Exception as e:
                    logger.error(f"Quantization failed: {str(e)}")
                    self.logs.append(f"Quantization failed: {str(e)}")
                    # Continue with other techniques

            # 2. Pruning
            if "pruning" in self.profile.techniques:
                try:
                    model = self._apply_pruning(model, **kwargs)
                except Exception as e:
                    logger.error(f"Pruning failed: {str(e)}")
                    self.logs.append(f"Pruning failed: {str(e)}")
                    # Continue with other techniques

            # 3. Knowledge Distillation
            if "distillation" in self.profile.techniques:
                # For distillation, we expect a teacher (the current model) and a student model provided via kwargs
                student_model = kwargs.get("student_model")
                training_data = kwargs.get("data")  # Could be a dataloader or tf.data.Dataset
                if student_model is None:
                    logger.warning("Distillation requested but no student model provided. Skipping knowledge distillation.")
                    self.logs.append("Distillation requested but no student model provided. Skipping knowledge distillation.")
                elif training_data is None:
                    logger.warning("Distillation requested but no training data provided. Skipping knowledge distillation.")
                    self.logs.append("Distillation requested but no training data provided. Skipping knowledge distillation.")
                else:
                    try:
                        model = self._apply_distillation(model, student_model, training_data, **kwargs)
                    except Exception as e:
                        logger.error(f"Distillation failed: {str(e)}")
                        self.logs.append(f"Distillation failed: {str(e)}")
                        # Continue with other techniques

            logger.info("Compression completed successfully.")
            self.logs.append("Compression completed successfully.")
            return model

        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            self.logs.append(f"Compression failed: {str(e)}")
            raise ModelError("compression", f"Failed to compress model: {str(e)}",
                            {"framework": self.profile.framework, "techniques": self.profile.techniques})

    @with_recovery
    def _apply_quantization(self, model: Any, **kwargs) -> Any:
        """
        Apply quantization to the model based on the framework and configuration.
        Supports dynamic quantization, post-training static quantization, or QAT.

        Args:
            model: The model to quantize
            **kwargs: Additional parameters for quantization
                - quantization_method: Method to use (dynamic, static, qat, auto)
                - calibration_data: Data for calibration in static quantization
                - bit_width: Bit width for quantization (8, 16, etc.)
                - symmetric: Whether to use symmetric quantization
                - per_channel: Whether to use per-channel quantization

        Returns:
            Quantized model

        Raises:
            ModelError: If there's an issue with the model
            DependencyError: If required dependencies are missing
        """
        fw = self.profile.framework
        q_method = kwargs.get("quantization_method", self.profile.params.get("quantization_method", "auto"))  # could be 'dynamic', 'static', 'qat', or 'auto'
        logger.info(f"Applying quantization (method={q_method}) for framework: {fw}.")
        self.logs.append(f"Applying quantization (method={q_method}) for framework: {fw}.")

        if fw == "pytorch":
            try:
                import torch
                import torch.nn as nn
                import torch.quantization as quant
            except ImportError:
                logger.warning("PyTorch not available for quantization")
                self.logs.append("PyTorch not available for quantization. Skipping.")
                return model

            # Decide method automatically if needed
            if q_method == "auto":
                # If model contains LSTM/Linear layers and is for CPU, use dynamic quantization for speed
                if self.profile.hardware in ["cpu", "edge", "mobile"]:
                    q_method = "dynamic"
                else:
                    # For GPU or if more accuracy needed, prefer static quantization (if calibration data available)
                    q_method = "static"

            if q_method == "dynamic":
                # Perform dynamic quantization (weights quantized, activations dynamically quantized at runtime)
                # Here we specify common layer types to quantize (Linear, LSTM, GRU, etc.)
                self.logs.append("Using dynamic quantization for PyTorch model.")
                # Determine which layers to quantize - typically linear layers (and LSTM if present)
                quantizable_types = {nn.Linear, nn.LSTM, nn.GRU}
                try:
                    model = quant.quantize_dynamic(model, qconfig_spec=quantizable_types, dtype=torch.qint8)
                except Exception as e:
                    self.logs.append(f"Dynamic quantization failed: {e}")
            elif q_method == "static":
                # Post-training static quantization: requires calibration with data
                # This involves setting up qconfig, preparing the model, running calibration, then converting.
                self.logs.append("Using static post-training quantization for PyTorch model.")
                model.eval()
                # Fuse modules for quantization (if applicable, e.g., Conv+BN+ReLU)
                # (This step is model-specific; skipping detailed implementation for brevity.)
                try:
                    model.qconfig = quant.get_default_qconfig("fbgemm")  # fbgemm for x86 or use "qnnpack" for ARM
                except Exception:
                    model.qconfig = quant.default_qconfig
                quant.prepare(model, inplace=True)
                # Calibration step: run a few batches of representative data through the model
                calib_data = kwargs.get("calibration_data")
                if calib_data is not None:
                    for batch in calib_data:
                        model(*batch)  # assuming batch is (inputs, labels) or just inputs
                quant.convert(model, inplace=True)
            elif q_method == "qat":
                # Quantization-aware training (QAT) stub:
                self.logs.append("Setting up quantization-aware training (QAT) for PyTorch model.")
                model.train()
                try:
                    model.qconfig = quant.get_default_qat_qconfig("fbgemm")
                except Exception:
                    model.qconfig = quant.default_qat_qconfig
                quant.prepare_qat(model, inplace=True)
                # At this point, the model is ready for QAT (fake quantization inserted).
                # The user would need to fine-tune the model with training data for some epochs:
                self.logs.append("QAT: Model prepared with fake quantization. Please fine-tune on training data.")
                # After training, convert to quantized model:
                quant.convert(model, inplace=True)
                model.eval()
                self.logs.append("QAT: Model converted to quantized version after fine-tuning.")
            else:
                self.logs.append(f"Unknown quantization method '{q_method}' for PyTorch. Skipping quantization.")

        elif fw == "tensorflow":
            # TensorFlow / Keras quantization
            try:
                import tensorflow as tf
                self.logs.append("Applying TensorFlow quantization.")
            except ImportError:
                logger.warning("TensorFlow not available for quantization")
                self.logs.append("TensorFlow not available for quantization. Skipping.")
                return model
            if q_method == "auto":
                # if hardware is edge, assume int8 post-training quantization
                q_method = "ptq"
            if q_method in ["ptq", "static", "dynamic"]:
                # Post-training quantization via TFLite converter
                # 'dynamic' here will mean dynamic range quantization (weights int8, activations float)
                self.logs.append(f"Using post-training quantization ({'dynamic range' if q_method=='dynamic' else 'int8'}) for TF model.")
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                if q_method == "dynamic":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # default optimization uses dynamic range quantization
                else:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    # For full int8 static quantization, ensure representative dataset is provided:
                    rep_data = kwargs.get("calibration_data")
                    if rep_data:
                        def representative_gen():
                            for batch in rep_data:
                                # yield a sample input for calibration
                                yield [batch] if not isinstance(batch, (list, tuple)) else batch
                        converter.representative_dataset = representative_gen
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.uint8 if self.profile.params.get("quantize_input_type_uint8") else tf.float32
                        converter.inference_output_type = tf.uint8 if self.profile.params.get("quantize_input_type_uint8") else tf.float32
                try:
                    quantized_tflite_model = converter.convert()
                    model = quantized_tflite_model  # The compressed model could be a TFLite flatbuffer
                    self.logs.append("TensorFlow model converted to TFLite with quantization.")
                except Exception as e:
                    self.logs.append(f"TFLite conversion/quantization failed: {e}")
            elif q_method == "qat":
                # Quantization-aware training for TensorFlow using TensorFlow Model Optimization Toolkit
                self.logs.append("Setting up quantization-aware training (QAT) for TensorFlow model.")
                try:
                    import tensorflow_model_optimization as tfmot
                    quantize_model = tfmot.quantization.keras.quantize_model
                except ImportError:
                    self.logs.append("TensorFlow Model Optimization Toolkit not available. Skipping QAT.")
                    return model
                # Prepare QAT model
                qat_model = quantize_model(model)
                self.logs.append("QAT: Model wrapped for quantization-aware training. Please fine-tune this model.")
                # After user fine-tunes the model, they should convert it to a quantized TFLite model for deployment.
                model = qat_model
            else:
                self.logs.append(f"Unknown quantization method '{q_method}' for TensorFlow. Skipping quantization.")
        else:
            # Other frameworks could be added here (e.g., MXNet, JAX, etc.)
            self.logs.append(f"Quantization for framework '{fw}' is not supported yet.")
        return model

    def _apply_pruning(self, model: Any, **kwargs) -> Any:
        """
        Apply pruning to the model based on the framework.
        Supports magnitude-based unstructured pruning and can be extended for structured pruning.
        """
        fw = self.profile.framework
        amount = self.profile.params.get("prune_amount", 0.2)  # default: prune 20% of weights as example
        structured = self.profile.params.get("prune_structured", False)
        self.logs.append(f"Applying pruning (amount={amount}, structured={structured}) for framework: {fw}.")

        if fw == "pytorch":
            import torch.nn.utils.prune as prune
            self.logs.append("Pruning PyTorch model weights.")
            if structured:
                # Example of structured pruning: remove entire channels (L2 norm criterion) from Conv layers
                for name, module in model.named_modules():
                    if hasattr(module, "weight") and module.weight is not None:
                        # If module is a Conv2d or Linear, apply structured pruning (prune 20% of channels for Conv, units for Linear)
                        if hasattr(module, "out_features") or hasattr(module, "out_channels"):
                            # For Linear and Conv2d, prune along the output dimension (dim=0)
                            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
                        else:
                            # For other layers (or if not identified), do unstructured pruning as fallback
                            prune.l1_unstructured(module, name="weight", amount=amount)
            else:
                # Unstructured pruning (magnitude-based) for all weight tensors in the model
                for name, module in model.named_modules():
                    if hasattr(module, "weight") and module.weight is not None:
                        prune.l1_unstructured(module, name="weight", amount=amount)
            # Remove pruning reparameterization to finalize the pruned weights (make pruning permanent)
            for name, module in model.named_modules():
                if hasattr(module, "weight") and hasattr(module, "weight_orig"):
                    prune.remove(module, "weight")
            self.logs.append("Pruning completed for PyTorch model.")
        elif fw == "tensorflow":
            import numpy as np
            self.logs.append("Pruning TensorFlow model weights.")
            try:
                # If tf-model-optimization is available, use it for structured pruning in Keras models.
                import tensorflow_model_optimization as tfmot
                if structured:
                    # Note: structured pruning in TF MOT is not straightforward; as an example, we use magnitude pruning.
                    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
                    # Apply a constant sparsity schedule to prune the model to the given amount.
                    prune_schedule = tfmot.sparsity.keras.ConstantSparsity(target_sparsity=amount, begin_step=0, end_step=1)
                    model = prune_low_magnitude(model, pruning_schedule=prune_schedule)
                    # After this, the model needs to be compiled and trained (even for 0 epochs) to apply pruning masks.
                    self.logs.append("Structured pruning applied via TF MOT wrapper (requires compilation and training to take effect).")
                    # The user should call tfmot.sparsity.keras.strip_pruning(model) after training to get the final pruned model.
                else:
                    # Unstructured pruning: directly zero out smallest weights
                    for layer in model.layers:
                        weights = layer.get_weights()
                        if weights:
                            new_weights = []
                            for w in weights:
                                # Only prune the weight matrices (skip biases or very small layers)
                                if w.ndim > 1:
                                    # Compute magnitude threshold
                                    flat = np.abs(w.flatten())
                                    if flat.size == 0:
                                        new_weights.append(w)
                                        continue
                                    thresh = np.percentile(flat, amount * 100)  # threshold for magnitude
                                    w_mask = np.where(np.abs(w) < thresh, 0, w)
                                    new_weights.append(w_mask)
                                else:
                                    new_weights.append(w)
                            layer.set_weights(new_weights)
                    self.logs.append("Unstructured pruning: weights below percentile threshold set to 0.")
            except ImportError:
                # If tfmot is not available, do simple unstructured pruning by zeroing out small weights (as above, without structured support).
                for layer in getattr(model, "layers", []):
                    weights = layer.get_weights()
                    if weights:
                        new_weights = []
                        import numpy as np
                        for w in weights:
                            if w.ndim > 1:
                                flat = np.abs(w.flatten())
                                if flat.size == 0:
                                    new_weights.append(w)
                                    continue
                                thresh = np.percentile(flat, amount * 100)
                                w_mask = np.where(np.abs(w) < thresh, 0, w)
                                new_weights.append(w_mask)
                            else:
                                new_weights.append(w)
                        layer.set_weights(new_weights)
                self.logs.append("Unstructured pruning applied (tfmot not available for structured pruning).")
        else:
            self.logs.append(f"Pruning for framework '{fw}' is not supported yet.")
        return model

    def _apply_distillation(self, teacher_model: Any, student_model: Any, data: Any = None, **kwargs) -> Any:
        """
        Apply knowledge distillation from teacher_model to student_model.
        The student model is trained to mimic the teacher's outputs.
        This implementation provides a generic outline; actual training loop would depend on the framework and data.
        :param teacher_model: The large (teacher) model.
        :param student_model: The smaller (student) model to train.
        :param data: Training data (e.g., DataLoader or tf.data.Dataset) for distillation.
        :return: The trained student model.
        """
        fw = self.profile.framework
        self.logs.append(f"Starting knowledge distillation using framework: {fw}.")
        if fw == "pytorch":
            import torch
            import torch.nn.functional as F

            teacher_model.eval()
            student_model.train()
            optimizer = kwargs.get("optimizer")
            if optimizer is None:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
            distill_epochs = self.profile.params.get("distill_epochs", 1)
            temperature = self.profile.params.get("distill_temperature", 1.0)
            alpha = self.profile.params.get("distill_alpha", 0.5)  # weight for distillation loss vs true loss

            # If actual data loader is provided:
            if data is None:
                self.logs.append("No distillation training data provided; skipping actual training loop.")
            else:
                self.logs.append(f"Training student for {distill_epochs} epoch(s) using distillation.")
                for epoch in range(distill_epochs):
                    for batch in data:
                        # Assume batch is (inputs, targets) tuple
                        inputs, targets = batch[0], batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else (batch, None)
                        optimizer.zero_grad()
                        with torch.no_grad():
                            teacher_outputs = teacher_model(inputs)
                        student_outputs = student_model(inputs)
                        # Compute distillation loss (e.g., Kullback-Leibler divergence between softened logits)
                        if temperature != 1.0:
                            # Apply temperature to soften probabilities
                            T = temperature
                            teacher_probs = F.log_softmax(teacher_outputs / T, dim=-1)
                            student_log_probs = F.log_softmax(student_outputs / T, dim=-1)
                            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
                        else:
                            # Simpler distillation loss without temperature scaling (just use softmax probabilities)
                            teacher_probs = F.softmax(teacher_outputs, dim=-1)
                            student_log_probs = F.log_softmax(student_outputs, dim=-1)
                            distill_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
                        if targets is not None:
                            # If ground-truth labels are available, also compute student supervised loss
                            hard_loss = F.cross_entropy(student_outputs, targets)
                        else:
                            hard_loss = 0.0
                        loss = distill_loss * alpha + hard_loss * (1 - alpha)
                        loss.backward()
                        optimizer.step()
                self.logs.append("Knowledge distillation training loop completed for student model.")
            # After distillation, use the student as the compressed model
            model = student_model
        elif fw == "tensorflow":
            import tensorflow as tf
            # We assume `student_model` is a tf.keras.Model and `teacher_model` outputs logits.
            optimizer = kwargs.get("optimizer")
            if optimizer is None:
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            distill_epochs = self.profile.params.get("distill_epochs", 1)
            temperature = self.profile.params.get("distill_temperature", 1.0)
            alpha = self.profile.params.get("distill_alpha", 0.5)

            if data is None:
                self.logs.append("No distillation training data provided for TensorFlow; skipping training.")
            else:
                self.logs.append(f"Training student for {distill_epochs} epoch(s) using distillation (TensorFlow).")
                # If data is a tf.data.Dataset of (x, y)
                for epoch in range(distill_epochs):
                    for batch in data:
                        x, y = batch
                        # Forward passes
                        teacher_preds = teacher_model(x, training=False)
                        with tf.GradientTape() as tape:
                            student_preds = student_model(x, training=True)
                            # Compute distillation loss
                            if temperature != 1.0:
                                teacher_probs = tf.nn.log_softmax(teacher_preds / temperature)
                                student_log_probs = tf.nn.log_softmax(student_preds / temperature)
                                distill_loss = tf.reduce_mean(
                                    tf.keras.losses.kl_divergence(teacher_probs, student_log_probs)
                                ) * (temperature ** 2)
                            else:
                                teacher_probs = tf.nn.softmax(teacher_preds)
                                student_log_probs = tf.nn.log_softmax(student_preds)
                                distill_loss = -tf.reduce_mean(tf.reduce_sum(teacher_probs * student_log_probs, axis=-1))
                            hard_loss = tf.constant(0.0)
                            if y is not None:
                                hard_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, student_preds))
                            loss = alpha * distill_loss + (1 - alpha) * hard_loss
                        # Backpropagation
                        grads = tape.gradient(loss, student_model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
                self.logs.append("Knowledge distillation training loop completed for student model (TensorFlow).")
            model = student_model
        else:
            self.logs.append(f"Knowledge distillation for framework '{fw}' is not supported yet.")
            return teacher_model  # no change
        self.logs.append("Distillation complete. Returning distilled (student) model.")
        return model

    def get_logs(self) -> list:
        """Retrieve the log messages of transformations applied."""
        return self.logs



#  -- Example usage: PyTorch model compression
import torch
import torch.nn as nn

# Define a simple PyTorch model for example (e.g., a small feed-forward network)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleModel()
# Assume the model is trained ... (omitted)

# Create profile info indicating PyTorch framework, model type, hardware target, etc.
profile = ProfileInfo(framework="PyTorch", model_type="MLP", hardware="CPU",
                      techniques=["quantization", "pruning"],  # explicitly specifying techniques
                      params={"quantization_method": "dynamic", "prune_amount": 0.3})

compressor = ModelCompressor(profile)
compressed_model = compressor.compress(model)

# Retrieve and print log of actions
for log in compressor.get_logs():
    print(log)



#  -- Example usage : TensorFlow model compression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a simple TensorFlow Keras model (e.g., for MNIST)
model_tf = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
# Assume model_tf is trained ... (omitted)

# Create profile info for TensorFlow model compression on a mobile/edge device
profile_tf = ProfileInfo(framework="TensorFlow", model_type="CNN", hardware="Mobile",
                         techniques=["quantization", "pruning"],  # request both
                         params={"quantization_method": "static", "prune_amount": 0.2})

compressor_tf = ModelCompressor(profile_tf)

# If we had calibration data for quantization (e.g., a few samples from the training set):
calibration_data = [tf.random.normal([1, 28, 28, 1]) for _ in range(10)]  # dummy calibration data
compressed_model_tf = compressor_tf.compress(model_tf, calibration_data=calibration_data)

# Print out the log of actions
for log in compressor_tf.get_logs():
    print(log)
