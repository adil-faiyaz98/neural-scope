"""
Adversarial Tester for Machine Learning Models

This module provides tools for testing the adversarial robustness of machine learning models.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import torch conditionally to avoid errors if not installed
try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

class AdversarialTester:
    """
    Tester for adversarial robustness of machine learning models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adversarial tester.

        Args:
            config: Configuration for the adversarial tester
        """
        self.config = config or {}

        # Default attack parameters
        self.default_params = {
            "fgsm": {
                "epsilon": 0.1,
                "norm": "inf"
            },
            "pgd": {
                "epsilon": 0.1,
                "alpha": 0.01,
                "iterations": 40,
                "norm": "inf"
            },
            "carlini_wagner": {
                "confidence": 0.0,
                "learning_rate": 0.01,
                "binary_search_steps": 9,
                "max_iterations": 1000
            },
            "deepfool": {
                "max_iterations": 100,
                "epsilon": 1e-6
            }
        }

    def test_adversarial_robustness(self, model, framework: str, test_data: Tuple[np.ndarray, np.ndarray],
                                  attack_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test the adversarial robustness of a model.

        Args:
            model: The model to test
            framework: The framework of the model (pytorch, tensorflow)
            test_data: Tuple of (inputs, labels)
            attack_types: List of attack types to test

        Returns:
            Dictionary with adversarial robustness results
        """
        if attack_types is None:
            attack_types = ["fgsm", "pgd"]

        results = {}

        # Get test data
        inputs, labels = test_data

        # Test each attack type
        for attack_type in attack_types:
            logger.info(f"Testing adversarial robustness with {attack_type} attack...")

            if framework == "pytorch":
                attack_results = self._test_pytorch_model(model, inputs, labels, attack_type)
            elif framework == "tensorflow":
                attack_results = self._test_tensorflow_model(model, inputs, labels, attack_type)
            else:
                logger.error(f"Unsupported framework: {framework}")
                continue

            results[attack_type] = attack_results

        # Calculate overall robustness score
        robustness_score = self._calculate_robustness_score(results)

        return {
            "attack_results": results,
            "robustness_score": robustness_score,
            "robustness_level": self._get_robustness_level(robustness_score)
        }

    def _test_pytorch_model(self, model, inputs: np.ndarray, labels: np.ndarray, attack_type: str) -> Dict[str, Any]:
        """
        Test a PyTorch model with adversarial attacks.

        Args:
            model: PyTorch model
            inputs: Input data
            labels: True labels
            attack_type: Type of attack to use

        Returns:
            Dictionary with attack results
        """
        try:
            import torch
            from torch import nn
            import torch.nn.functional as F
        except ImportError:
            logger.error("PyTorch is required for testing PyTorch models")
            return {"error": "PyTorch is required for testing PyTorch models"}

        # Convert inputs and labels to PyTorch tensors if they're not already
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        # Ensure model is in evaluation mode
        model.eval()

        # Get attack parameters
        attack_params = self.config.get(attack_type, self.default_params.get(attack_type, {}))

        # Implement attacks
        if attack_type == "fgsm":
            return self._fgsm_attack_pytorch(model, inputs, labels, attack_params)
        elif attack_type == "pgd":
            return self._pgd_attack_pytorch(model, inputs, labels, attack_params)
        else:
            logger.warning(f"Unsupported attack type for PyTorch: {attack_type}")
            return {"error": f"Unsupported attack type: {attack_type}"}

    def _test_tensorflow_model(self, model, inputs: np.ndarray, labels: np.ndarray, attack_type: str) -> Dict[str, Any]:
        """
        Test a TensorFlow model with adversarial attacks.

        Args:
            model: TensorFlow model
            inputs: Input data
            labels: True labels
            attack_type: Type of attack to use

        Returns:
            Dictionary with attack results
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow is required for testing TensorFlow models")
            return {"error": "TensorFlow is required for testing TensorFlow models"}

        # Convert inputs and labels to TensorFlow tensors if they're not already
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels, dtype=tf.int64)

        # Get attack parameters
        attack_params = self.config.get(attack_type, self.default_params.get(attack_type, {}))

        # Implement attacks
        if attack_type == "fgsm":
            return self._fgsm_attack_tensorflow(model, inputs, labels, attack_params)
        elif attack_type == "pgd":
            return self._pgd_attack_tensorflow(model, inputs, labels, attack_params)
        else:
            logger.warning(f"Unsupported attack type for TensorFlow: {attack_type}")
            return {"error": f"Unsupported attack type: {attack_type}"}

    def _fgsm_attack_pytorch(self, model, inputs: torch.Tensor, labels: torch.Tensor,
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform FGSM attack on a PyTorch model.

        Args:
            model: PyTorch model
            inputs: Input data
            labels: True labels
            params: Attack parameters

        Returns:
            Dictionary with attack results
        """
        import torch
        import torch.nn.functional as F

        # Get parameters
        epsilon = params.get("epsilon", 0.1)

        # Create a copy of the inputs that requires gradient
        perturbed_inputs = inputs.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = model(perturbed_inputs)

        # Calculate loss
        if outputs.shape[1] > 1:  # Multi-class classification
            loss = F.cross_entropy(outputs, labels)
        else:  # Binary classification or regression
            loss = F.mse_loss(outputs, labels.float().unsqueeze(1))

        # Backward pass
        loss.backward()

        # Create adversarial examples
        with torch.no_grad():
            # Get the sign of the gradient
            gradient_sign = perturbed_inputs.grad.sign()

            # Create adversarial examples
            adversarial_inputs = perturbed_inputs + epsilon * gradient_sign

            # Clamp to valid range [0, 1]
            adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)

        # Evaluate on adversarial examples
        model.eval()
        with torch.no_grad():
            # Original accuracy
            original_outputs = model(inputs)
            if original_outputs.shape[1] > 1:  # Multi-class classification
                original_preds = original_outputs.argmax(dim=1)
                original_accuracy = (original_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                original_preds = (original_outputs > 0.5).float()
                original_accuracy = (original_preds.squeeze() == labels).float().mean().item()

            # Adversarial accuracy
            adversarial_outputs = model(adversarial_inputs)
            if adversarial_outputs.shape[1] > 1:  # Multi-class classification
                adversarial_preds = adversarial_outputs.argmax(dim=1)
                adversarial_accuracy = (adversarial_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                adversarial_preds = (adversarial_outputs > 0.5).float()
                adversarial_accuracy = (adversarial_preds.squeeze() == labels).float().mean().item()

        # Calculate robustness
        robustness = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0

        return {
            "attack_type": "fgsm",
            "epsilon": epsilon,
            "original_accuracy": original_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "robustness": robustness
        }

    def _pgd_attack_pytorch(self, model, inputs: torch.Tensor, labels: torch.Tensor,
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform PGD attack on a PyTorch model.

        Args:
            model: PyTorch model
            inputs: Input data
            labels: True labels
            params: Attack parameters

        Returns:
            Dictionary with attack results
        """
        import torch
        import torch.nn.functional as F

        # Get parameters
        epsilon = params.get("epsilon", 0.1)
        alpha = params.get("alpha", 0.01)
        iterations = params.get("iterations", 40)

        # Create a copy of the inputs
        original_inputs = inputs.clone().detach()

        # Add small random noise to start
        perturbed_inputs = original_inputs + torch.empty_like(original_inputs).uniform_(-epsilon, epsilon)
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)

        # PGD attack
        for _ in range(iterations):
            perturbed_inputs.requires_grad = True

            # Forward pass
            outputs = model(perturbed_inputs)

            # Calculate loss
            if outputs.shape[1] > 1:  # Multi-class classification
                loss = F.cross_entropy(outputs, labels)
            else:  # Binary classification or regression
                loss = F.mse_loss(outputs, labels.float().unsqueeze(1))

            # Backward pass
            loss.backward()

            # Update adversarial examples
            with torch.no_grad():
                # Get the sign of the gradient
                gradient_sign = perturbed_inputs.grad.sign()

                # Update adversarial examples
                perturbed_inputs = perturbed_inputs + alpha * gradient_sign

                # Project back to epsilon ball
                delta = torch.clamp(perturbed_inputs - original_inputs, -epsilon, epsilon)
                perturbed_inputs = original_inputs + delta

                # Clamp to valid range [0, 1]
                perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)

        # Evaluate on adversarial examples
        model.eval()
        with torch.no_grad():
            # Original accuracy
            original_outputs = model(inputs)
            if original_outputs.shape[1] > 1:  # Multi-class classification
                original_preds = original_outputs.argmax(dim=1)
                original_accuracy = (original_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                original_preds = (original_outputs > 0.5).float()
                original_accuracy = (original_preds.squeeze() == labels).float().mean().item()

            # Adversarial accuracy
            adversarial_outputs = model(perturbed_inputs)
            if adversarial_outputs.shape[1] > 1:  # Multi-class classification
                adversarial_preds = adversarial_outputs.argmax(dim=1)
                adversarial_accuracy = (adversarial_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                adversarial_preds = (adversarial_outputs > 0.5).float()
                adversarial_accuracy = (adversarial_preds.squeeze() == labels).float().mean().item()

        # Calculate robustness
        robustness = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0

        return {
            "attack_type": "pgd",
            "epsilon": epsilon,
            "alpha": alpha,
            "iterations": iterations,
            "original_accuracy": original_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "robustness": robustness
        }

    def _fgsm_attack_tensorflow(self, model, inputs: tf.Tensor, labels: tf.Tensor,
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform FGSM attack on a TensorFlow model.

        Args:
            model: TensorFlow model
            inputs: Input data
            labels: True labels
            params: Attack parameters

        Returns:
            Dictionary with attack results
        """
        import tensorflow as tf

        # Get parameters
        epsilon = params.get("epsilon", 0.1)

        # Create a copy of the inputs
        perturbed_inputs = tf.identity(inputs)

        # FGSM attack
        with tf.GradientTape() as tape:
            tape.watch(perturbed_inputs)

            # Forward pass
            outputs = model(perturbed_inputs)

            # Calculate loss
            if outputs.shape[1] > 1:  # Multi-class classification
                loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, outputs)
            else:  # Binary classification or regression
                loss = tf.keras.losses.MeanSquaredError()(labels, outputs)

        # Get the gradient
        gradient = tape.gradient(loss, perturbed_inputs)

        # Create adversarial examples
        gradient_sign = tf.sign(gradient)
        adversarial_inputs = perturbed_inputs + epsilon * gradient_sign

        # Clamp to valid range [0, 1]
        adversarial_inputs = tf.clip_by_value(adversarial_inputs, 0, 1)

        # Evaluate on adversarial examples
        # Original accuracy
        original_outputs = model(inputs)
        if original_outputs.shape[1] > 1:  # Multi-class classification
            original_preds = tf.argmax(original_outputs, axis=1)
            original_accuracy = tf.reduce_mean(tf.cast(tf.equal(original_preds, labels), tf.float32)).numpy()
        else:  # Binary classification or regression
            original_preds = tf.cast(original_outputs > 0.5, tf.float32)
            original_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(original_preds), labels), tf.float32)).numpy()

        # Adversarial accuracy
        adversarial_outputs = model(adversarial_inputs)
        if adversarial_outputs.shape[1] > 1:  # Multi-class classification
            adversarial_preds = tf.argmax(adversarial_outputs, axis=1)
            adversarial_accuracy = tf.reduce_mean(tf.cast(tf.equal(adversarial_preds, labels), tf.float32)).numpy()
        else:  # Binary classification or regression
            adversarial_preds = tf.cast(adversarial_outputs > 0.5, tf.float32)
            adversarial_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(adversarial_preds), labels), tf.float32)).numpy()

        # Calculate robustness
        robustness = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0

        return {
            "attack_type": "fgsm",
            "epsilon": epsilon,
            "original_accuracy": float(original_accuracy),
            "adversarial_accuracy": float(adversarial_accuracy),
            "robustness": float(robustness)
        }

    def _pgd_attack_tensorflow(self, model, inputs: tf.Tensor, labels: tf.Tensor,
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform PGD attack on a TensorFlow model.

        Args:
            model: TensorFlow model
            inputs: Input data
            labels: True labels
            params: Attack parameters

        Returns:
            Dictionary with attack results
        """
        import tensorflow as tf

        # Get parameters
        epsilon = params.get("epsilon", 0.1)
        alpha = params.get("alpha", 0.01)
        iterations = params.get("iterations", 40)

        # Create a copy of the inputs
        original_inputs = tf.identity(inputs)

        # Add small random noise to start
        perturbed_inputs = original_inputs + tf.random.uniform(original_inputs.shape, -epsilon, epsilon)
        perturbed_inputs = tf.clip_by_value(perturbed_inputs, 0, 1)

        # PGD attack
        for _ in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(perturbed_inputs)

                # Forward pass
                outputs = model(perturbed_inputs)

                # Calculate loss
                if outputs.shape[1] > 1:  # Multi-class classification
                    loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, outputs)
                else:  # Binary classification or regression
                    loss = tf.keras.losses.MeanSquaredError()(labels, outputs)

            # Get the gradient
            gradient = tape.gradient(loss, perturbed_inputs)

            # Update adversarial examples
            gradient_sign = tf.sign(gradient)
            perturbed_inputs = perturbed_inputs + alpha * gradient_sign

            # Project back to epsilon ball
            delta = tf.clip_by_value(perturbed_inputs - original_inputs, -epsilon, epsilon)
            perturbed_inputs = original_inputs + delta

            # Clamp to valid range [0, 1]
            perturbed_inputs = tf.clip_by_value(perturbed_inputs, 0, 1)

        # Evaluate on adversarial examples
        # Original accuracy
        original_outputs = model(inputs)
        if original_outputs.shape[1] > 1:  # Multi-class classification
            original_preds = tf.argmax(original_outputs, axis=1)
            original_accuracy = tf.reduce_mean(tf.cast(tf.equal(original_preds, labels), tf.float32)).numpy()
        else:  # Binary classification or regression
            original_preds = tf.cast(original_outputs > 0.5, tf.float32)
            original_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(original_preds), labels), tf.float32)).numpy()

        # Adversarial accuracy
        adversarial_outputs = model(perturbed_inputs)
        if adversarial_outputs.shape[1] > 1:  # Multi-class classification
            adversarial_preds = tf.argmax(adversarial_outputs, axis=1)
            adversarial_accuracy = tf.reduce_mean(tf.cast(tf.equal(adversarial_preds, labels), tf.float32)).numpy()
        else:  # Binary classification or regression
            adversarial_preds = tf.cast(adversarial_outputs > 0.5, tf.float32)
            adversarial_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(adversarial_preds), labels), tf.float32)).numpy()

        # Calculate robustness
        robustness = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0

        return {
            "attack_type": "pgd",
            "epsilon": epsilon,
            "alpha": alpha,
            "iterations": iterations,
            "original_accuracy": float(original_accuracy),
            "adversarial_accuracy": float(adversarial_accuracy),
            "robustness": float(robustness)
        }

    def _calculate_robustness_score(self, results: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate an overall robustness score.

        Args:
            results: Dictionary with attack results

        Returns:
            Robustness score (0-100)
        """
        if not results:
            return 0

        # Calculate average robustness across all attacks
        robustness_values = [result.get("robustness", 0) for result in results.values()]
        avg_robustness = sum(robustness_values) / len(robustness_values)

        # Convert to a 0-100 scale
        return avg_robustness * 100

    def _get_robustness_level(self, robustness_score: float) -> str:
        """
        Get a robustness level based on the robustness score.

        Args:
            robustness_score: Robustness score (0-100)

        Returns:
            Robustness level (very low, low, medium, high, very high)
        """
        if robustness_score < 20:
            return "very low"
        elif robustness_score < 40:
            return "low"
        elif robustness_score < 60:
            return "medium"
        elif robustness_score < 80:
            return "high"
        else:
            return "very high"

    def generate_robustness_report(self, model, framework: str, test_data: Tuple[np.ndarray, np.ndarray],
                                 output_dir: str, attack_types: Optional[List[str]] = None) -> str:
        """
        Generate a robustness report for a model.

        Args:
            model: The model to test
            framework: The framework of the model (pytorch, tensorflow)
            test_data: Tuple of (inputs, labels)
            output_dir: Directory to save the report
            attack_types: List of attack types to test

        Returns:
            Path to the generated report
        """
        # Test adversarial robustness
        robustness_results = self.test_adversarial_robustness(model, framework, test_data, attack_types)

        # Create report
        report = {
            "framework": framework,
            "robustness_score": robustness_results["robustness_score"],
            "robustness_level": robustness_results["robustness_level"],
            "attack_results": robustness_results["attack_results"],
            "recommendations": self._generate_recommendations(robustness_results)
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save report
        report_path = os.path.join(output_dir, "robustness_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Robustness report saved to {report_path}")

        return report_path

    def _generate_recommendations(self, robustness_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on robustness results.

        Args:
            robustness_results: Dictionary with robustness results

        Returns:
            List of recommendations
        """
        recommendations = []

        robustness_score = robustness_results["robustness_score"]

        if robustness_score < 50:
            recommendations.append("Implement adversarial training to improve model robustness")
            recommendations.append("Consider using ensemble methods to improve robustness")
            recommendations.append("Add input validation to detect and reject adversarial examples")

        if robustness_score < 30:
            recommendations.append("Consider using a more robust architecture")
            recommendations.append("Implement defensive distillation to improve robustness")

        if "fgsm" in robustness_results["attack_results"]:
            fgsm_result = robustness_results["attack_results"]["fgsm"]
            if fgsm_result.get("robustness", 0) < 0.5:
                recommendations.append("Model is particularly vulnerable to FGSM attacks. Consider gradient masking techniques")

        if "pgd" in robustness_results["attack_results"]:
            pgd_result = robustness_results["attack_results"]["pgd"]
            if pgd_result.get("robustness", 0) < 0.3:
                recommendations.append("Model is highly vulnerable to PGD attacks. Consider implementing certified defenses")

        # Add general recommendations
        recommendations.extend([
            "Regularly test model robustness as part of the CI/CD pipeline",
            "Monitor for adversarial attacks in production",
            "Keep up-to-date with the latest adversarial defense techniques"
        ])

        return recommendations
