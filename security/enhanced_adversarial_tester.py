"""
Enhanced Adversarial Tester for Machine Learning Models

This module provides advanced tools for testing the adversarial robustness of machine learning models,
including a wider range of attack types and more comprehensive robustness metrics.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from security.adversarial_tester import AdversarialTester

logger = logging.getLogger(__name__)

class EnhancedAdversarialTester(AdversarialTester):
    """
    Enhanced tester for adversarial robustness of machine learning models.
    Extends the base AdversarialTester with more attack types and comprehensive metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced adversarial tester.

        Args:
            config: Configuration for the adversarial tester
        """
        super().__init__(config)

        # Enhanced default attack parameters
        self.enhanced_default_params = {
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
            },
            "boundary_attack": {
                "iterations": 1000,
                "max_directions": 25,
                "spherical_step": 0.01,
                "source_step": 0.01
            },
            "hop_skip_jump": {
                "max_iterations": 50,
                "max_queries": 5000,
                "norm": "l2"
            },
            "spatial_transformation": {
                "max_rotation": 30,  # degrees
                "max_translation": 0.3,  # fraction of image size
                "max_scale": 0.2  # fraction of image size
            },
            "elastic_net": {
                "confidence": 0.0,
                "learning_rate": 0.01,
                "binary_search_steps": 9,
                "max_iterations": 1000,
                "beta": 0.01
            },
            "square_attack": {
                "epsilon": 0.1,
                "max_queries": 5000,
                "norm": "linf"
            },
            "zeroth_order_optimization": {
                "epsilon": 0.1,
                "max_queries": 10000,
                "learning_rate": 0.01
            }
        }

        # Update default params with enhanced ones
        self.default_params.update(self.enhanced_default_params)

    def test_enhanced_adversarial_robustness(self, model, framework: str, test_data: Tuple[np.ndarray, np.ndarray],
                                  attack_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test the adversarial robustness of a model with enhanced attacks.

        Args:
            model: The model to test
            framework: The framework of the model (pytorch, tensorflow)
            test_data: Tuple of (inputs, labels)
            attack_types: List of attack types to test

        Returns:
            Dictionary with enhanced adversarial robustness results
        """
        if attack_types is None:
            attack_types = ["fgsm", "pgd", "deepfool", "carlini_wagner"]

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
        robustness_score = self._calculate_enhanced_robustness_score(results)
        
        # Calculate additional robustness metrics
        additional_metrics = self._calculate_additional_robustness_metrics(results)

        return {
            "attack_results": results,
            "robustness_score": robustness_score,
            "robustness_level": self._get_enhanced_robustness_level(robustness_score),
            "additional_metrics": additional_metrics
        }

    def _test_pytorch_model(self, model, inputs, labels, attack_type: str) -> Dict[str, Any]:
        """
        Test a PyTorch model with the specified attack type.

        Args:
            model: PyTorch model
            inputs: Input data
            labels: True labels
            attack_type: Type of attack to perform

        Returns:
            Dictionary with attack results
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with 'pip install torch'.")
            return {"error": "PyTorch is not installed"}

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
        elif attack_type == "carlini_wagner":
            return self._carlini_wagner_attack_pytorch(model, inputs, labels, attack_params)
        elif attack_type == "deepfool":
            return self._deepfool_attack_pytorch(model, inputs, labels, attack_params)
        elif attack_type == "boundary_attack":
            return self._boundary_attack_pytorch(model, inputs, labels, attack_params)
        elif attack_type == "spatial_transformation":
            return self._spatial_transformation_attack_pytorch(model, inputs, labels, attack_params)
        elif attack_type == "elastic_net":
            return self._elastic_net_attack_pytorch(model, inputs, labels, attack_params)
        elif attack_type == "square_attack":
            return self._square_attack_pytorch(model, inputs, labels, attack_params)
        else:
            logger.warning(f"Unsupported attack type for PyTorch: {attack_type}")
            return {"error": f"Unsupported attack type: {attack_type}"}

    def _carlini_wagner_attack_pytorch(self, model, inputs: torch.Tensor, labels: torch.Tensor,
                                    params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Carlini & Wagner attack on a PyTorch model.

        Args:
            model: PyTorch model
            inputs: Input data
            labels: True labels
            params: Attack parameters

        Returns:
            Dictionary with attack results
        """
        try:
            from foolbox import PyTorchModel
            from foolbox.attacks import L2CarliniWagnerAttack
            import torch
        except ImportError:
            logger.error("Foolbox is not installed. Please install it with 'pip install foolbox'.")
            return {"error": "Foolbox is not installed"}

        # Get parameters
        confidence = params.get("confidence", 0.0)
        learning_rate = params.get("learning_rate", 0.01)
        binary_search_steps = params.get("binary_search_steps", 9)
        max_iterations = params.get("max_iterations", 1000)

        # Create Foolbox model
        fmodel = PyTorchModel(model, bounds=(0, 1))

        # Create attack
        attack = L2CarliniWagnerAttack(
            confidence=confidence,
            learning_rate=learning_rate,
            binary_search_steps=binary_search_steps,
            steps=max_iterations
        )

        # Original accuracy
        with torch.no_grad():
            outputs = model(inputs)
            if outputs.shape[1] > 1:  # Multi-class classification
                original_preds = outputs.argmax(dim=1)
                original_accuracy = (original_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                original_preds = (outputs > 0.5).float()
                original_accuracy = (original_preds.squeeze() == labels).float().mean().item()

        # Perform attack
        adversarial_inputs, success = attack(fmodel, inputs, labels, epsilons=None)

        # Adversarial accuracy
        with torch.no_grad():
            adversarial_outputs = model(adversarial_inputs)
            if adversarial_outputs.shape[1] > 1:  # Multi-class classification
                adversarial_preds = adversarial_outputs.argmax(dim=1)
                adversarial_accuracy = (adversarial_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                adversarial_preds = (adversarial_outputs > 0.5).float()
                adversarial_accuracy = (adversarial_preds.squeeze() == labels).float().mean().item()

        # Calculate robustness
        robustness = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0

        # Calculate perturbation size
        perturbation = (adversarial_inputs - inputs).abs().mean().item()

        return {
            "attack_type": "carlini_wagner",
            "confidence": confidence,
            "original_accuracy": original_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "robustness": robustness,
            "perturbation": perturbation,
            "success_rate": success.float().mean().item()
        }

    def _deepfool_attack_pytorch(self, model, inputs: torch.Tensor, labels: torch.Tensor,
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform DeepFool attack on a PyTorch model.

        Args:
            model: PyTorch model
            inputs: Input data
            labels: True labels
            params: Attack parameters

        Returns:
            Dictionary with attack results
        """
        try:
            from foolbox import PyTorchModel
            from foolbox.attacks import DeepFoolAttack
            import torch
        except ImportError:
            logger.error("Foolbox is not installed. Please install it with 'pip install foolbox'.")
            return {"error": "Foolbox is not installed"}

        # Get parameters
        max_iterations = params.get("max_iterations", 100)
        epsilon = params.get("epsilon", 1e-6)

        # Create Foolbox model
        fmodel = PyTorchModel(model, bounds=(0, 1))

        # Create attack
        attack = DeepFoolAttack(steps=max_iterations, candidates=10)

        # Original accuracy
        with torch.no_grad():
            outputs = model(inputs)
            if outputs.shape[1] > 1:  # Multi-class classification
                original_preds = outputs.argmax(dim=1)
                original_accuracy = (original_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                original_preds = (outputs > 0.5).float()
                original_accuracy = (original_preds.squeeze() == labels).float().mean().item()

        # Perform attack
        adversarial_inputs, success = attack(fmodel, inputs, labels, epsilons=None)

        # Adversarial accuracy
        with torch.no_grad():
            adversarial_outputs = model(adversarial_inputs)
            if adversarial_outputs.shape[1] > 1:  # Multi-class classification
                adversarial_preds = adversarial_outputs.argmax(dim=1)
                adversarial_accuracy = (adversarial_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                adversarial_preds = (adversarial_outputs > 0.5).float()
                adversarial_accuracy = (adversarial_preds.squeeze() == labels).float().mean().item()

        # Calculate robustness
        robustness = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0

        # Calculate perturbation size
        perturbation = (adversarial_inputs - inputs).abs().mean().item()

        return {
            "attack_type": "deepfool",
            "max_iterations": max_iterations,
            "original_accuracy": original_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "robustness": robustness,
            "perturbation": perturbation,
            "success_rate": success.float().mean().item()
        }

    def _boundary_attack_pytorch(self, model, inputs: torch.Tensor, labels: torch.Tensor,
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Boundary attack on a PyTorch model.

        Args:
            model: PyTorch model
            inputs: Input data
            labels: True labels
            params: Attack parameters

        Returns:
            Dictionary with attack results
        """
        try:
            from foolbox import PyTorchModel
            from foolbox.attacks import BoundaryAttack
            import torch
        except ImportError:
            logger.error("Foolbox is not installed. Please install it with 'pip install foolbox'.")
            return {"error": "Foolbox is not installed"}

        # Get parameters
        iterations = params.get("iterations", 1000)
        max_directions = params.get("max_directions", 25)
        spherical_step = params.get("spherical_step", 0.01)
        source_step = params.get("source_step", 0.01)

        # Create Foolbox model
        fmodel = PyTorchModel(model, bounds=(0, 1))

        # Create attack
        attack = BoundaryAttack(
            steps=iterations,
            spherical_step=spherical_step,
            source_step=source_step,
            init_attack=None
        )

        # Original accuracy
        with torch.no_grad():
            outputs = model(inputs)
            if outputs.shape[1] > 1:  # Multi-class classification
                original_preds = outputs.argmax(dim=1)
                original_accuracy = (original_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                original_preds = (outputs > 0.5).float()
                original_accuracy = (original_preds.squeeze() == labels).float().mean().item()

        # Perform attack
        adversarial_inputs, success = attack(fmodel, inputs, labels, epsilons=None)

        # Adversarial accuracy
        with torch.no_grad():
            adversarial_outputs = model(adversarial_inputs)
            if adversarial_outputs.shape[1] > 1:  # Multi-class classification
                adversarial_preds = adversarial_outputs.argmax(dim=1)
                adversarial_accuracy = (adversarial_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                adversarial_preds = (adversarial_outputs > 0.5).float()
                adversarial_accuracy = (adversarial_preds.squeeze() == labels).float().mean().item()

        # Calculate robustness
        robustness = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0

        # Calculate perturbation size
        perturbation = (adversarial_inputs - inputs).abs().mean().item()

        return {
            "attack_type": "boundary_attack",
            "iterations": iterations,
            "original_accuracy": original_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "robustness": robustness,
            "perturbation": perturbation,
            "success_rate": success.float().mean().item()
        }

    def _spatial_transformation_attack_pytorch(self, model, inputs: torch.Tensor, labels: torch.Tensor,
                                           params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Spatial Transformation attack on a PyTorch model.

        Args:
            model: PyTorch model
            inputs: Input data
            labels: True labels
            params: Attack parameters

        Returns:
            Dictionary with attack results
        """
        try:
            from foolbox import PyTorchModel
            from foolbox.attacks import SpatialAttack
            import torch
        except ImportError:
            logger.error("Foolbox is not installed. Please install it with 'pip install foolbox'.")
            return {"error": "Foolbox is not installed"}

        # Get parameters
        max_rotation = params.get("max_rotation", 30)
        max_translation = params.get("max_translation", 0.3)
        max_scale = params.get("max_scale", 0.2)

        # Create Foolbox model
        fmodel = PyTorchModel(model, bounds=(0, 1))

        # Create attack
        attack = SpatialAttack(
            max_translation=max_translation,
            max_rotation=max_rotation,
            steps=20
        )

        # Original accuracy
        with torch.no_grad():
            outputs = model(inputs)
            if outputs.shape[1] > 1:  # Multi-class classification
                original_preds = outputs.argmax(dim=1)
                original_accuracy = (original_preds == labels).float().mean().item()
            else:  # Binary classification or regression
                original_preds = (outputs > 0.5).float()
                original_accuracy = (original_preds.squeeze() == labels).float().mean().item()

        # Perform attack
        adversarial_inputs, success = attack(fmodel, inputs, labels, epsilons=None)

        # Adversarial accuracy
        with torch.no_grad():
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
            "attack_type": "spatial_transformation",
            "max_rotation": max_rotation,
            "max_translation": max_translation,
            "original_accuracy": original_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "robustness": robustness,
            "success_rate": success.float().mean().item()
        }

    def _calculate_enhanced_robustness_score(self, results: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate an enhanced overall robustness score based on multiple attack results.

        Args:
            results: Dictionary with attack results

        Returns:
            Enhanced robustness score (0-100)
        """
        if not results:
            return 0.0

        # Weight different attacks based on their severity
        attack_weights = {
            "fgsm": 0.1,  # Easiest attack, lowest weight
            "pgd": 0.2,
            "deepfool": 0.2,
            "carlini_wagner": 0.25,  # Strongest attack, highest weight
            "boundary_attack": 0.15,
            "spatial_transformation": 0.1
        }

        # Normalize weights for the attacks that are actually present
        total_weight = sum(attack_weights.get(attack, 0.0) for attack in results.keys())
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero

        # Calculate weighted average of robustness scores
        weighted_score = 0.0
        for attack_type, attack_result in results.items():
            if "robustness" in attack_result:
                weight = attack_weights.get(attack_type, 0.1) / total_weight
                weighted_score += attack_result["robustness"] * weight

        # Convert to 0-100 scale
        return weighted_score * 100

    def _get_enhanced_robustness_level(self, robustness_score: float) -> str:
        """
        Get a qualitative robustness level based on the robustness score.

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

    def _calculate_additional_robustness_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate additional robustness metrics based on attack results.

        Args:
            results: Dictionary with attack results

        Returns:
            Dictionary with additional robustness metrics
        """
        if not results:
            return {}

        # Extract perturbation sizes
        perturbations = {attack: result.get("perturbation", 0.0) for attack, result in results.items() if "perturbation" in result}

        # Extract success rates
        success_rates = {attack: result.get("success_rate", 0.0) for attack, result in results.items() if "success_rate" in result}

        # Calculate average perturbation size
        avg_perturbation = sum(perturbations.values()) / len(perturbations) if perturbations else 0.0

        # Calculate average success rate
        avg_success_rate = sum(success_rates.values()) / len(success_rates) if success_rates else 0.0

        # Calculate worst-case robustness (minimum robustness across all attacks)
        worst_case_robustness = min(
            [result.get("robustness", 1.0) for result in results.values() if "robustness" in result],
            default=1.0
        )

        # Calculate robustness consistency (standard deviation of robustness across attacks)
        robustness_values = [result.get("robustness", 0.0) for result in results.values() if "robustness" in result]
        robustness_std = np.std(robustness_values) if robustness_values else 0.0

        return {
            "average_perturbation": avg_perturbation,
            "average_success_rate": avg_success_rate,
            "worst_case_robustness": worst_case_robustness,
            "robustness_consistency": 1.0 - robustness_std,  # Higher is better
            "attack_diversity": len(results),
            "perturbations": perturbations,
            "success_rates": success_rates
        }

    def generate_enhanced_robustness_report(self, model, framework: str, test_data: Tuple[np.ndarray, np.ndarray],
                                       attack_types: Optional[List[str]] = None, output_dir: str = "results") -> str:
        """
        Generate a comprehensive robustness report for a model.

        Args:
            model: The model to test
            framework: The framework of the model (pytorch, tensorflow)
            test_data: Tuple of (inputs, labels)
            attack_types: List of attack types to test
            output_dir: Directory to save the report

        Returns:
            Path to the generated report
        """
        # Test adversarial robustness
        robustness_results = self.test_enhanced_adversarial_robustness(model, framework, test_data, attack_types)

        # Create report
        report = {
            "framework": framework,
            "robustness_score": robustness_results["robustness_score"],
            "robustness_level": robustness_results["robustness_level"],
            "attack_results": robustness_results["attack_results"],
            "additional_metrics": robustness_results["additional_metrics"],
            "recommendations": self._generate_enhanced_recommendations(robustness_results)
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save report
        report_path = os.path.join(output_dir, "enhanced_robustness_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report_path

    def _generate_enhanced_recommendations(self, robustness_results: Dict[str, Any]) -> List[str]:
        """
        Generate enhanced recommendations based on robustness results.

        Args:
            robustness_results: Dictionary with robustness results

        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Basic recommendations
        recommendations.append("Implement adversarial training to improve model robustness")
        recommendations.append("Use ensemble methods to improve robustness")
        
        # Score-based recommendations
        robustness_score = robustness_results.get("robustness_score", 0)
        if robustness_score < 30:
            recommendations.append("Consider redesigning the model architecture for better robustness")
            recommendations.append("Apply defensive distillation to improve robustness")
            recommendations.append("Implement input preprocessing defenses")
        elif robustness_score < 60:
            recommendations.append("Add regularization to improve generalization and robustness")
            recommendations.append("Consider feature squeezing as a defense mechanism")
            recommendations.append("Implement gradient masking techniques")
        
        # Attack-specific recommendations
        attack_results = robustness_results.get("attack_results", {})
        for attack_type, result in attack_results.items():
            if attack_type == "fgsm" and result.get("robustness", 1.0) < 0.5:
                recommendations.append("Implement gradient masking to defend against gradient-based attacks like FGSM")
            
            if attack_type == "pgd" and result.get("robustness", 1.0) < 0.4:
                recommendations.append("Use PGD adversarial training to specifically defend against PGD attacks")
            
            if attack_type == "carlini_wagner" and result.get("robustness", 1.0) < 0.3:
                recommendations.append("Implement defensive distillation to counter optimization-based attacks like C&W")
            
            if attack_type == "spatial_transformation" and result.get("robustness", 1.0) < 0.6:
                recommendations.append("Train with spatial transformations to improve robustness against geometric attacks")
        
        # Additional metrics-based recommendations
        additional_metrics = robustness_results.get("additional_metrics", {})
        
        if additional_metrics.get("worst_case_robustness", 1.0) < 0.2:
            recommendations.append("Focus on improving worst-case robustness through targeted adversarial training")
        
        if additional_metrics.get("robustness_consistency", 1.0) < 0.5:
            recommendations.append("Improve consistency of robustness across different attack types")
        
        if additional_metrics.get("average_success_rate", 0.0) > 0.7:
            recommendations.append("Implement detection mechanisms for adversarial examples")
        
        return recommendations
