Quickstart
==========

This guide will help you get started with Neural-Scope quickly.

Analyzing a Model
----------------

.. code-block:: python

    from neural_scope import NeuralScope
    import torch

    # Load a model (PyTorch example)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    # Create Neural-Scope instance
    neural_scope = NeuralScope()

    # Analyze the model
    results = neural_scope.analyze_model(
        model=model,
        model_name="resnet18",
        framework="pytorch"
    )

    # Print results
    print(f"Parameters: {results['parameters']}")
    print(f"Layers: {results['layers']}")
    print(f"Memory usage: {results['memory_usage_mb']} MB")
    print(f"Inference time: {results['inference_time_ms']} ms")

Optimizing a Model
-----------------

.. code-block:: python

    from neural_scope import NeuralScope
    import torch

    # Load a model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    # Create Neural-Scope instance
    neural_scope = NeuralScope()

    # Optimize the model
    optimized_model, results = neural_scope.optimize_model(
        model=model,
        model_name="resnet18",
        framework="pytorch",
        techniques=["quantization", "pruning"]
    )

    # Print results
    print(f"Original size: {results['original_size']} MB")
    print(f"Optimized size: {results['optimized_size']} MB")
    print(f"Size reduction: {results['size_reduction_percentage']}%")
    print(f"Inference speedup: {results['inference_speedup']}x")

Security Analysis
---------------

.. code-block:: python

    from neural_scope import NeuralScope
    import torch

    # Load a model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    # Create Neural-Scope instance
    neural_scope = NeuralScope()

    # Analyze security
    security_results = neural_scope.analyze_security(
        model=model,
        model_name="resnet18",
        framework="pytorch"
    )

    # Print results
    print(f"Security score: {security_results['security_score']}/100")
    print(f"Vulnerabilities: {security_results['total_vulnerabilities']}")
    for severity in ['critical', 'high', 'medium', 'low']:
        vulns = security_results['vulnerabilities'][severity]
        if vulns:
            print(f"{severity.capitalize()} severity: {len(vulns)}")

MLflow Integration
----------------

.. code-block:: python

    from neural_scope import NeuralScope
    import torch

    # Load a model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    # Create Neural-Scope instance with MLflow tracking
    neural_scope = NeuralScope(
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_experiment_name="model-analysis"
    )

    # Analyze the model
    results = neural_scope.analyze_model(
        model=model,
        model_name="resnet18",
        framework="pytorch"
    )

    # Results are automatically tracked in MLflow
    print(f"Results tracked in MLflow run: {neural_scope.mlflow_run_id}")

Command Line Interface
--------------------

Neural-Scope provides a command-line interface for easy use:

.. code-block:: bash

    # Analyze a model
    neural-scope analyze \
        --model-path models/model.pt \
        --framework pytorch \
        --output-dir results

    # Optimize a model
    neural-scope optimize \
        --model-path models/model.pt \
        --framework pytorch \
        --output-dir results \
        --techniques quantization,pruning

    # Test security
    neural-scope security \
        --model-path models/model.pt \
        --framework pytorch \
        --output-dir results

    # Test robustness
    neural-scope robustness \
        --model-path models/model.pt \
        --framework pytorch \
        --output-dir results \
        --attack-types fgsm,pgd
