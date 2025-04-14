Model Analysis
==============

Neural-Scope provides comprehensive model analysis capabilities to help you understand your models better.

Basic Analysis
------------

The basic analysis provides information about the model's architecture, parameters, memory usage, and inference time.

.. code-block:: python

    from neural_scope import NeuralScope
    import torch

    # Load a model
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

Analysis Results
--------------

The analysis results include:

- **Parameters**: The total number of trainable parameters in the model
- **Layers**: The total number of layers in the model
- **Architecture**: The architecture of the model
- **Memory Usage**: The amount of memory required to store the model weights
- **Inference Time**: The average time taken to perform a forward pass on a single input

Analyzing Pre-trained Models
--------------------------

Neural-Scope supports analyzing pre-trained models from various sources:

PyTorch Hub
~~~~~~~~~~

.. code-block:: python

    from neural_scope import NeuralScope
    from model_sources.pytorch_hub import PyTorchHubSource

    # Create source
    source = PyTorchHubSource()

    # Fetch model
    model_path = source.fetch_model("resnet18", "models")

    # Load model
    import torch
    model = torch.load(model_path)

    # Analyze model
    neural_scope = NeuralScope()
    results = neural_scope.analyze_model(
        model=model,
        model_name="resnet18",
        framework="pytorch"
    )

TensorFlow Hub
~~~~~~~~~~~~

.. code-block:: python

    from neural_scope import NeuralScope
    from model_sources.tensorflow_hub import TensorFlowHubSource

    # Create source
    source = TensorFlowHubSource()

    # Fetch model
    model_path = source.fetch_model("efficientnet_b0", "models")

    # Load model
    import tensorflow as tf
    model = tf.saved_model.load(model_path)

    # Analyze model
    neural_scope = NeuralScope()
    results = neural_scope.analyze_model(
        model=model,
        model_name="efficientnet_b0",
        framework="tensorflow"
    )

Hugging Face
~~~~~~~~~~

.. code-block:: python

    from neural_scope import NeuralScope
    from model_sources.huggingface import HuggingFaceSource

    # Create source
    source = HuggingFaceSource()

    # Fetch model
    model_path = source.fetch_model("bert-base-uncased", "models")

    # Load model
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path)

    # Analyze model
    neural_scope = NeuralScope()
    results = neural_scope.analyze_model(
        model=model,
        model_name="bert-base-uncased",
        framework="pytorch"
    )

Generating Reports
----------------

Neural-Scope can generate comprehensive reports in JSON and HTML formats:

.. code-block:: python

    from neural_scope import NeuralScope
    import torch

    # Load a model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    # Create Neural-Scope instance
    neural_scope = NeuralScope()

    # Analyze the model and generate reports
    results = neural_scope.analyze_model(
        model=model,
        model_name="resnet18",
        framework="pytorch",
        output_dir="results",
        generate_reports=True
    )

    # Print report paths
    print(f"JSON report: {results['report_path']}")
    print(f"HTML report: {results['html_report_path']}")

Command Line Interface
--------------------

Neural-Scope provides a command-line interface for model analysis:

.. code-block:: bash

    neural-scope analyze \
        --model-path models/model.pt \
        --framework pytorch \
        --output-dir results \
        --generate-reports
