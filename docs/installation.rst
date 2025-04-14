Installation
============

Neural-Scope can be installed using pip:

.. code-block:: bash

    # Basic installation
    pip install neural-scope

With specific dependencies:

.. code-block:: bash

    # With PyTorch support
    pip install neural-scope[pytorch]

    # With TensorFlow support
    pip install neural-scope[tensorflow]

    # With MLflow integration
    pip install neural-scope[mlflow]

    # With security and robustness testing
    pip install neural-scope[security]

    # With all dependencies
    pip install neural-scope[all]

Requirements
-----------

Neural-Scope requires Python 3.7 or later. The basic installation includes the following dependencies:

- numpy>=1.19.0
- pandas>=1.1.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
- scikit-learn>=0.23.0
- tqdm>=4.50.0
- pyyaml>=5.3.0
- requests>=2.25.0
- pillow>=8.0.0
- jsonschema>=3.2.0

Optional Dependencies
-------------------

Depending on your use case, you may want to install additional dependencies:

PyTorch
~~~~~~~

.. code-block:: bash

    pip install torch>=1.7.0 torchvision>=0.8.0

TensorFlow
~~~~~~~~~~

.. code-block:: bash

    pip install tensorflow>=2.4.0 tensorflow-hub>=0.10.0

Hugging Face
~~~~~~~~~~~

.. code-block:: bash

    pip install transformers>=4.5.0

MLflow
~~~~~~

.. code-block:: bash

    pip install mlflow>=1.15.0

ONNX
~~~~

.. code-block:: bash

    pip install onnx>=1.8.0 onnxruntime>=1.7.0

AWS
~~~

.. code-block:: bash

    pip install boto3>=1.17.0 sagemaker>=2.35.0

Security
~~~~~~~~

.. code-block:: bash

    pip install foolbox>=3.3.0 adversarial-robustness-toolbox>=1.9.0

Development Installation
----------------------

For development, you can install Neural-Scope from source:

.. code-block:: bash

    git clone https://github.com/adil-faiyaz98/neural-scope.git
    cd neural-scope
    pip install -e .
