# Neural-Scope Metrics Explained

This document explains the metrics used in Neural-Scope for model analysis, optimization, security, and robustness testing.

## Model Analysis Metrics

### Parameters
**Definition**: The total number of trainable parameters in the model.  
**Interpretation**: Higher values indicate more complex models that may require more memory and computation. Modern deep learning models can range from millions to billions of parameters.  
**Ideal Range**: Depends on the task, but generally:
- Small models: < 10M parameters
- Medium models: 10M - 100M parameters
- Large models: > 100M parameters

### Layers
**Definition**: The total number of layers in the model.  
**Interpretation**: More layers generally indicate deeper models, which can learn more complex features but may be harder to train and more prone to overfitting.  
**Ideal Range**: Depends on the architecture and task.

### Memory Usage (MB)
**Definition**: The amount of memory required to store the model weights.  
**Interpretation**: Lower values are better for deployment on resource-constrained devices.  
**Ideal Range**: Depends on deployment target:
- Mobile: < 50 MB
- Edge devices: < 200 MB
- Server: < 1 GB

### Inference Time (ms)
**Definition**: The average time taken to perform a forward pass on a single input.  
**Interpretation**: Lower values indicate faster models, which is crucial for real-time applications.  
**Ideal Range**: Depends on the application:
- Real-time applications: < 20 ms
- Interactive applications: < 100 ms
- Batch processing: < 1000 ms

## Optimization Metrics

### Original Size (MB)
**Definition**: The size of the original model in megabytes.  
**Interpretation**: Baseline for measuring optimization effectiveness.

### Optimized Size (MB)
**Definition**: The size of the optimized model in megabytes.  
**Interpretation**: Lower values indicate more effective optimization.

### Size Reduction (%)
**Definition**: The percentage reduction in model size after optimization.  
**Interpretation**: Higher values indicate more effective optimization.  
**Ideal Range**: 
- Good: > 30%
- Excellent: > 70%

### Inference Speedup (x)
**Definition**: The factor by which inference speed improves after optimization.  
**Interpretation**: Higher values indicate faster optimized models.  
**Ideal Range**:
- Good: > 1.2x
- Excellent: > 2x

## Security Metrics

### Vulnerabilities
**Definition**: Potential security issues in the model.  
**Interpretation**: Fewer vulnerabilities indicate a more secure model.  
**Severity Levels**:
- **Critical**: Immediate action required
- **High**: Action required soon
- **Medium**: Should be addressed
- **Low**: Consider addressing

### Security Score
**Definition**: Overall security rating from 0-100.  
**Interpretation**: Higher values indicate more secure models.  
**Ideal Range**:
- Poor: < 50
- Good: 50-80
- Excellent: > 80

## Adversarial Robustness Metrics

### Robustness Score
**Definition**: Overall robustness rating from 0-100.  
**Interpretation**: Higher values indicate models more resistant to adversarial attacks.  
**Ideal Range**:
- Poor: < 30
- Fair: 30-60
- Good: 60-80
- Excellent: > 80

### Robustness Level
**Definition**: Qualitative assessment of robustness.  
**Interpretation**: Better levels indicate more robust models.  
**Levels**: Very Low, Low, Medium, High, Very High

### Original Accuracy
**Definition**: Model accuracy on clean (non-adversarial) examples.  
**Interpretation**: Higher values indicate better performance on normal inputs.

### Adversarial Accuracy
**Definition**: Model accuracy on adversarial examples.  
**Interpretation**: Higher values indicate better resistance to adversarial attacks.  
**Ideal Range**: As close as possible to original accuracy.

### Robustness Ratio
**Definition**: Ratio of adversarial accuracy to original accuracy.  
**Interpretation**: Values closer to 1.0 indicate more robust models.  
**Ideal Range**:
- Poor: < 0.3
- Fair: 0.3-0.6
- Good: 0.6-0.8
- Excellent: > 0.8

## Attack-Specific Metrics

### FGSM Attack
**Definition**: Fast Gradient Sign Method, a simple one-step attack.  
**Epsilon**: Perturbation magnitude, higher values indicate stronger attacks.

### PGD Attack
**Definition**: Projected Gradient Descent, a stronger iterative attack.  
**Epsilon**: Perturbation magnitude.  
**Alpha**: Step size for each iteration.  
**Iterations**: Number of attack iterations, higher values indicate stronger attacks.

## Versioning Metrics

### Version
**Definition**: Unique identifier for a model version.  
**Interpretation**: Used for tracking model lineage.

### Status
**Definition**: Current deployment status of the model.  
**Levels**:
- **Registered**: Initial state
- **Staging**: Ready for testing
- **Production**: Deployed to production

## How to Use These Metrics

1. **Model Selection**: Compare models based on parameters, memory usage, and inference time to select the most appropriate model for your deployment target.

2. **Optimization Strategy**: Use size reduction and inference speedup to evaluate different optimization techniques.

3. **Security Assessment**: Review vulnerabilities and security score to identify potential security issues.

4. **Robustness Evaluation**: Use robustness score and adversarial accuracy to assess model resilience against attacks.

5. **Version Management**: Track model versions and their metrics to ensure continuous improvement.

Remember that the ideal values for these metrics depend on your specific use case and constraints. Neural-Scope provides these metrics to help you make informed decisions about your models.
