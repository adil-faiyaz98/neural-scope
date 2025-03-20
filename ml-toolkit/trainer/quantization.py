from typing import Any, Dict
import torch
import torch.quantization

def quantize_model(model: torch.nn.Module, 
                   data: Any, 
                   quantization_config: Dict[str, Any]) -> torch.nn.Module:
    """
    Quantizes the given PyTorch model using the specified configuration.

    Args:
        model (torch.nn.Module): The model to be quantized.
        data (Any): Sample input data for calibration.
        quantization_config (Dict[str, Any]): Configuration for quantization.

    Returns:
        torch.nn.Module: The quantized model.
    """
    # Set the model to evaluation mode
    model.eval()

    # Prepare the model for quantization
    model.qconfig = torch.quantization.get_default_qconfig(quantization_config.get('backend', 'fbgemm'))
    torch.quantization.prepare(model, inplace=True)

    # Calibrate the model with the provided data
    with torch.no_grad():
        model(data)

    # Convert the model to a quantized version
    quantized_model = torch.quantization.convert(model, inplace=False)

    return quantized_model

def save_quantized_model(quantized_model: torch.nn.Module, 
                          save_path: str) -> None:
    """
    Saves the quantized model to the specified path.

    Args:
        quantized_model (torch.nn.Module): The quantized model to save.
        save_path (str): The path where the model will be saved.
    """
    torch.save(quantized_model.state_dict(), save_path)

def load_quantized_model(model: torch.nn.Module, 
                         load_path: str) -> torch.nn.Module:
    """
    Loads the quantized model from the specified path.

    Args:
        model (torch.nn.Module): The model architecture to load weights into.
        load_path (str): The path from where the model will be loaded.

    Returns:
        torch.nn.Module: The model with loaded quantized weights.
    """
    model.load_state_dict(torch.load(load_path))
    return model

def quantization_pipeline(model: torch.nn.Module, 
                          data: Any, 
                          quantization_config: Dict[str, Any], 
                          save_path: str) -> torch.nn.Module:
    """
    Full pipeline for quantizing a model, calibrating it with data, and saving it.

    Args:
        model (torch.nn.Module): The model to be quantized.
        data (Any): Sample input data for calibration.
        quantization_config (Dict[str, Any]): Configuration for quantization.
        save_path (str): Path to save the quantized model.

    Returns:
        torch.nn.Module: The quantized model.
    """
    quantized_model = quantize_model(model, data, quantization_config)
    save_quantized_model(quantized_model, save_path)
    return quantized_model