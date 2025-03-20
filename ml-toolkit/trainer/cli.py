import argparse
from trainer import hyperparameter_tuning, quantization, distillation, drift_analyzer

def main():
    parser = argparse.ArgumentParser(description="CLI for managing model training tasks.")
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Hyperparameter tuning
    tuning_parser = subparsers.add_parser('tune', help='Hyperparameter tuning for models')
    tuning_parser.add_argument('--model', type=str, required=True, help='Model to tune')
    tuning_parser.add_argument('--params', type=str, required=True, help='Parameters for tuning in JSON format')

    # Model quantization
    quantization_parser = subparsers.add_parser('quantize', help='Quantize a model')
    quantization_parser.add_argument('--model', type=str, required=True, help='Model to quantize')
    quantization_parser.add_argument('--format', type=str, choices=['int8', 'float16'], required=True, help='Quantization format')

    # Model distillation
    distillation_parser = subparsers.add_parser('distill', help='Distill a model')
    distillation_parser.add_argument('--teacher', type=str, required=True, help='Teacher model')
    distillation_parser.add_argument('--student', type=str, required=True, help='Student model')

    # Drift analysis
    drift_parser = subparsers.add_parser('drift', help='Analyze data drift')
    drift_parser.add_argument('--data', type=str, required=True, help='Path to the dataset for drift analysis')

    args = parser.parse_args()

    if args.command == 'tune':
        hyperparameter_tuning.tune_model(args.model, args.params)
    elif args.command == 'quantize':
        quantization.quantize_model(args.model, args.format)
    elif args.command == 'distill':
        distillation.distill_model(args.teacher, args.student)
    elif args.command == 'drift':
        drift_analyzer.analyze_drift(args.data)

if __name__ == "__main__":
    main()