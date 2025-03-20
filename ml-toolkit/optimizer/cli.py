from argparse import ArgumentParser
from optimizer.performance import measure_performance
from optimizer.memory_profiler import profile_memory
from optimizer.gpu_profiler import profile_gpu

def main():
    parser = ArgumentParser(description="Optimizer CLI for performance checks and optimizations.")
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Performance command
    performance_parser = subparsers.add_parser('performance', help='Measure and optimize performance')
    performance_parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    performance_parser.add_argument('--data', type=str, required=True, help='Path to the dataset for evaluation')

    # Memory profiling command
    memory_parser = subparsers.add_parser('memory', help='Profile memory usage during model training')
    memory_parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    memory_parser.add_argument('--data', type=str, required=True, help='Path to the dataset for evaluation')

    # GPU profiling command
    gpu_parser = subparsers.add_parser('gpu', help='Profile GPU usage during model training')
    gpu_parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    gpu_parser.add_argument('--data', type=str, required=True, help='Path to the dataset for evaluation')

    args = parser.parse_args()

    if args.command == 'performance':
        measure_performance(args.model, args.data)
    elif args.command == 'memory':
        profile_memory(args.model, args.data)
    elif args.command == 'gpu':
        profile_gpu(args.model, args.data)

if __name__ == "__main__":
    main()