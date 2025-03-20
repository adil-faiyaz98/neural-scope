from time import time
import psutil
import numpy as np
import matplotlib.pyplot as plt

class PerformanceOptimizer:
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.cpu_usage = []

    def start_profiling(self):
        self.start_time = time()
        self.memory_usage.append(psutil.Process().memory_info().rss / (1024 * 1024))  # Convert to MB
        self.cpu_usage.append(psutil.cpu_percent(interval=None))

    def stop_profiling(self):
        elapsed_time = time() - self.start_time
        self.memory_usage.append(psutil.Process().memory_info().rss / (1024 * 1024))  # Convert to MB
        self.cpu_usage.append(psutil.cpu_percent(interval=None))
        return elapsed_time

    def plot_performance(self):
        plt.figure(figsize=(12, 6))

        # Plot memory usage
        plt.subplot(1, 2, 1)
        plt.plot(self.memory_usage, marker='o', color='blue')
        plt.title('Memory Usage (MB)')
        plt.xlabel('Time (iterations)')
        plt.ylabel('Memory (MB)')
        plt.grid()

        # Plot CPU usage
        plt.subplot(1, 2, 2)
        plt.plot(self.cpu_usage, marker='o', color='red')
        plt.title('CPU Usage (%)')
        plt.xlabel('Time (iterations)')
        plt.ylabel('CPU Usage (%)')
        plt.grid()

        plt.tight_layout()
        plt.show()

    def optimize_memory(self, model):
        # Placeholder for memory optimization logic
        pass

    def optimize_performance(self, model):
        # Placeholder for performance optimization logic
        pass

    def report_performance(self):
        elapsed_time = self.stop_profiling()
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Final Memory Usage: {self.memory_usage[-1]:.2f} MB")
        print(f"Final CPU Usage: {self.cpu_usage[-1]:.2f} %")