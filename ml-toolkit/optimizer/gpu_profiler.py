from typing import List
import psutil
import GPUtil
import time
import matplotlib.pyplot as plt

class GPUProfiler:
    def __init__(self):
        self.gpu_stats = []

    def record_usage(self):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            self.gpu_stats.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,  # Convert to percentage
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            })

    def display_usage(self):
        for stat in self.gpu_stats:
            print(f"GPU ID: {stat['id']}, Name: {stat['name']}, Load: {stat['load']}%, "
                  f"Memory Used: {stat['memory_used']}MB/{stat['memory_total']}MB, "
                  f"Temperature: {stat['temperature']}°C")

    def plot_usage(self):
        ids = [stat['id'] for stat in self.gpu_stats]
        loads = [stat['load'] for stat in self.gpu_stats]
        memory_used = [stat['memory_used'] for stat in self.gpu_stats]
        memory_total = [stat['memory_total'] for stat in self.gpu_stats]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('GPU ID')
        ax1.set_ylabel('Load (%)', color='tab:blue')
        ax1.bar(ids, loads, color='tab:blue', alpha=0.6, label='Load (%)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Memory (MB)', color='tab:orange')
        ax2.bar(ids, memory_used, color='tab:orange', alpha=0.6, label='Memory Used (MB)')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        plt.title('GPU Usage Statistics')
        fig.tight_layout()
        plt.show()

    def profile(self, duration: int = 60):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.record_usage()
            time.sleep(1)  # Record every second
        self.display_usage()
        self.plot_usage()