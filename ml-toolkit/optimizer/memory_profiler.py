from memory_profiler import memory_usage
import psutil
import time
import matplotlib.pyplot as plt

class MemoryProfiler:
    def __init__(self):
        self.memory_usage_data = []

    def profile_memory(self, func, *args, **kwargs):
        """
        Profiles the memory usage of a function.

        Parameters:
        func (callable): The function to profile.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

        Returns:
        The return value of the function being profiled.
        """
        start_time = time.time()
        self.memory_usage_data = memory_usage((func, args, kwargs), interval=0.1)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds.")
        return self.memory_usage_data

    def plot_memory_usage(self):
        """
        Plots the memory usage data collected during profiling.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.memory_usage_data, label='Memory Usage (MB)')
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory Usage (MB)')
        plt.legend()
        plt.grid()
        plt.show()

    def get_memory_info(self):
        """
        Returns the current memory usage of the process.
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss': mem_info.rss / (1024 ** 2),  # Resident Set Size in MB
            'vms': mem_info.vms / (1024 ** 2),  # Virtual Memory Size in MB
            'shared': mem_info.shared / (1024 ** 2),  # Shared Memory in MB
            'text': mem_info.text / (1024 ** 2),  # Text (code) in MB
            'lib': mem_info.lib / (1024 ** 2),  # Library in MB
            'data': mem_info.data / (1024 ** 2),  # Data in MB
            'dirty': mem_info.dirty / (1024 ** 2)  # Dirty pages in MB
        }