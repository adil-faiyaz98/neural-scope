# performance_profiler.py
import time

class MLPerformanceProfiler:
    def __init__(self, func):
        self.func = func

    def run_profile(self, inputs, repeat=3):
        empirical_data = []
        for inp in inputs:
            start_t = time.time()
            for _ in range(repeat):
                self.func(inp)
            end_t = time.time()
            empirical_data.append({"input_size": len(inp), "time_sec": round((end_t - start_t) / repeat, 5)})
        return {"empirical_data": empirical_data}
