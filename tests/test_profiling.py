import time
from aiml_complexity import profiling

def test_profile_time_results():
    # Define a simple function for testing
    def add(a, b):
        return a + b
    result, elapsed = profiling.profile_time(add, 3, 4)
    assert result == 7
    assert elapsed >= 0.0

def test_profile_time_timing():
    # Test that timing roughly captures sleep duration
    def sleeper(t):
        time.sleep(t)
        return "done"
    result, elapsed = profiling.profile_time(sleeper, 0.05)
    assert result == "done"
    assert elapsed >= 0.05

def test_profile_memory_usage():
    def allocate_list(n):
        # allocate a list of n zeros
        return [0] * n
    # Profile memory usage for allocating a list
    result, mem_used = profiling.profile_memory(allocate_list, 10000)
    # Should return correct list and some memory usage > 0
    assert isinstance(result, list)
    assert len(result) == 10000
    assert mem_used > 0
