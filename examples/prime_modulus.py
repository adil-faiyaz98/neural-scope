import time
import sys
import matplotlib.pyplot as plt
from collections import Counter
import random
import string


def sieve_of_atkin(limit):
    """
    Highly optimized implementation of the Sieve of Atkin algorithm.
    Returns a list of prime numbers up to the specified limit.
    """
    # Initialize sieve array with False values
    sieve = [False] * (limit + 1)
    # 2 and 3 are known primes
    if limit >= 2:
        sieve[2] = True
    if limit >= 3:
        sieve[3] = True

    # Mark potential primes based on quadratic forms
    # Step 1: Quadratic form: 4x² + y² where:
    #    x > 0 and y > 0
    #    Remainder of division by 12 is 1 or 5
    for x in range(1, int(limit ** 0.5) + 1):
        for y in range(1, int(limit ** 0.5) + 1):
            n = 4 * x * x + y * y
            if n <= limit and (n % 12 == 1 or n % 12 == 5):
                sieve[n] = not sieve[n]

    # Step 2: Quadratic form: 3x² + y² where:
    #    x > 0 and y > 0
    #    Remainder of division by 12 is 7
    for x in range(1, int(limit ** 0.5) + 1):
        for y in range(1, int(limit ** 0.5) + 1):
            n = 3 * x * x + y * y
            if n <= limit and n % 12 == 7:
                sieve[n] = not sieve[n]

    # Step 3: Quadratic form: 3x² - y² where:
    #    x > y > 0
    #    Remainder of division by 12 is 11
    for x in range(1, int(limit ** 0.5) + 1):
        for y in range(1, x):
            n = 3 * x * x - y * y
            if n <= limit and n % 12 == 11:
                sieve[n] = not sieve[n]

    # Step 4: Remove all multiples of squares of primes
    for n in range(5, int(limit ** 0.5) + 1):
        if sieve[n]:
            for k in range(n * n, limit + 1, n * n):
                sieve[k] = False

    # Convert to actual list of primes for comparison purposes
    return [p for p in range(2, limit + 1) if sieve[p]]


def sieve_of_eratosthenes(limit):
    """
    Optimized implementation of the Sieve of Eratosthenes algorithm.
    Returns a list of prime numbers up to the specified limit.
    """
    # Initialize the sieve array with all True values
    sieve = [True] * (limit + 1)

    # 0 and 1 are not prime
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False

    # Main sieve process - mark multiples of each prime as composite
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            # Start from i*i to avoid redundant marking
            # Use step i to mark all multiples
            for j in range(i * i, limit + 1, i):
                sieve[j] = False

    # Convert to a list of primes
    return [p for p in range(2, limit + 1) if sieve[p]]


def optimized_hash(string, modulus, base=33):
    """
    Enhanced polynomial rolling hash function with configurable base.
    Using the Bernstein hash algorithm (DJB2) with enhancements.
    """
    hash_value = 5381  # DJB2 starting value
    for char in string:
        # hash = ((hash * base) + hash) ^ ord(char)
        hash_value = ((hash_value << 5) + hash_value) ^ ord(char)
    return hash_value % modulus


def pick_prime(primes, min_size=1000):
    """Returns a suitable prime to use as modulus"""
    for prime in primes:
        if prime >= min_size:
            return prime
    # If no prime large enough exists, use last one on list
    if primes:
        return primes[-1]
    return min_size  # Fallback if no primes are found


def generate_test_strings(count, length=10):
    """Generate a specified number of random strings with given length"""
    return [''.join(random.choices(string.ascii_letters + string.digits, k=length))
            for _ in range(count)]


def analyze_hash_distribution(strings, hash_func, modulus):
    """Analyze the distribution of hash values"""
    hash_values = [hash_func(s, modulus) for s in strings]
    collisions = len(strings) - len(set(hash_values))
    counts = Counter(hash_values)
    max_bucket_size = max(counts.values()) if counts else 0

    return {
        "total_strings": len(strings),
        "unique_hashes": len(set(hash_values)),
        "collisions": collisions,
        "collision_rate": collisions / len(strings) if strings else 0,
        "max_bucket_size": max_bucket_size,
        "bucket_load_factor": max_bucket_size / (len(strings) / modulus) if strings else 0
    }


def benchmark_algorithm(func, *args, **kwargs):
    """Benchmark a function for time and memory usage"""
    # Time measurement
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time

    # Memory usage estimation - returned result size
    memory_usage = sys.getsizeof(result)

    return {
        "result": result,
        "execution_time": execution_time,
        "memory_usage": memory_usage
    }


def run_benchmarks():
    """Run comprehensive benchmarks for prime generation and hashing"""
    limits = [1000, 10000, 100000, 1000000]
    results = {
        "atkin": [],
        "eratosthenes": []
    }

    print("Running prime number generation benchmarks...")
    print("=" * 80)
    print(
        f"{'Limit':<10} | {'Algorithm':<15} | {'Execution Time (s)':<20} | {'Memory (bytes)':<15} | {'Primes Found':<15}")
    print("-" * 80)

    # Benchmark prime generation
    for limit in limits:
        # Benchmark Sieve of Atkin
        atkin_benchmark = benchmark_algorithm(sieve_of_atkin, limit)
        atkin_primes = atkin_benchmark["result"]
        results["atkin"].append({
            "limit": limit,
            "execution_time": atkin_benchmark["execution_time"],
            "memory_usage": atkin_benchmark["memory_usage"],
            "primes_count": len(atkin_primes)
        })

        print(
            f"{limit:<10} | {'Atkin':<15} | {atkin_benchmark['execution_time']:<20.6f} | {atkin_benchmark['memory_usage']:<15} | {len(atkin_primes):<15}")

        # Benchmark Sieve of Eratosthenes
        eratosthenes_benchmark = benchmark_algorithm(sieve_of_eratosthenes, limit)
        eratosthenes_primes = eratosthenes_benchmark["result"]
        results["eratosthenes"].append({
            "limit": limit,
            "execution_time": eratosthenes_benchmark["execution_time"],
            "memory_usage": eratosthenes_benchmark["memory_usage"],
            "primes_count": len(eratosthenes_primes)
        })

        print(
            f"{limit:<10} | {'Eratosthenes':<15} | {eratosthenes_benchmark['execution_time']:<20.6f} | {eratosthenes_benchmark['memory_usage']:<15} | {len(eratosthenes_primes):<15}")

        # Verify both algorithms produce the same result
        if set(atkin_primes) != set(eratosthenes_primes):
            print(f"WARNING: Algorithms produced different results for limit {limit}!")
            print(f"Atkin unique primes: {set(atkin_primes) - set(eratosthenes_primes)}")
            print(f"Eratosthenes unique primes: {set(eratosthenes_primes) - set(atkin_primes)}")

    print("\nRunning hash function benchmarks...")
    print("=" * 100)

    # Use the largest prime set for hashing benchmarks
    atkin_primes = results["atkin"][-1]["result"] if results["atkin"] else sieve_of_atkin(10000)
    modulus = pick_prime(atkin_primes, 10000)

    string_counts = [100, 1000, 10000]
    hash_results = []

    print(f"{'String Count':<15} | {'Collision Rate':<15} | {'Max Bucket Size':<15} | {'Execution Time (s)':<20}")
    print("-" * 100)

    for count in string_counts:
        test_strings = generate_test_strings(count)

        # Benchmark hash function
        start_time = time.time()
        hash_values = [optimized_hash(s, modulus) for s in test_strings]
        execution_time = time.time() - start_time

        # Analyze hash distribution
        analysis = analyze_hash_distribution(test_strings, optimized_hash, modulus)
        hash_results.append({
            "string_count": count,
            "collision_rate": analysis["collision_rate"],
            "max_bucket_size": analysis["max_bucket_size"],
            "execution_time": execution_time,
            **analysis
        })

        print(
            f"{count:<15} | {analysis['collision_rate']:<15.4f} | {analysis['max_bucket_size']:<15} | {execution_time:<20.6f}")

    # Generate similar strings to test hash resistance to slight changes
    similar_strings = ["".join(['a'] * 10) for _ in range(5)]
    for i in range(1, 5):
        similar_strings[i] = similar_strings[i][:i - 1] + 'b' + similar_strings[i][i:]

    print("\nTesting hash resistance to similar inputs:")
    print("-" * 50)
    for s in similar_strings:
        h = optimized_hash(s, modulus)
        print(f"Hash of '{s}': {h}")

    # Plot benchmark results
    plot_benchmark_results(results, hash_results)

    return {
        "prime_benchmarks": results,
        "hash_benchmarks": hash_results,
        "modulus_used": modulus
    }


def plot_benchmark_results(prime_results, hash_results):
    """Plot the benchmark results"""
    plt.figure(figsize=(15, 10))

    # Plot 1: Prime generation time complexity
    plt.subplot(2, 2, 1)
    limits = [r["limit"] for r in prime_results["atkin"]]
    atkin_times = [r["execution_time"] for r in prime_results["atkin"]]
    eratosthenes_times = [r["execution_time"] for r in prime_results["eratosthenes"]]

    plt.plot(limits, atkin_times, 'o-', label='Sieve of Atkin')
    plt.plot(limits, eratosthenes_times, 's-', label='Sieve of Eratosthenes')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Limit (log scale)')
    plt.ylabel('Execution Time (s) (log scale)')
    plt.title('Prime Generation Time Complexity')
    plt.legend()
    plt.grid(True)

    # Plot 2: Prime generation memory usage
    plt.subplot(2, 2, 2)
    atkin_memory = [r["memory_usage"] for r in prime_results["atkin"]]
    eratosthenes_memory = [r["memory_usage"] for r in prime_results["eratosthenes"]]

    plt.plot(limits, atkin_memory, 'o-', label='Sieve of Atkin')
    plt.plot(limits, eratosthenes_memory, 's-', label='Sieve of Eratosthenes')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Limit (log scale)')
    plt.ylabel('Memory Usage (bytes) (log scale)')
    plt.title('Prime Generation Space Complexity')
    plt.legend()
    plt.grid(True)

    # Plot 3: Hash collision rate
    plt.subplot(2, 2, 3)
    string_counts = [r["string_count"] for r in hash_results]
    collision_rates = [r["collision_rate"] for r in hash_results]

    plt.plot(string_counts, collision_rates, 'o-')
    plt.xlabel('Number of Strings')
    plt.ylabel('Collision Rate')
    plt.title('Hash Function Collision Rate')
    plt.grid(True)

    # Plot 4: Hash execution time
    plt.subplot(2, 2, 4)
    hash_times = [r["execution_time"] for r in hash_results]

    plt.plot(string_counts, hash_times, 'o-')
    plt.xlabel('Number of Strings')
    plt.ylabel('Total Execution Time (s)')
    plt.title('Hash Function Performance')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('prime_hash_benchmark.png')
    print("\nBenchmark plots saved to 'prime_hash_benchmark.png'")


if __name__ == '__main__':
    print("Prime Number Generation and Hash Function Analysis")
    print("=" * 50)

    try:
        results = run_benchmarks()

        print("\nSummary:")
        print("=" * 50)
        print(f"Best performing algorithm for prime generation: " +
              ("Sieve of Atkin" if results["prime_benchmarks"]["atkin"][-1]["execution_time"] <
                                   results["prime_benchmarks"]["eratosthenes"][-1][
                                       "execution_time"] else "Sieve of Eratosthenes"))

        print(f"Hash function modulus used: {results['modulus_used']}")
        print(
            f"Average collision rate: {sum(r['collision_rate'] for r in results['hash_benchmarks']) / len(results['hash_benchmarks']):.4f}")

        # Run a final test with a fixed set of strings
        test_array = ["alpha", "beta", "gamma", "delta", "epsilon",
                      "alpha1", "alpha2", "alphax", "alphaX", "alph4"]

        print("\nTest hashing with specific strings:")
        print("-" * 50)
        for string in test_array:
            hash_value = optimized_hash(string, results['modulus_used'])
            print(f"Hash of '{string}' is {hash_value}")

    except Exception as e:
        print(f"An error occurred: {e}")

    print("\nAnalysis complete.")