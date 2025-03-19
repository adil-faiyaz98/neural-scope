# ml_patterns.py
MLPatternDatabase = {
    "Linear Regression": {
        "patterns": [r"LinearRegression", r"np\.linalg\.lst", r"gradient descent", r"normal equation"],
        "std_complexity": "O(n^3) for normal eq, O(n*m) per iteration for GD",
        "optimizations": ["Use mini-batch gradient descent or normal eq on smaller data, L-BFGS for large dims"]
    },
    "KNN": {
        "patterns": [r"KNeighborsClassifier", r"knn", r"distance\(.*\)", r"for i in range.*distance.*", r"nearest neighbors"],
        "std_complexity": "O(n^2) brute force, O(log n) with KD-Tree or approximate methods",
        "optimizations": ["Switch to KD-Tree or Annoy/FAISS for approximate NN to reduce search from O(n^2)"]
    }
}

def get_algorithm_details(algorithm_name):
    return MLPatternDatabase.get(algorithm_name, {"error": "Algorithm not found"})
