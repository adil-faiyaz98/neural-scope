from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DriftAnalyzer:
    def __init__(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        self.reference_data = reference_data
        self.current_data = current_data
        self.drift_scores = {}

    def detect_drift(self, features: list, contamination: float = 0.05):
        scaler = StandardScaler()
        reference_scaled = scaler.fit_transform(self.reference_data[features])
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(reference_scaled)

        current_scaled = scaler.transform(self.current_data[features])
        predictions = iso_forest.predict(current_scaled)
        drifted_indices = np.where(predictions == -1)[0]
        drift_score = len(drifted_indices) / len(self.current_data)

        self.drift_scores = {
            'drift_score': drift_score,
            'drifted_indices': drifted_indices
        }
        return self.drift_scores

    def visualize_drift(self, feature: str):
        plt.figure(figsize=(10, 6))
        plt.hist(self.reference_data[feature], bins=30, alpha=0.5, label='Reference Data', color='blue')
        plt.hist(self.current_data[feature], bins=30, alpha=0.5, label='Current Data', color='red')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def report_drift(self):
        if self.drift_scores['drift_score'] > 0:
            print(f"Data Drift Detected: Drift Score = {self.drift_scores['drift_score']:.4f}")
            print(f"Indices of Drifted Samples: {self.drift_scores['drifted_indices']}")
        else:
            print("No significant data drift detected.")