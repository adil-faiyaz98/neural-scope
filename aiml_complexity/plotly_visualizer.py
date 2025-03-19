"""
Interactive Visualization for AI/ML Complexity and Profiling Data

Integrates with:
- PostgresStorage (storage.py) to fetch analysis results
- Plotly for advanced interactive charts: line, scatter, heatmaps, CPU/GPU usage trends

Usage:
from aiml_complexity.visualization import PlotlyVisualizer
viz = PlotlyVisualizer()
fig = viz.generate_time_vs_input_figure(db_config, "KNN")
fig.show()
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import numpy as np

# If you have an ML pattern DB, you can import it:
# from aiml_complexity.ml_patterns import MLPatternDatabase
# For demonstration, we'll define a minimal placeholder here:
MLPatternDatabase = {
    "KNN": {
        "std_complexity": "O(n^2)",
        "optimizations": ["Use KD-Tree or approximate nearest neighbors for large n"]
    },
    # ... other algorithms
}

try:
    from aiml_complexity.storage import PostgresStorage
except ImportError:
    # For demonstration, define a stub PostgresStorage here
    class PostgresStorage:
        def __init__(self, **kwargs):
            pass
        def fetch_all(self, algo_name=None):
            return []
        def close(self):
            pass


class PlotlyVisualizer:
    def __init__(self):
        pass

    def plot_time_vs_input(self, analysis_data: List[Dict[str, Any]], title="Time vs. Input Size"):
        """
        analysis_data: list of dicts, each with keys
        {
          "id": int,
          "algorithm_name": str,
          "theoretical_complexity": str,
          "empirical_results": {
            "empirical_data": [
               {"input_size": X, "time_sec": Y, "memory_mb": Z}, ...
            ]
          }
        }
        """
        fig = go.Figure()
        for entry in analysis_data:
            algo = entry.get("algorithm_name", "UnknownAlgo")
            entry_id = entry.get("id", None)
            emp_data = entry.get("empirical_results", {}).get("empirical_data", [])
            if not emp_data:
                continue
            x_vals = [d["input_size"] for d in emp_data if "input_size" in d]
            y_vals = [d["time_sec"] for d in emp_data if "time_sec" in d]
            # We skip if there's no data
            if not x_vals or not y_vals:
                continue

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=f"{algo} (id={entry_id})"
            ))
        fig.update_layout(title=title,
                          xaxis_title="Input Size",
                          yaxis_title="Time (sec)",
                          hovermode="x unified")
        return fig

    def compare_against_standard(self, algo_name: str, found_complexity: str):
        """
        Compare the found_complexity with the standard from MLPatternDatabase.
        Return a textual summary.
        """
        best_key = None
        for k in MLPatternDatabase.keys():
            if k.lower() in algo_name.lower() or algo_name.lower() in k.lower():
                best_key = k
                break
        if best_key:
            db_info = MLPatternDatabase[best_key]
            standard = db_info.get("std_complexity", "Unknown")
            if found_complexity == standard:
                return f"Your code complexity matches standard: {found_complexity}."
            else:
                return f"Detected: {found_complexity}, Standard: {standard}. Potential mismatch or suboptimal approach?"
        else:
            return "Algorithm not found in database for standard comparison."

    def generate_time_vs_input_figure(self, db_config: Dict[str, Any], algo_name=None):
        """
        Fetch all runs from the DB for a given algo (or all if None), and plot time vs input size using Plotly.
        """
        storage = PostgresStorage(**db_config)
        try:
            rows = storage.fetch_all(algo_name)
        finally:
            storage.close()
        return self.plot_time_vs_input(rows, title=f"Time vs. Input for {algo_name or 'all'}")

    def plot_cpu_usage(self, cpu_samples: List[Dict[str, Any]], title="CPU Usage Over Time"):
        """
        cpu_samples: list of dicts e.g.:
        [
          {"timestamp": 1.234, "cpu_percent": 50.0},
          {"timestamp": 1.284, "cpu_percent": 55.2},
          ...
        ]
        Plot a line graph showing CPU usage over time (seconds).
        """
        if not cpu_samples:
            fig = go.Figure()
            fig.update_layout(title="No CPU usage data")
            return fig
        x_vals = [d["timestamp"] for d in cpu_samples]
        y_vals = [d["cpu_percent"] for d in cpu_samples]
        fig = go.Figure(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name="CPU %"))
        fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="CPU Usage (%)",
                          hovermode="x unified")
        return fig

    def plot_heatmap_cpu_usage(self, data_matrix: np.ndarray, x_vals: np.ndarray, y_vals: np.ndarray,
                               title="CPU Usage Heatmap"):
        """
        data_matrix: 2D np array shape (len(y_vals), len(x_vals)) or similar, representing CPU usage or threads usage.
        x_vals: e.g. time
        y_vals: e.g. core indexes
        This yields a heatmap over time vs cores, CPU usage.
        """
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            hoverongaps=False
        ))
        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Core Index")
        return fig

    def generate_interactive_dashboard(self, db_config: Dict[str, Any], host="127.0.0.1", port=8050):
        """
        Example of a minimal Dash-based interactive dashboard
        that shows time-vs-input plot and CPU usage if stored.
        """
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
        import json
        storage = PostgresStorage(**db_config)

        # We'll just fetch all data once for simplicity
        all_results = storage.fetch_all()

        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H3("AI/ML Complexity Interactive Dashboard"),
            dcc.Dropdown(
                id='algo-dropdown',
                options=[{'label': row['algorithm_name'], 'value': row['algorithm_name']}
                         for row in all_results],
                multi=False,
                placeholder="Select an algorithm"
            ),
            dcc.Graph(id='time-vs-input-graph'),
            dcc.Graph(id='cpu-usage-graph'),
        ])

        @app.callback(
            [Output('time-vs-input-graph', 'figure'),
             Output('cpu-usage-graph', 'figure')],
            [Input('algo-dropdown', 'value')]
        )
        def update_graphs(selected_algo):
            if not selected_algo:
                # return empty figs
                return [go.Figure(), go.Figure()]
            # Filter DB rows for the selected algo
            filtered = [r for r in all_results if r["algorithm_name"] == selected_algo]
            time_fig = self.plot_time_vs_input(filtered, title=f"Time vs Input: {selected_algo}")

            # For CPU usage, we might store usage as part of empirical_results
            # e.g. empirical_results -> { "cpu_usage_samples": [{"timestamp": 0.1, "cpu_percent": 40}, ...] }
            # We'll combine them if multiple rows
            cpu_samples_all = []
            for row in filtered:
                emp = row["empirical_results"]
                if emp and "cpu_usage_samples" in emp:
                    # We'll assume each sample has a 'timestamp' or we generate one
                    # This is a simplified approach
                    row_samples = emp["cpu_usage_samples"]
                    # We might unify them by offsetting timestamps to avoid collisions
                    for s in row_samples:
                        cpu_samples_all.append(s)
            # Sort by timestamp
            cpu_samples_all.sort(key=lambda x: x.get("timestamp", 0))
            # Now plot
            if cpu_samples_all:
                cpu_fig = self.plot_cpu_usage(cpu_samples_all, title=f"CPU Usage: {selected_algo}")
            else:
                cpu_fig = go.Figure()
                cpu_fig.update_layout(title="No CPU usage data found")
            return [time_fig, cpu_fig]

        app.run_server(debug=False, host=host, port=port)

