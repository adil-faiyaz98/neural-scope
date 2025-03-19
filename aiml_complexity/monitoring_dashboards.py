import psycopg2
from psycopg2 import sql
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import datetime

# Optional: For real-time OS metrics (if needed for profiling)
try:
    import psutil  # to get CPU usage, etc.
except ImportError:
    psutil = None

# Optional: For cloud API integration (placeholders for actual API calls)
try:
    import boto3  # AWS SDK for Cost Explorer
except ImportError:
    boto3 = None

class PostgresStorage:
    """Handles PostgreSQL storage of analysis results and provides query functions."""
    def __init__(self, db_config):
        """
        Initialize the database connection using db_config dict with keys:
        host, port, database, user, password.
        """
        self.db_config = db_config
        # Connect to PostgreSQL
        self.conn = psycopg2.connect(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database'),
            user=db_config.get('user'),
            password=db_config.get('password')
        )
        self.cur = self.conn.cursor()
        # Ensure the analysis_results table exists
        self._create_tables()

    def _create_tables(self):
        """Create tables for storing analysis results and recommendations if not exist."""
        # Table for analysis results
        create_analysis_table = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id SERIAL PRIMARY KEY,
            algorithm VARCHAR(100),
            input_size BIGINT,
            parameters JSONB,  -- JSONB to store various metrics (time, memory, etc.)
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        # Table for recommendations (optional, could be merged into analysis_results as JSON)
        create_recommend_table = """
        CREATE TABLE IF NOT EXISTS analysis_recommendations (
            analysis_id INTEGER REFERENCES analysis_results(id),
            recommendation TEXT
        );
        """
        self.cur.execute(create_analysis_table)
        self.cur.execute(create_recommend_table)
        self.conn.commit()

    def save_analysis(self, algorithm, input_size, metrics_dict, recommendations=None):
        """
        Save a new analysis record to the database.
        :param algorithm: Name of the algorithm or test.
        :param input_size: Size of input data (if applicable, else use 0 or None).
        :param metrics_dict: Dictionary of measured metrics (e.g. {"time_sec": 1.23, "memory_mb": 45.6, "cpu_percent": 50, "gpu_percent": 80}).
        :param recommendations: Optional list of optimization recommendations (strings).
        """
        # Convert metrics_dict to JSON for storage
        params_json = json.dumps(metrics_dict)
        insert_query = """
        INSERT INTO analysis_results (algorithm, input_size, parameters)
        VALUES (%s, %s, %s) RETURNING id;
        """
        self.cur.execute(insert_query, (algorithm, input_size, params_json))
        analysis_id = self.cur.fetchone()[0]
        if recommendations:
            for rec in recommendations:
                self.cur.execute(
                    "INSERT INTO analysis_recommendations (analysis_id, recommendation) VALUES (%s, %s);",
                    (analysis_id, rec)
                )
        self.conn.commit()
        return analysis_id

    def query_analysis(self, algorithm=None, since=None, until=None):
        """
        Query past analysis results, optionally filtering by algorithm name and time range.
        :param algorithm: Filter by algorithm name (exact match) if provided.
        :param since: Filter results after this timestamp (datetime or string).
        :param until: Filter results before this timestamp.
        :return: List of dicts for each analysis result (with metrics merged).
        """
        query = "SELECT id, algorithm, input_size, parameters, timestamp FROM analysis_results"
        conditions = []
        params = []
        if algorithm:
            conditions.append("algorithm = %s")
            params.append(algorithm)
        if since:
            conditions.append("timestamp >= %s")
            # ensure since is datetime or convert
            since_dt = since if isinstance(since, datetime.datetime) else datetime.datetime.fromisoformat(str(since))
            params.append(since_dt)
        if until:
            conditions.append("timestamp <= %s")
            until_dt = until if isinstance(until, datetime.datetime) else datetime.datetime.fromisoformat(str(until))
            params.append(until_dt)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp;"
        self.cur.execute(query, tuple(params))
        rows = self.cur.fetchall()
        results = []
        for row in rows:
            rec_id, algo, size, params_json, ts = row
            metrics = json.loads(params_json) if params_json else {}
            results.append({
                "id": rec_id,
                "algorithm": algo,
                "input_size": size,
                "timestamp": ts,
                **metrics  # merge metrics into the dict
            })
        return results

    def get_recommendations(self, analysis_id):
        """Retrieve any stored recommendations for a given analysis record."""
        self.cur.execute("SELECT recommendation FROM analysis_recommendations WHERE analysis_id = %s;", (analysis_id,))
        rows = self.cur.fetchall()
        return [r[0] for r in rows]

    def close(self):
        """Close the database connection."""
        self.cur.close()
        self.conn.close()

class Visualizer:
    """Generates interactive Plotly visualizations for performance metrics."""
    def __init__(self):
        # Could hold state or configuration for plots (e.g., theme or default template).
        pass

    def plot_metric(self, data, x_field, y_field, chart_type="line", title=None, x_title=None, y_title=None):
        """
        Create an interactive Plotly chart for the given data.
        :param data: A list of dictionaries or a pandas DataFrame containing the data.
        :param x_field: The field name for the x-axis.
        :param y_field: The field name for the y-axis (or list of fields for multiple series).
        :param chart_type: Type of chart: "line", "bar", "scatter", "heatmap", "3d", etc.
        :param title: Chart title.
        :param x_title: X-axis title.
        :param y_title: Y-axis title.
        :return: A Plotly Graph Objects Figure.
        """
        fig = go.Figure()
        # Support multiple y fields (for multiple traces) or single
        if isinstance(y_field, list):
            # Multiple series on the same chart
            for y in y_field:
                fig.add_trace(go.Scatter(x=[row[x_field] for row in data],
                                         y=[row[y] for row in data],
                                         mode='lines+markers',
                                         name=y))
        else:
            # Single series chart
            if chart_type == "line":
                fig.add_trace(go.Scatter(x=[row[x_field] for row in data],
                                         y=[row[y_field] for row in data],
                                         mode='lines+markers',
                                         name=y_field))
            elif chart_type == "bar":
                fig.add_trace(go.Bar(x=[row[x_field] for row in data],
                                     y=[row[y_field] for row in data],
                                     name=y_field))
            elif chart_type == "scatter":
                fig.add_trace(go.Scatter(x=[row[x_field] for row in data],
                                         y=[row[y_field] for row in data],
                                         mode='markers',
                                         name=y_field))
            # Additional chart types (heatmap, etc.) can be added as needed.
            # For brevity, we handle a few common types.
        # Set titles
        fig.update_layout(title=title or (y_field if isinstance(y_field, str) else "Metrics"),
                          xaxis_title=x_title or x_field,
                          yaxis_title=y_title or (y_field if isinstance(y_field, str) else "value"),
                          hovermode="x unified")
        # Enable dynamic tooltips (Plotly does this by default; hovermode 'x unified' shows one tooltip for all traces at same x)
        return fig

    def add_annotations(self, fig, annotations):
        """
        Add annotation texts to a given figure.
        :param fig: Plotly Figure to annotate.
        :param annotations: List of dicts with annotation parameters (x, y, text, etc.).
        """
        for ann in annotations:
            fig.add_annotation(**ann)
        return fig

    def export_data(self, data, format="csv", filename=None):
        """
        Export analysis data (or latest query result) to CSV or JSON. Can also export chart to HTML.
        :param data: Data to export (list of dicts or pandas DataFrame).
        :param format: "csv", "json", or "html".
        :param filename: Optional filename to save to. If not provided, returns string.
        """
        if format == "csv":
            # If pandas is available and data is suitable, use it for convenience
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                if filename:
                    df.to_csv(filename, index=False)
                    return None
                else:
                    return df.to_csv(index=False)
            except ImportError:
                # Manual CSV if pandas not available
                import csv
                output = []
                fieldnames = list(data[0].keys()) if data else []
                if filename:
                    with open(filename, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in data:
                            writer.writerow(row)
                    return None
                else:
                    # Write to string
                    output.append(",".join(fieldnames))
                    for row in data:
                        line = ",".join(str(row[field]) for field in fieldnames)
                        output.append(line)
                    return "\n".join(output)
        elif format == "json":
            if filename:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4, default=str)
                return None
            else:
                return json.dumps(data, indent=4, default=str)
        elif format == "html":
            # Expecting data is a Plotly figure in this case to export the chart
            # If a figure object is passed as data, export that
            if hasattr(data, "to_html"):
                html_str = data.to_html(full_html=True)
                if filename:
                    with open(filename, 'w') as f:
                        f.write(html_str)
                    return None
                else:
                    return html_str
            else:
                # If data is not a figure, we can return a simple HTML table of the data.
                html = "<table border='1'><tr>" + "".join(f"<th>{col}</th>" for col in data[0].keys()) + "</tr>"
                for row in data:
                    html += "<tr>" + "".join(f"<td>{row[col]}</td>" for col in row.keys()) + "</tr>"
                html += "</table>"
                if filename:
                    with open(filename, 'w') as f:
                        f.write(html)
                    return None
                else:
                    return html

class Profiler:
    """Profiles code execution and detects inefficiencies for AI/ML workloads."""
    def __init__(self):
        # We could allow toggling certain profiling aspects (like enabling GPU checks).
        pass

    def profile_function(self, func, *args, **kwargs):
        """
        Execute the given function with provided arguments, measuring time, CPU, and GPU usage.
        Returns a tuple (metrics_dict, recommendations).
        """
        metrics = {}
        recommendations = []
        # Measure CPU usage before and after (if psutil is available)
        cpu_usage_before = psutil.cpu_percent(interval=None) if psutil else None

        # GPU metrics (if torch available)
        gpu_used_before = None
        gpu_util_before = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_used_before = torch.cuda.memory_allocated()
                # torch doesn't have direct util% API; could use NVML via pynvml if installed (not in this snippet for simplicity).
        except ImportError:
            pass

        # Memory (RAM) usage before
        mem_before = psutil.virtual_memory().used if psutil else None

        start_time = time.time()
        result = func(*args, **kwargs)  # Execute the target function
        end_time = time.time()
        exec_time = end_time - start_time
        metrics["execution_time_sec"] = round(exec_time, 6)

        # Measure CPU usage after
        cpu_usage_after = psutil.cpu_percent(interval=None) if psutil else None
        if cpu_usage_before is not None and cpu_usage_after is not None:
            # We could take an average or just use after as representative
            metrics["cpu_percent"] = cpu_usage_after

        # Measure memory after
        mem_after = psutil.virtual_memory().used if psutil else None
        if mem_before is not None and mem_after is not None:
            metrics["memory_used_mb"] = round((mem_after - mem_before) / (1024*1024), 2)

        # GPU metrics after
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # ensure all GPU ops finished
                gpu_used_after = torch.cuda.memory_allocated()
                gpu_peak = torch.cuda.max_memory_allocated()
                metrics["gpu_memory_used_mb"] = round((gpu_used_after - gpu_used_before) / (1024*1024), 2) if gpu_used_before is not None else round(gpu_used_after / (1024*1024), 2)
                metrics["gpu_peak_memory_mb"] = round(gpu_peak / (1024*1024), 2)
                # (For GPU utilization, one could sample nvidia-smi or use torch.profiler, omitted here)
        except ImportError:
            pass

        # Basic recommendations based on metrics:
        # 1. If CPU usage is high and execution time is long relative to work done, suggest optimizing CPU code or parallelizing.
        if metrics.get("cpu_percent") and metrics["cpu_percent"] > 80:
            recommendations.append("High CPU usage detected; consider multi-threading or optimizing CPU-bound sections.")
        # 2. If memory usage increased a lot, suggest reviewing memory leaks or using more memory-efficient structures.
        if metrics.get("memory_used_mb") and metrics["memory_used_mb"] > 500:  # threshold example
            recommendations.append("Significant memory usage; consider optimizing data structures or processing in batches to lower memory footprint.")
        # 3. If GPU memory usage is low and execution is slow, maybe the GPU was underutilized or most work done on CPU.
        if metrics.get("gpu_memory_used_mb") is not None:
            if metrics["gpu_memory_used_mb"] < 1.0 and "execution_time_sec" in metrics:
                recommendations.append("GPU usage appears low; ensure operations are on GPU and consider increasing batch size to better utilize GPU.")
        # 4. If execution time is long and GPU not fully utilized, maybe I/O or CPU is a bottleneck.
        if metrics.get("cpu_percent") and metrics.get("gpu_memory_used_mb") is not None:
            if metrics["cpu_percent"] > 50 and metrics["gpu_memory_used_mb"] < 10:
                recommendations.append("GPU was likely waiting on CPU or I/O; consider using asynchronous data loaders or optimizing input pipeline.")
        # 5. If GPU memory peak is close to device limit (not measured here directly), we could suggest checking batch size or model size.

        # (More sophisticated analysis could integrate cProfile for function-level timings or use torch.profiler for detailed insights.)
        return metrics, recommendations

    def analyze_pytorch_model(self, model, inputs):
        """
        (Optional) Specific profiling for PyTorch models – e.g., using torch.autograd.profiler.
        For brevity, not fully implemented. Could run model(inputs) and measure layer times.
        """
        # This could use torch.profiler to get low-level details.
        metrics, recs = self.profile_function(model, inputs)
        # Additional analysis: if using PyTorch and multiple GPUs available but only one used:
        try:
            import torch
            if torch.cuda.device_count() > 1:
                # If multiple GPUs and only one used, suggest DataParallel or DDP.
                recs.append("Multiple GPUs detected but model ran on one. Consider DistributedDataParallel for multi-GPU scaling&#8203;:contentReference[oaicite:17]{index=17}.")
        except ImportError:
            pass
        return metrics, recs

class CloudCostAnalyzer:
    """Optional integration with cloud APIs to analyze and reduce cloud costs."""
    def __init__(self, provider=None, credentials=None):
        """
        Initialize cost analyzer with specific cloud provider (e.g., 'AWS', 'GCP').
        credentials: e.g., AWS keys or GCP service account info.
        """
        self.provider = provider
        self.credentials = credentials
        # If AWS, initialize boto3 client for Cost Explorer
        self.aws_client = None
        if provider == 'AWS' and boto3:
            # Assume credentials are set in environment or provided via credentials dict
            self.aws_client = boto3.client('ce', **credentials) if credentials else boto3.client('ce')
        # (For GCP, could integrate Google Cloud Billing API similarly.)

    def estimate_run_cost(self, runtime_seconds, instance_type=None):
        """
        Estimate cost of a run given runtime and instance type.
        If cloud API is available and configured, fetch actual pricing. Otherwise, use static mapping or return None.
        """
        hours = runtime_seconds / 3600.0
        if self.provider == 'AWS':
            # This is a simplistic approach: in real scenario, query pricing API or Cost Explorer.
            # For demonstration, let's assume a pricing dictionary for some instance types (USD/hour).
            pricing = {
                "g4dn.xlarge": 0.526,  # example on-demand rate
                "p3.2xlarge": 3.06,
                "t3.medium": 0.0416
            }
            if instance_type in pricing:
                return round(pricing[instance_type] * hours, 4)
        # If provider not AWS or instance not in our map, return None to indicate unknown.
        return None

    def get_cloud_cost_history(self, start_date, end_date):
        """
        Query the cloud provider's billing API to get cost between start_date and end_date.
        Only AWS implemented as example using Cost Explorer API.
        """
        if self.provider == 'AWS' and self.aws_client:
            # Using AWS Cost Explorer API to get total cost in the time period
            # Note: Requires appropriate IAM permissions.
            response = self.aws_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='MONTHLY',
                Metrics=["UnblendedCost"]
            )
            # Parse response to get the total cost
            if "ResultsByTime" in response:
                cost = 0.0
                for item in response["ResultsByTime"]:
                    amount = float(item["Total"]["UnblendedCost"]["Amount"])
                    cost += amount
                return cost
        # For other providers or if API not available, return None
        return None

    def recommend_cost_savings(self, utilization_metrics):
        """
        Provide recommendations to reduce cost based on utilization metrics.
        e.g., If GPU utilization is low, suggest smaller instance; if CPU is bottleneck, suggest different instance type, etc.
        """
        recs = []
        gpu_util = utilization_metrics.get("gpu_util_percent")
        cpu_util = utilization_metrics.get("cpu_percent")
        if gpu_util is not None and gpu_util < 30:
            recs.append("GPU utilization is low; consider using a cheaper instance or consolidating workloads to fewer GPUs&#8203;:contentReference[oaicite:18]{index=18}.")
        if cpu_util is not None and gpu_util is not None:
            if cpu_util < 50 and gpu_util < 50:
                recs.append("Both CPU and GPU utilization are low; you might over-provisioned resources. Scale down instance size to save costs.")
        # If multiple GPUs are present but not all used:
        if utilization_metrics.get("gpu_count") and utilization_metrics.get("gpu_active_count"):
            total = utilization_metrics["gpu_count"]
            active = utilization_metrics["gpu_active_count"]
            if total > active:
                recs.append(f"Only {active}/{total} GPUs are utilized; consider releasing unused GPUs or using a smaller instance with fewer GPUs to cut costs.")
        return recs

class RealTimeDashboard:
    """Sets up a live dashboard for real-time performance monitoring using Plotly Dash."""
    def __init__(self, storage: PostgresStorage, update_interval=5):
        """
        :param storage: PostgresStorage instance to query data from in real-time.
        :param update_interval: Update interval in seconds for the dashboard.
        """
        self.storage = storage
        self.update_interval = update_interval
        # Dash app will be initialized when run_dashboard is called to avoid unnecessary dependency if not used.
        self.app = None

    def run_dashboard(self, host="127.0.0.1", port=8050):
        """
        Launch a Dash server that auto-refreshes charts with live data from the storage.
        """
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output

        # Initialize Dash app
        app = Dash(__name__)
        app.layout = html.Div([
            html.H3("Real-Time Performance Dashboard"),
            dcc.Graph(id='live-metrics-graph'),
            dcc.Interval(id='interval-component', interval=self.update_interval*1000, n_intervals=0)
        ])

        # Define the callback to update graph
        @app.callback(Output('live-metrics-graph', 'figure'),
                      Input('interval-component', 'n_intervals'))
        def update_graph(n):
            # Query the latest analysis results (could limit to last N or last few minutes)
            data = self.storage.query_analysis(since=datetime.datetime.now() - datetime.timedelta(minutes=5))
            if not data:
                # If no data, return an empty figure or placeholder
                fig = go.Figure()
                fig.update_layout(title="No data available yet")
                return fig
            # For example, plot execution time over timestamp for a specific algorithm or overall
            df = data  # using list of dicts
            # Use Plotly Express for quick plotting
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                fig = px.line(df, x="timestamp", y="execution_time_sec", color="algorithm",
                              title="Execution Time (last 5 minutes)", labels={"timestamp": "Time", "execution_time_sec": "Execution Time (s)"})
            except ImportError:
                # Fallback to go if pandas/px not available
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[row["timestamp"] for row in data],
                                         y=[row.get("execution_time_sec", 0) for row in data],
                                         mode='lines+markers', name="Execution Time"))
                fig.update_layout(title="Execution Time (last 5 minutes)", xaxis_title="Time", yaxis_title="Execution Time (s)")
            return fig

        # Run the Dash app server
        app.run_server(debug=False, host=host, port=port)
        self.app = app

# Example usage (for illustration; in practice, this code would be in a main or separate script):
if __name__ == "__main__":
    # Database configuration (fill with actual credentials)
    db_conf = {
        "host": "localhost",
        "port": 5432,
        "database": "performance_db",
        "user": "postgres",
        "password": "postgres"
    }
    storage = PostgresStorage(db_conf)
    visualizer = Visualizer()
    profiler = Profiler()
    cost_analyzer = CloudCostAnalyzer(provider="AWS")  # assume AWS credentials set in environment
    dashboard = RealTimeDashboard(storage)

    # Example: profile a dummy function
    def dummy_algorithm(n):
        # Dummy workload: sum of 0..n using a Python loop (inefficient on purpose)
        s = 0
        for i in range(n):
            s += i
        return s

    metrics, recs = profiler.profile_function(dummy_algorithm, 1000000)
    print("Metrics:", metrics)
    print("Recommendations:", recs)
    # Save the results to the database
    storage.save_analysis("DummyAlgo", input_size=1000000, metrics_dict=metrics, recommendations=recs)

    # Query history and plot it
    history = storage.query_analysis("DummyAlgo")
    fig = visualizer.plot_metric(history, x_field="timestamp", y_field="execution_time_sec", chart_type="line",
                                 title="DummyAlgo Execution Time Over Time", x_title="Timestamp", y_title="Time (s)")
    # Export the plot to HTML
    visualizer.export_data(fig, format="html", filename="dummy_algo_time.html")

    # Run real-time dashboard (this will block and start the server)
    # dashboard.run_dashboard()
