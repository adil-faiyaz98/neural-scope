"""
High-Precision Performance Tracking and Optimization System for Transformer-Based Models

This Python code implements:
1. GPU Monitoring (TensorFlow & PyTorch)
2. CPU Optimization & Parallelism Profiling
3. Model-Specific Insights (Parsing Transformer / Model Configs)
4. PostgreSQL Data Storage for Longitudinal Analysis
5. Interactive Dash (Plotly) Dashboard for Visualization

The system aims to:
- Detect inefficiencies in large Transformer models (RoBERTa, DistilBERT, LLaMA, FLAN-TF, ViT, etc.).
- Track GPU/CPU usage, memory, parallelism, and highlight bottlenecks.
- Provide actionable advice for improved batch sizing, attention optimization, pruning, quantization, etc.
- Store all results in PostgreSQL for comparisons over time.
- Deliver interactive dashboards for AI/ML engineers to visualize results and tune performance.
"""

import os
import ast
import re
import json
import time
import datetime
import psutil
import threading
import textwrap

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import psycopg2
except ImportError:
    raise ImportError("psycopg2 is required for PostgreSQL integration")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output
except ImportError:
    raise ImportError("Plotly and Dash are required for interactive visualization")

###############################################################################
# 1. PostgreSQL Storage
###############################################################################

class PostgresStorage:
    """
    Handles PostgreSQL connections, storing and fetching performance data.
    """
    def __init__(self, host, port, dbname, user, password):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        self.cur = self.conn.cursor()
        self._setup_tables()

    def _setup_tables(self):
        """
        Create a table for storing model performance records if it doesn't exist.
        """
        create_table = """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            framework TEXT,        -- e.g. 'PyTorch' or 'TensorFlow'
            batch_size INT,
            gpu_mem_allocated_mb REAL,
            gpu_mem_reserved_mb REAL,
            gpu_util REAL,         -- in percent
            cpu_util REAL,         -- in percent
            step_time_ms REAL,     -- time per iteration or step
            notes TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cur.execute(create_table)
        self.conn.commit()

    def insert_performance_record(self, model_name, framework, batch_size,
                                  gpu_mem_allocated_mb, gpu_mem_reserved_mb,
                                  gpu_util, cpu_util,
                                  step_time_ms, notes=""):
        """
        Insert a new performance record into the database.
        """
        query = """
        INSERT INTO model_performance
        (model_name, framework, batch_size,
         gpu_mem_allocated_mb, gpu_mem_reserved_mb, gpu_util, cpu_util,
         step_time_ms, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self.cur.execute(query, (
            model_name,
            framework,
            batch_size,
            gpu_mem_allocated_mb,
            gpu_mem_reserved_mb,
            gpu_util,
            cpu_util,
            step_time_ms,
            notes
        ))
        self.conn.commit()

    def fetch_records(self, model_name=None):
        """
        Fetch performance records from the DB, optionally filtered by model_name.
        Return a list of dicts.
        """
        if model_name:
            self.cur.execute("""
                SELECT id, model_name, framework, batch_size,
                       gpu_mem_allocated_mb, gpu_mem_reserved_mb,
                       gpu_util, cpu_util, step_time_ms, notes, timestamp
                FROM model_performance
                WHERE model_name = %s
                ORDER BY timestamp;
            """, (model_name,))
        else:
            self.cur.execute("""
                SELECT id, model_name, framework, batch_size,
                       gpu_mem_allocated_mb, gpu_mem_reserved_mb,
                       gpu_util, cpu_util, step_time_ms, notes, timestamp
                FROM model_performance
                ORDER BY timestamp;
            """)
        rows = self.cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "model_name": r[1],
                "framework": r[2],
                "batch_size": r[3],
                "gpu_mem_allocated_mb": r[4],
                "gpu_mem_reserved_mb": r[5],
                "gpu_util": r[6],
                "cpu_util": r[7],
                "step_time_ms": r[8],
                "notes": r[9],
                "timestamp": r[10].isoformat()
            })
        return results

    def close(self):
        self.cur.close()
        self.conn.close()

###############################################################################
# 2. GPU & CPU Performance Profiling
###############################################################################

class GPUMonitor:
    """
    Tracks GPU usage for PyTorch or TensorFlow-based models.
    """
    def __init__(self, framework="PyTorch"):
        self.framework = framework

    def get_usage(self):
        """
        Return (allocated_mb, reserved_mb, gpu_util_percent).
        If not available, return (None, None, None).
        """
        gpu_alloc = None
        gpu_resv = None
        gpu_util = None
        if self.framework.lower() == "pytorch" and torch:
            if torch.cuda.is_available():
                # allocated vs reserved
                dev_id = torch.cuda.current_device()
                alloc = torch.cuda.memory_allocated(dev_id)
                resv = torch.cuda.memory_reserved(dev_id)
                gpu_alloc = round(alloc / (1024**2), 2)
                gpu_resv = round(resv / (1024**2), 2)
                # GPU util: we might parse from nvidia-smi or use a library.
                # Here we skip advanced steps, so return None or a placeholder.
                # Alternatively, could do a subprocess call to parse nvidia-smi.
        elif self.framework.lower() == "tensorflow" and tf:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info = tf.config.experimental.get_memory_info('GPU:0')
                gpu_alloc = round(info['current'] / (1024**2), 2)
                # 'peak' is also available
                # For 'reserved', TF doesn't provide a direct reserved measure, so set None
                gpu_resv = None
                # GPU util is not directly from TF
        return (gpu_alloc, gpu_resv, gpu_util)


class CPUProfiler:
    """
    Tracks CPU usage before/during/after a function call to detect
    suboptimal parallelism or single-thread usage.
    """
    def __init__(self):
        pass

    def profile_function(self, func, *args, **kwargs):
        """
        Run the function in a thread, measure CPU usage in the background.
        Returns (avg_cpu_util, max_cpu_util).
        """
        usage_samples = []
        def target_func():
            self.result = func(*args, **kwargs)

        t = threading.Thread(target=target_func)
        t.start()
        sample_interval = 0.05
        while t.is_alive():
            cpu_percent = psutil.cpu_percent(interval=None)
            usage_samples.append(cpu_percent)
            time.sleep(sample_interval)
        t.join()

        if len(usage_samples) == 0:
            return (0.0, 0.0)
        avg_cpu = sum(usage_samples) / len(usage_samples)
        max_cpu = max(usage_samples)
        return (avg_cpu, max_cpu)

###############################################################################
# 3. Model Parsing and Actionable Insights
###############################################################################

class TransformerAnalyzer:
    """
    Parses code or model definitions to identify
    big Transformer-based architectures,
    plus fine-tuning expansions or new layers.
    Also suggests improvements:
       - attention optimization
       - layer pruning or quantization
       - batch size improvements, etc.
    """
    def __init__(self, code_str):
        # We could parse code or reflect on the model.
        # We'll do a simple approach: searching for known references.
        self.code_str = code_str
        try:
            self.tree = ast.parse(textwrap.dedent(code_str))
        except:
            self.tree = None

    def detect_transformer_usage(self):
        """
        Return a list of identified models or sub-libs. e.g. "RoBERTa", "ViT", "StableDiffusion", etc.
        We'll match known keywords in the code.
        """
        patterns = {
            "roberta": ["roberta", "RobertaModel", "RobertaForSequenceClassification"],
            "distilbert": ["distilbert", "DistilBertModel"],
            "llama": ["llama", "LlamaForCausal"],
            "flan-t": ["FlanT5Model", "flan", "FlanForConditionalGeneration"],
            "vit": ["ViTModel", "VisionTransformer"],
            "stable_diff": ["StableDiffusionPipeline"],
            "midjourney": ["MidJourney", "MJ-Transform"],
            "resnet": ["ResNet50", "resnet"],
            "yolo": ["yolov5", "yolov7", "yolov8", "ultralytics.yolo"],
            # etc... more
        }
        found = []
        lower_code = self.code_str.lower()
        for key, pats in patterns.items():
            for p in pats:
                if p.lower() in lower_code:
                    found.append(key)
                    break
        return found

    def advise_transformer_optimizations(self):
        """
        Suggest advanced optimization for
        large Transformer-based or CV models
        (like attention improvements, pruning, etc.)
        """
        suggestions = []
        # We'll do some simplistic checks
        found_models = self.detect_transformer_usage()
        if "roberta" in found_models or "distilbert" in found_models or "llama" in found_models:
            suggestions.append("Consider using FlashAttention or a similar approach for memory/time efficient attention.")
            suggestions.append("Check if LoRA or adapter-based fine-tuning can reduce overhead instead of full model updates.")
        if "vit" in found_models:
            suggestions.append("For Vision Transformer, check patch size vs. input resolution. Large resolution may require attention optimizations or partial attention.")
        if "stable_diff" in found_models:
            suggestions.append("Stable Diffusion tip: use cross-attention optimizations or memory-efficient cross-attn. Also consider xFormers for faster training.")
        # ...
        return suggestions


def generate_batch_size_advice(batch_sizes, latencies):
    """
    Takes a list of batch_sizes and corresponding latencies,
    to produce an advice about optimal batch.
    """
    # We can do a simple approach: find the batch size with the best throughput
    # throughput = batch_size / latency
    # or store a small explanation
    best_throughput = 0.0
    best_bsz = None
    for b, l in zip(batch_sizes, latencies):
        if l <= 0:
            continue
        thr = b / (l)
        if thr > best_throughput:
            best_throughput = thr
            best_bsz = b
    if best_bsz is not None:
        return f"Based on latency vs. batch size, batch={best_bsz} yields highest throughput. Consider using that as default if memory allows."
    return "No batch size advice (insufficient data)."

###############################################################################
# 4. Putting It All Together: A High-Impact Example
###############################################################################

class TransformerPerfSystem:
    """
    A comprehensive class orchestrating:
    - GPU usage checks
    - CPU usage checks
    - Model code parsing
    - PostgreSQL insertion
    - Dash-based visualization
    """
    def __init__(self, db_config):
        self.db = PostgresStorage(**db_config)

    def analyze_model_code(self, code_str):
        """
        Perform static analysis of the code
        to detect advanced transformer usage,
        produce optimization suggestions.
        """
        analyzer = TransformerAnalyzer(code_str)
        suggestions = analyzer.advise_transformer_optimizations()
        return suggestions

    def profile_inference(self, framework, model_name, func, batch_size, repeat=5):
        """
        Profile an inference function (could be huggingface pipeline, torch model forward, etc.)
        capturing GPU/CPU usage, time, and store in PostgreSQL.
        """
        # 1) GPU usage
        gpu_monitor = GPUMonitor(framework)
        # 2) CPU usage
        cpu_profiler = CPUProfiler()

        # measure CPU usage in background,
        # as well as measure step time or average
        start_t = time.time()
        (avg_cpu, max_cpu) = cpu_profiler.profile_function(
            lambda: [func() for _ in range(repeat)]
        )
        end_t = time.time()

        step_time_ms = ((end_t - start_t) / repeat) * 1000.0
        (gpu_alloc, gpu_resv, gpu_util) = gpu_monitor.get_usage()

        if gpu_alloc is None:
            gpu_alloc = 0.0
        if gpu_resv is None:
            gpu_resv = 0.0
        # We didn't measure real GPU util, so set it to 0 or some placeholder
        if gpu_util is None:
            gpu_util = 0.0

        # Insert record into DB
        notes = ""
        if avg_cpu < 30.0:
            notes += "Low CPU usage found. Possibly I/O or GPU-limited.\n"
        elif avg_cpu > 70.0:
            notes += "High CPU usage indicates potential single-thread or GIL limitation.\n"

        self.db.insert_performance_record(
            model_name=model_name,
            framework=framework,
            batch_size=batch_size,
            gpu_mem_allocated_mb=gpu_alloc,
            gpu_mem_reserved_mb=gpu_resv,
            gpu_util=gpu_util,
            cpu_util=avg_cpu,
            step_time_ms=round(step_time_ms, 3),
            notes=notes
        )
        return {
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "gpu_alloc_mb": gpu_alloc,
            "gpu_resv_mb": gpu_resv,
            "gpu_util_percent": gpu_util,
            "step_time_ms": step_time_ms,
            "notes": notes
        }

    def run_dashboard(self, host="127.0.0.1", port=8050):
        """
        Launch a Dash app to visualize data from the 'model_performance' table.
        """
        app = Dash(__name__)

        # fetch all data
        data_all = self.db.fetch_records(model_name=None)
        model_names = sorted(list(set(d["model_name"] for d in data_all)))

        app.layout = html.Div([
            html.H1("Transformer Model Performance Dashboard"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": m, "value": m} for m in model_names],
                placeholder="Select a model to view metrics"
            ),
            dcc.Graph(id="time-vs-batch-graph"),
            dcc.Graph(id="gpu-cpu-scatter-graph")
        ])

        @app.callback(
            [Output("time-vs-batch-graph", "figure"),
             Output("gpu-cpu-scatter-graph", "figure")],
            [Input("model-dropdown", "value")]
        )
        def update_graphs(selected_model):
            if not selected_model:
                return go.Figure(), go.Figure()
            # filter data
            filtered = [r for r in data_all if r["model_name"] == selected_model]
            # time vs batch figure
            if not filtered:
                return go.Figure(), go.Figure()

            # create a DF-like structure
            x_vals = [f["batch_size"] for f in filtered]
            y_vals = [f["step_time_ms"] for f in filtered]
            # build a simple line or scatter
            fig_time = go.Figure(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+lines',
                name="Step Time (ms)"
            ))
            fig_time.update_layout(title=f"Step Time vs. Batch Size: {selected_model}",
                                   xaxis_title="Batch Size",
                                   yaxis_title="Time (ms)")

            # second figure: GPU vs CPU usage
            fig_gpu_cpu = go.Figure()
            fig_gpu_cpu.add_trace(go.Scatter(
                x=[f["gpu_util"] for f in filtered],
                y=[f["cpu_util"] for f in filtered],
                mode='markers',
                text=[f"id={f['id']}" for f in filtered],
                name="GPU vs CPU"
            ))
            fig_gpu_cpu.update_layout(title=f"GPU vs CPU Utilization: {selected_model}",
                                      xaxis_title="GPU Util (%)",
                                      yaxis_title="CPU Util (%)")

            return fig_time, fig_gpu_cpu

        app.run_server(debug=False, host=host, port=port)

    def close(self):
        self.db.close()


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    # DB config
    db_config = {
        "host": "localhost",
        "port": 5432,
        "dbname": "perfdb",
        "user": "postgres",
        "password": "postgres"
    }
    system = TransformerPerfSystem(db_config)

    # Example code snippet that includes references to a Transformer
    code_sample = r"""
import torch
from transformers import RobertaModel

model = RobertaModel.from_pretrained('roberta-base')
# Some fine-tuning code...
"""

    # Analyze static code for suggestions
    recommendations = system.analyze_model_code(code_sample)
    print("Static Analysis Transformer Advice:")
    for rec in recommendations:
        print(" -", rec)

    # Suppose we have a trivial inference function
    if torch:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        roberta = None
        try:
            from transformers import RobertaModel
            roberta = RobertaModel.from_pretrained('roberta-base').to(dev)
        except:
            pass

        def dummy_inference():
            if not roberta:
                return
            x = torch.randint(0, 1000, (8, 16)).to(dev)  # (batch=8, seq=16) as example
            with torch.no_grad():
                roberta(x)

        # Profile with batch_size=8, repeated=5 times
        result_metrics = system.profile_inference(
            framework="PyTorch",
            model_name="roberta-base-example",
            func=dummy_inference,
            batch_size=8,
            repeat=5
        )
        print("Profiling result:", result_metrics)

    # Launch Dash Dashboard to see results
    # system.run_dashboard(host="127.0.0.1", port=8050)

    system.close()
