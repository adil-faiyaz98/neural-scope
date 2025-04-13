import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Neural Model Analysis", layout="wide")

# Title and description
st.title("Neural Model Analysis Dashboard")
st.markdown("""
This dashboard visualizes comprehensive analysis results for neural network models,
including performance metrics, memory usage, and optimization recommendations.
""")

# Load metrics data
def load_metrics():
    metrics_file = Path("test_results/model_metrics.json")
    if not metrics_file.exists():
        st.error("No metrics file found. Please run the model analysis tests first.")
        return None
    
    try:
        with open(metrics_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error("Error reading metrics file. The file might be corrupted.")
        return None

metrics = load_metrics()

if metrics:
    # Create tabs for different sections
    tabs = st.tabs([
        "Model Overview",
        "Performance Analysis",
        "Memory Analysis",
        "Layer Analysis",
        "Optimization Suggestions"
    ])
    
    # Model Overview Tab
    with tabs[0]:
        st.header("Model Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            st.write(f"Model Name: {metrics.get('model_name', 'N/A')}")
            st.write(f"Framework: {metrics.get('framework', 'N/A')}")
            st.write(f"Device: {metrics.get('device', 'N/A')}")
            
            # Parameter counts
            params = metrics.get("parameters", {})
            st.subheader("Parameters")
            st.write(f"Total Parameters: {params.get('total', 0):,}")
            st.write(f"Trainable Parameters: {params.get('trainable', 0):,}")
            st.write(f"Non-trainable Parameters: {params.get('non_trainable', 0):,}")
        
        with col2:
            st.subheader("Architecture")
            if "architecture" in metrics:
                arch = metrics["architecture"]
                if "layer_counts" in arch:
                    # Create pie chart of layer distribution
                    fig = px.pie(
                        values=list(arch["layer_counts"].values()),
                        names=list(arch["layer_counts"].keys()),
                        title="Layer Type Distribution"
                    )
                    st.plotly_chart(fig)
    
    # Performance Analysis Tab
    with tabs[1]:
        st.header("Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Inference Time")
            if "performance" in metrics and "inference_time_ms" in metrics["performance"]:
                perf = metrics["performance"]["inference_time_ms"]
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=[perf["mean"]],
                    name="Mean",
                    boxpoints="all"
                ))
                fig.update_layout(
                    title="Inference Time Distribution (ms)",
                    yaxis_title="Time (ms)"
                )
                st.plotly_chart(fig)
                
                st.write(f"Mean: {perf['mean']:.2f} ms")
                st.write(f"Std: {perf['std']:.2f} ms")
                st.write(f"Min: {perf['min']:.2f} ms")
                st.write(f"Max: {perf['max']:.2f} ms")
        
        with col2:
            st.subheader("Compute Requirements")
            if "compute" in metrics and "flops" in metrics["compute"]:
                compute = metrics["compute"]["flops"]
                st.write(f"Total GFLOPs: {compute.get('gflops', 0):.2f}")
                
                # Create bar chart of FLOPs by layer type
                if "by_layer" in compute:
                    fig = px.bar(
                        x=list(compute["by_layer"].keys()),
                        y=list(compute["by_layer"].values()),
                        title="FLOPs by Layer Type",
                        labels={"x": "Layer Type", "y": "FLOPs"}
                    )
                    st.plotly_chart(fig)
    
    # Memory Analysis Tab
    with tabs[2]:
        st.header("Memory Analysis")
        if "memory" in metrics:
            memory = metrics["memory"]
            
            # Create stacked bar chart of memory usage
            memory_data = {
                "Category": ["Parameters", "Buffers", "Activation"],
                "Memory (MB)": [
                    memory.get("parameters_mb", 0),
                    memory.get("buffers_mb", 0),
                    memory.get("activation_mb", 0)
                ]
            }
            fig = px.bar(
                memory_data,
                x="Category",
                y="Memory (MB)",
                title="Memory Usage Breakdown"
            )
            st.plotly_chart(fig)
            
            st.write(f"Total Memory: {memory.get('total_mb', 0):.2f} MB")
    
    # Layer Analysis Tab
    with tabs[3]:
        st.header("Layer Analysis")
        if "layer_analysis" in metrics:
            layers = metrics["layer_analysis"]
            
            # Convert layer data to DataFrame for easier visualization
            layer_data = []
            for name, info in layers.items():
                layer_data.append({
                    "name": name,
                    "type": info["type"],
                    "parameters": info["parameters"],
                    "param_percentage": info["param_percentage"]
                })
            
            df = pd.DataFrame(layer_data)
            
            # Create bar chart of parameter distribution
            fig = px.bar(
                df,
                x="name",
                y="parameters",
                color="type",
                title="Parameter Distribution Across Layers",
                labels={"name": "Layer Name", "parameters": "Number of Parameters"}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
            
            # Show layer details in a table
            st.dataframe(df)
    
    # Optimization Suggestions Tab
    with tabs[4]:
        st.header("Optimization Suggestions")
        if "optimization_suggestions" in metrics:
            suggestions = metrics["optimization_suggestions"]["suggestions"]
            
            # Group suggestions by category
            for category in set(s["category"] for s in suggestions):
                st.subheader(category.replace("_", " ").title())
                category_suggestions = [s for s in suggestions if s["category"] == category]
                
                for suggestion in category_suggestions:
                    with st.expander(suggestion["suggestion"]):
                        st.write(f"**Impact:** {suggestion['impact']}")
                        st.write(f"**Priority:** {suggestion['priority']}")
else:
    st.warning("Please run the model analysis tests to generate metrics data.") 