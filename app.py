import streamlit as st
import json
import os
from glob import glob
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
def load_metrics(directory="test_results"):
    metrics = {}
    files = glob(os.path.join(directory, "*.json"))

    if not files:
        st.error(f"No JSON files found in {directory}. Please run the model analysis tests first.")
        return {}

    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                file_name = Path(file).stem
                metrics[file_name] = data

        except json.JSONDecodeError as e:
            st.error(f"Error reading {file}. The file might be corrupted. {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred while reading {file}: {e}")

    return metrics



metrics = load_metrics()

if metrics:
    # Create tabs for different sections
    tabs = st.tabs([
        "Model Overview",
        "Performance Analysis",
        "Memory Analysis",
        "Complexity Analysis",
        "Cloud Cost Analysis",
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
            
    # Complexity Analysis Tab
    with tabs[3]:
        st.header("Complexity Analysis")
        if "complexity_analysis" in metrics:
            complexity = metrics["complexity_analysis"]
            summary = complexity.get("summary", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Time Complexity")
                st.write(f"Time Complexity: {summary.get('time_complexity', 'N/A')}")
                st.write(f"Time Quality: {summary.get('time_quality', 'N/A')}")
                if "plots" in complexity:
                    st.image(complexity["plots"]["runtime"], caption="Runtime vs. Input Size")
            
            with col2:
                st.subheader("Space Complexity")
                st.write(f"Space Complexity: {summary.get('space_complexity', 'N/A')}")
                st.write(f"Space Quality: {summary.get('space_quality', 'N/A')}")
                if "space_complexity.png" in os.listdir("test_results"):
                    st.image("test_results/space_complexity.png", caption="Memory vs Batch Size")

    # Cloud Cost Analysis Tab
    with tabs[4]:
        st.header("Cloud Cost Analysis")
        if "cloud_costs" in metrics:
            costs = metrics["cloud_costs"]
            estimates = costs.get("estimates", {})
            recommendations = costs.get("recommendations", {})

            # Create bar chart of cloud costs
            cost_data = {"Provider": list(estimates.keys()), "Monthly Cost": [e["monthly_cost"] for e in estimates.values()]}
            fig = px.bar(
                cost_data,
                x="Provider",
                y="Monthly Cost",
                title="Monthly Cloud Cost Estimates"
            )
            st.plotly_chart(fig)
            
            st.subheader("Recommendations")
            st.write(f"Optimal Provider: {recommendations.get('optimal_provider', 'N/A')}")
            st.write(f"Cost Savings: {recommendations.get('cost_savings_percentage', 'N/A'):.2f}%")

    
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
    with tabs[5]:
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