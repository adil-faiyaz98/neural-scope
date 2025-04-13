import streamlit as st
import json
import os
from glob import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import traceback # For better error reporting during debugging

# --- Configuration ---
RESULTS_DIR = "test_results" # Directory containing JSON analysis files

st.set_page_config(page_title="Neural Scope Analysis", layout="wide")

# --- Title and Description ---
st.title("Neural Scope Analysis Dashboard")
st.markdown(f"""
This dashboard visualizes comprehensive analysis results for models, code, and data,
sourced from JSON files in the `{RESULTS_DIR}` directory.
""")

# --- Data Loading ---
@st.cache_data # Cache the loaded data
def load_analysis_results(directory=RESULTS_DIR):
    """Loads all JSON analysis results from the specified directory."""
    results = {}
    json_files = glob(os.path.join(directory, "*.json"))

    if not json_files:
        st.warning(f"No JSON analysis files found in the '{directory}' directory.")
        st.info("Please run the analysis commands (e.g., `neural-scope analyze-code`, `neural-scope profile-model`) to generate results.")
        return None # Return None if no files found

    for file_path in json_files:
        try:
            file_name = Path(file_path).stem
            with open(file_path, "r") as f:
                data = json.load(f)
                results[file_name] = data # Store data keyed by filename (without extension)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON from '{file_path}': {e}. File might be corrupted or not valid JSON.")
        except Exception as e:
            st.error(f"Error reading file '{file_path}': {e}")
            st.error(traceback.format_exc()) # Show full traceback for debugging

    if not results:
         st.error(f"Found JSON files, but failed to load any valid data from '{directory}'. Check file contents and permissions.")
         return None

    return results

# --- Main App Logic ---
analysis_results = load_analysis_results()

if analysis_results:
    # --- Sidebar: Select Analysis Report ---
    st.sidebar.header("Select Analysis Report")
    # Use filenames as keys for selection
    available_reports = list(analysis_results.keys())
    selected_report_key = st.sidebar.selectbox(
        "Choose a report to view:",
        options=available_reports,
        index=0 # Default to the first report
    )

    # Get the data for the selected report
    metrics = analysis_results.get(selected_report_key, {})

    if not metrics:
        st.error(f"Selected report '{selected_report_key}' seems to be empty or invalid.")
    else:
        st.header(f"Analysis Report: `{selected_report_key}.json`")

        # --- Create Tabs ---
        # Ensure correct number of tabs and unique assignments
        tab_titles = [
            "Model Overview",       # 0
            "Performance",          # 1
            "Memory",               # 2
            "Complexity",           # 3
            "Layer Details",        # 4
            "Cloud Cost",           # 5 (If applicable)
            "Optimization",         # 6 (If applicable)
            "Code Analysis",        # 7 (If applicable)
            "Data Quality"          # 8 (If applicable)
        ]
        # Dynamically create tabs based on available data sections? Or keep fixed?
        # Let's keep fixed for now, and sections will show "N/A" if data is missing.
        tabs = st.tabs(tab_titles)

        # --- Tab 0: Model Overview ---
        with tabs[0]:
            st.subheader("Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model Name:** {metrics.get('model_name', 'N/A')}")
                st.write(f"**Framework:** {metrics.get('framework', 'N/A')}")
                st.write(f"**Device:** {metrics.get('device', 'N/A')}")

                # Parameters
                params = metrics.get("parameters", {})
                if params: # Only show if parameter info exists
                    st.subheader("Parameters")
                    st.metric("Total Parameters", f"{params.get('total', 0):,}")
                    st.metric("Trainable Parameters", f"{params.get('trainable', 0):,}")
                    st.metric("Non-trainable Parameters", f"{params.get('non_trainable', 0):,}")
                else:
                     st.info("Parameter information not found in this report.")

            with col2:
                # Architecture Layer Distribution
                arch = metrics.get("architecture", {})
                layer_counts = arch.get("layer_counts") # Safely get layer_counts
                if layer_counts and isinstance(layer_counts, dict) and layer_counts:
                    st.subheader("Layer Type Distribution")
                    try:
                        fig = px.pie(
                            values=list(layer_counts.values()),
                            names=list(layer_counts.keys()),
                            title="Layer Types"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not plot layer distribution: {e}")
                else:
                    st.info("Layer count information not found or invalid in this report.")

        # --- Tab 1: Performance Analysis ---
        with tabs[1]:
            st.subheader("Performance Metrics")
            perf = metrics.get("performance", {})
            if perf:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Inference Time")
                    inf_time = perf.get("inference_time_ms", {})
                    if inf_time and isinstance(inf_time, dict):
                        # Display stats directly instead of a potentially misleading box plot for single values
                        st.metric("Mean Inference Time (ms)", f"{inf_time.get('mean', 'N/A'):.2f}" if isinstance(inf_time.get('mean'), (int, float)) else 'N/A')
                        st.write(f"**Std Dev:** {inf_time.get('std', 'N/A'):.2f}" if isinstance(inf_time.get('std'), (int, float)) else 'N/A')
                        st.write(f"**Min:** {inf_time.get('min', 'N/A'):.2f}" if isinstance(inf_time.get('min'), (int, float)) else 'N/A')
                        st.write(f"**Max:** {inf_time.get('max', 'N/A'):.2f}" if isinstance(inf_time.get('max'), (int, float)) else 'N/A')
                        # If raw times were available (e.g., key 'raw_times': [list of times]), a box plot would be:
                        # raw_times = inf_time.get('raw_times')
                        # if raw_times and isinstance(raw_times, list):
                        #     fig = px.box(y=raw_times, title="Inference Time Distribution (ms)", points="all")
                        #     fig.update_layout(yaxis_title="Time (ms)")
                        #     st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Inference time data not found or invalid.")

                with col2:
                    st.subheader("Compute Requirements (FLOPs)")
                    compute = metrics.get("compute", {})
                    flops_data = compute.get("flops", {})
                    if flops_data and isinstance(flops_data, dict):
                        gflops = flops_data.get('gflops')
                        st.metric("Total GFLOPs", f"{gflops:.2f}" if isinstance(gflops, (int, float)) else "N/A")

                        flops_by_layer = flops_data.get("by_layer")
                        if flops_by_layer and isinstance(flops_by_layer, dict) and flops_by_layer:
                            try:
                                fig = px.bar(
                                    x=list(flops_by_layer.keys()),
                                    y=list(flops_by_layer.values()),
                                    title="FLOPs by Layer Type",
                                    labels={"x": "Layer Type", "y": "FLOPs"}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not plot FLOPs by layer: {e}")
                        else:
                            st.info("FLOPs breakdown by layer type not available.")
                    else:
                        st.info("FLOPs data not found or invalid.")
            else:
                st.info("Performance section not found in this report.")

        # --- Tab 2: Memory Analysis ---
        with tabs[2]:
            st.subheader("Memory Usage")
            memory = metrics.get("memory", {})
            if memory and isinstance(memory, dict):
                mem_params_mb = memory.get("parameters_mb", 0)
                mem_buffers_mb = memory.get("buffers_mb", 0)
                mem_activations_mb = memory.get("activation_mb", 0) # Corrected key? Check JSON source
                mem_total_mb = memory.get("total_mb")

                # Ensure values are numeric for plotting
                mem_data = {
                    "Category": [],
                    "Memory (MB)": []
                }
                if isinstance(mem_params_mb, (int, float)):
                    mem_data["Category"].append("Parameters")
                    mem_data["Memory (MB)"].append(mem_params_mb)
                if isinstance(mem_buffers_mb, (int, float)):
                     mem_data["Category"].append("Buffers")
                     mem_data["Memory (MB)"].append(mem_buffers_mb)
                if isinstance(mem_activations_mb, (int, float)):
                     mem_data["Category"].append("Activations") # Changed label
                     mem_data["Memory (MB)"].append(mem_activations_mb)

                if mem_data["Category"]:
                    try:
                        fig = px.bar(
                            pd.DataFrame(mem_data),
                            x="Category",
                            y="Memory (MB)",
                            title="Memory Usage Breakdown (MB)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                         st.error(f"Could not plot memory usage: {e}")

                if isinstance(mem_total_mb, (int, float)):
                     st.metric("Estimated Total Memory (MB)", f"{mem_total_mb:.2f}")
                else:
                     st.info("Total memory information not available.")

            else:
                st.info("Memory analysis section not found in this report.")

        # --- Tab 3: Complexity Analysis ---
        with tabs[3]:
            st.subheader("Algorithmic Complexity")
            complexity = metrics.get("complexity_analysis", {})
            if complexity and isinstance(complexity, dict):
                summary = complexity.get("summary", {})
                plots = complexity.get("plots", {}) # Safely get plots dict

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Time Complexity")
                    st.write(f"**Estimated Big-O:** {summary.get('time_complexity', 'N/A')}")
                    st.write(f"**Quality:** {summary.get('time_quality', 'N/A')}")

                    # Safely get and check plot path
                    runtime_plot_path = plots.get("runtime")
                    if runtime_plot_path and isinstance(runtime_plot_path, str):
                         # Assume path is relative to RESULTS_DIR if not absolute
                         full_plot_path = runtime_plot_path if os.path.isabs(runtime_plot_path) else os.path.join(RESULTS_DIR, runtime_plot_path)
                         if os.path.exists(full_plot_path):
                              st.image(full_plot_path, caption="Runtime vs. Input Size")
                         else:
                              st.warning(f"Runtime plot image not found at: {full_plot_path}")
                    else:
                         st.info("Runtime plot data not available.")

                with col2:
                    st.subheader("Space Complexity")
                    st.write(f"**Estimated Big-O:** {summary.get('space_complexity', 'N/A')}")
                    st.write(f"**Quality:** {summary.get('space_quality', 'N/A')}")

                    # Check for the specific hardcoded image, assuming it's in RESULTS_DIR
                    space_plot_path = os.path.join(RESULTS_DIR, "space_complexity.png")
                    if os.path.exists(space_plot_path):
                        st.image(space_plot_path, caption="Memory vs. Batch Size")
                    else:
                        # Also check if path is provided in JSON
                        space_plot_path_json = plots.get("space") # Assuming key 'space'
                        if space_plot_path_json and isinstance(space_plot_path_json, str):
                            full_plot_path = space_plot_path_json if os.path.isabs(space_plot_path_json) else os.path.join(RESULTS_DIR, space_plot_path_json)
                            if os.path.exists(full_plot_path):
                                st.image(full_plot_path, caption="Memory vs. Input/Batch Size")
                            else:
                                st.warning(f"Space complexity plot image not found at: {full_plot_path}")
                        else:
                            st.info("Space complexity plot data not available.")
            else:
                st.info("Complexity analysis section not found in this report.")


        # --- Tab 4: Layer Details Analysis ---
        with tabs[4]:
            st.subheader("Layer Details")
            layers = metrics.get("layer_analysis", {}) # Changed key based on original code
            if layers and isinstance(layers, dict):
                layer_data = []
                for name, info in layers.items():
                    if isinstance(info, dict): # Ensure info is a dictionary
                        layer_data.append({
                            "name": name,
                            "type": info.get("type", "N/A"),
                            "parameters": info.get("parameters", 0),
                            "param_percentage": info.get("param_percentage", 0.0)
                        })
                    else:
                        st.warning(f"Invalid data format for layer '{name}'. Skipping.")

                if layer_data:
                    df_layers = pd.DataFrame(layer_data)

                    # Bar chart of parameter distribution
                    try:
                        fig = px.bar(
                            df_layers.sort_values("parameters", ascending=False).head(30), # Show top 30 layers by params
                            x="name",
                            y="parameters",
                            color="type",
                            title="Parameter Count Across Layers (Top 30)",
                            labels={"name": "Layer Name", "parameters": "Number of Parameters"}
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not plot layer parameter distribution: {e}")

                    # Show layer details in a table
                    st.dataframe(df_layers)
                else:
                    st.info("No valid layer data found to display.")
            else:
                st.info("Layer analysis section not found in this report.")

        # --- Tab 5: Cloud Cost Analysis ---
        with tabs[5]:
            st.subheader("Cloud Cost Estimation")
            costs = metrics.get("cloud_costs", {})
            if costs and isinstance(costs, dict):
                estimates = costs.get("estimates", {})
                recommendations = costs.get("recommendations", {})

                if estimates and isinstance(estimates, dict):
                    # Prepare data for bar chart
                    cost_data_list = []
                    for provider, details in estimates.items():
                        if isinstance(details, dict) and 'monthly_cost' in details:
                             cost_value = details['monthly_cost']
                             if isinstance(cost_value, (int, float)):
                                 cost_data_list.append({"Provider": provider, "Monthly Cost ($)": cost_value})

                    if cost_data_list:
                        try:
                            df_costs = pd.DataFrame(cost_data_list)
                            fig = px.bar(
                                df_costs,
                                x="Provider",
                                y="Monthly Cost ($)",
                                title="Estimated Monthly Cloud Costs"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not plot cloud costs: {e}")
                    else:
                        st.info("No valid cost estimate data found.")

                else:
                    st.info("Cost estimates not available.")

                if recommendations and isinstance(recommendations, dict):
                    st.subheader("Recommendations")
                    st.write(f"**Optimal Provider:** {recommendations.get('optimal_provider', 'N/A')}")
                    savings = recommendations.get('cost_savings_percentage')
                    if isinstance(savings, (int, float)):
                        st.write(f"**Potential Savings:** {savings:.2f}%")
                    else:
                        st.write(f"**Potential Savings:** N/A")
                else:
                    st.info("Cost recommendations not available.")
            else:
                st.info("Cloud cost analysis section not found in this report.")

        # --- Tab 6: Optimization Suggestions ---
        with tabs[6]:
            st.subheader("Optimization Suggestions")
            opt_suggestions_data = metrics.get("optimization_suggestions", {})
            if opt_suggestions_data and isinstance(opt_suggestions_data, dict):
                suggestions = opt_suggestions_data.get("suggestions", []) # Default to empty list

                if suggestions and isinstance(suggestions, list):
                    # Group suggestions by category
                    categories = sorted(list(set(s.get("category", "Uncategorized") for s in suggestions if isinstance(s, dict))))

                    for category in categories:
                        st.markdown(f"#### {category.replace('_', ' ').title()}")
                        category_suggestions = [s for s in suggestions if isinstance(s, dict) and s.get("category") == category]

                        for i, suggestion in enumerate(category_suggestions):
                             # Safely access suggestion details
                             title = suggestion.get("suggestion", f"Suggestion {i+1}")
                             impact = suggestion.get("impact", "N/A")
                             priority = suggestion.get("priority", "N/A")
                             details = suggestion.get("details", "No details provided.") # Add details field

                             with st.expander(f"{title} (Priority: {priority}, Impact: {impact})"):
                                 st.write(f"**Details:** {details}")
                                 # Optionally add more fields if they exist in your JSON
                                 # st.write(f"**Affected Layers:** {suggestion.get('affected_layers', 'N/A')}")

                else:
                    st.info("No optimization suggestions found in this report.")
            else:
                st.info("Optimization suggestions section not found in this report.")

        # --- Tab 7: Code Analysis ---
        with tabs[7]:
             st.subheader("Code Analysis Summary")
             # Assuming code analysis results might be nested under a key like 'code_analysis'
             # Or if the entire JSON is from analyze_code, access directly
             if "files" in metrics and isinstance(metrics["files"], dict): # Check if it looks like analyze_code output
                 st.info("Displaying aggregated code analysis results.")
                 total_files = len(metrics["files"])
                 total_issues = 0
                 avg_complexity = 0
                 file_complexities = []

                 st.write(f"Analyzed **{total_files}** Python file(s).")

                 all_issues_list = []
                 for file_path, file_results in metrics["files"].items():
                     if isinstance(file_results, dict):
                         issues = file_results.get("issues", [])
                         complexity = file_results.get("complexity", {}).get("cyclomatic_complexity", {}).get("average")

                         if isinstance(issues, list):
                             total_issues += len(issues)
                             for issue in issues:
                                 if isinstance(issue, dict):
                                      all_issues_list.append({
                                          "File": Path(file_path).name,
                                          "Line": issue.get("lineno", "N/A"),
                                          "Type": issue.get("type", "N/A"),
                                          "Description": issue.get("description", "N/A")
                                      })
                         if isinstance(complexity, (int, float)):
                             file_complexities.append(complexity)

                 if file_complexities:
                     avg_complexity = sum(file_complexities) / len(file_complexities)
                     st.metric("Average Cyclomatic Complexity", f"{avg_complexity:.2f}")

                 st.metric("Total Issues Found", total_issues)

                 if all_issues_list:
                     st.dataframe(pd.DataFrame(all_issues_list))
                 else:
                     st.success("No issues found in the analyzed code.")

             else:
                 st.info("Code analysis results not found in this report format.")


        # --- Tab 8: Data Quality ---
        with tabs[8]:
            st.subheader("Data Quality Report")
            # Data quality reports might be structured differently.
            # Assuming the JSON contains keys like 'summary_stats', 'missing_values', 'duplicates' etc.
            # This is a placeholder - adjust based on actual DataGuardian output format if it saves to JSON.
            # The current cli.py saves DataGuardian output as text/html/json string representation of the pandas DataFrame/Series report.
            # If the JSON output is just a string dump, we can't easily parse it here.
            # Suggestion: Modify cli.py analyze_data to save a structured JSON if JSON format is chosen.
            # Example structure assumption:
            # { "report_type": "data_quality", "summary": {...}, "missing": {...}, ... }

            if metrics.get("report_type") == "data_quality":
                 st.write("Data Quality Summary:")
                 summary = metrics.get("summary", {})
                 if summary:
                     st.json(summary, expanded=False) # Display raw summary JSON
                 else:
                     st.info("Summary statistics not available.")

                 missing = metrics.get("missing_values", {})
                 if missing:
                     st.write("Missing Values:")
                     st.dataframe(pd.Series(missing, name="Missing Count"))
                 else:
                     st.info("Missing value analysis not available.")

                 # Add sections for duplicates, outliers, etc. based on expected JSON structure
            else:
                 st.info("Data quality report not found in this report format.")
                 st.info("Note: The current `analyze_data --format json` might save a JSON string representation, not a structured JSON. This needs adjustment in `cli.py` for optimal dashboard display.")


# --- Footer or final messages ---
st.sidebar.markdown("---")
st.sidebar.info("Dashboard powered by Streamlit and Neural Scope.")

# Add a check for dependencies
try:
    import pandas
    import plotly
except ImportError:
    st.error("Missing required libraries. Please install them: pip install pandas plotly")

