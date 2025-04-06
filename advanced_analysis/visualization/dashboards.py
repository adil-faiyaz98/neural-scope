"""
Interactive dashboards for visualizing analysis results.

This module provides tools for creating interactive dashboards to visualize
performance metrics, data quality assessments, and other analysis results.
"""

import logging
import json
import time
import datetime
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Visualization features will be limited.")

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not available. Interactive dashboard features will be limited.")

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not available. Database storage features will be limited.")


class PostgresStorage:
    """Handles PostgreSQL storage of analysis results and provides query functions."""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize the database connection using db_config dict with keys:
        host, port, database, user, password.
        
        Args:
            db_config: Database configuration dictionary
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL storage")
            
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
        self.cur.execute(create_analysis_table)
        
        # Table for performance metrics
        create_performance_table = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100),
            batch_size INTEGER,
            framework VARCHAR(50),
            device VARCHAR(50),
            throughput FLOAT,
            latency_ms FLOAT,
            memory_mb FLOAT,
            parameters JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cur.execute(create_performance_table)
        
        # Table for data quality metrics
        create_data_quality_table = """
        CREATE TABLE IF NOT EXISTS data_quality_metrics (
            id SERIAL PRIMARY KEY,
            dataset_name VARCHAR(100),
            completeness FLOAT,
            uniqueness FLOAT,
            consistency FLOAT,
            accuracy FLOAT,
            parameters JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cur.execute(create_data_quality_table)
        
        self.conn.commit()

    def store_analysis_result(self, algorithm: str, input_size: int, parameters: Dict[str, Any]):
        """
        Store an analysis result in the database.
        
        Args:
            algorithm: Algorithm name
            input_size: Input size
            parameters: Dictionary of parameters to store
        """
        query = """
        INSERT INTO analysis_results (algorithm, input_size, parameters)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        self.cur.execute(query, (algorithm, input_size, json.dumps(parameters)))
        result_id = self.cur.fetchone()[0]
        self.conn.commit()
        return result_id

    def store_performance_metric(self, model_name: str, batch_size: int, framework: str, 
                               device: str, throughput: float, latency_ms: float, 
                               memory_mb: float, parameters: Dict[str, Any]):
        """
        Store a performance metric in the database.
        
        Args:
            model_name: Model name
            batch_size: Batch size
            framework: Framework name (e.g., 'pytorch', 'tensorflow')
            device: Device name (e.g., 'cpu', 'cuda')
            throughput: Throughput in samples/second
            latency_ms: Latency in milliseconds
            memory_mb: Memory usage in MB
            parameters: Dictionary of additional parameters to store
        """
        query = """
        INSERT INTO performance_metrics 
        (model_name, batch_size, framework, device, throughput, latency_ms, memory_mb, parameters)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        self.cur.execute(query, (model_name, batch_size, framework, device, throughput, 
                                latency_ms, memory_mb, json.dumps(parameters)))
        result_id = self.cur.fetchone()[0]
        self.conn.commit()
        return result_id

    def store_data_quality_metric(self, dataset_name: str, completeness: float, 
                                uniqueness: float, consistency: float, accuracy: float, 
                                parameters: Dict[str, Any]):
        """
        Store a data quality metric in the database.
        
        Args:
            dataset_name: Dataset name
            completeness: Completeness score (0-1)
            uniqueness: Uniqueness score (0-1)
            consistency: Consistency score (0-1)
            accuracy: Accuracy score (0-1)
            parameters: Dictionary of additional parameters to store
        """
        query = """
        INSERT INTO data_quality_metrics 
        (dataset_name, completeness, uniqueness, consistency, accuracy, parameters)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        self.cur.execute(query, (dataset_name, completeness, uniqueness, 
                                consistency, accuracy, json.dumps(parameters)))
        result_id = self.cur.fetchone()[0]
        self.conn.commit()
        return result_id

    def get_performance_history(self, model_name: Optional[str] = None, 
                              days: int = 30) -> List[Dict[str, Any]]:
        """
        Get performance history for a model.
        
        Args:
            model_name: Model name (if None, get all models)
            days: Number of days to look back
            
        Returns:
            List of performance metrics
        """
        if model_name:
            query = """
            SELECT * FROM performance_metrics
            WHERE model_name = %s AND timestamp > NOW() - INTERVAL %s DAY
            ORDER BY timestamp;
            """
            self.cur.execute(query, (model_name, days))
        else:
            query = """
            SELECT * FROM performance_metrics
            WHERE timestamp > NOW() - INTERVAL %s DAY
            ORDER BY timestamp;
            """
            self.cur.execute(query, (days,))
            
        columns = [desc[0] for desc in self.cur.description]
        results = []
        
        for row in self.cur.fetchall():
            result = dict(zip(columns, row))
            # Parse JSONB parameters
            if 'parameters' in result and result['parameters']:
                result['parameters'] = json.loads(result['parameters'])
            results.append(result)
            
        return results

    def get_data_quality_history(self, dataset_name: Optional[str] = None, 
                               days: int = 30) -> List[Dict[str, Any]]:
        """
        Get data quality history for a dataset.
        
        Args:
            dataset_name: Dataset name (if None, get all datasets)
            days: Number of days to look back
            
        Returns:
            List of data quality metrics
        """
        if dataset_name:
            query = """
            SELECT * FROM data_quality_metrics
            WHERE dataset_name = %s AND timestamp > NOW() - INTERVAL %s DAY
            ORDER BY timestamp;
            """
            self.cur.execute(query, (dataset_name, days))
        else:
            query = """
            SELECT * FROM data_quality_metrics
            WHERE timestamp > NOW() - INTERVAL %s DAY
            ORDER BY timestamp;
            """
            self.cur.execute(query, (days,))
            
        columns = [desc[0] for desc in self.cur.description]
        results = []
        
        for row in self.cur.fetchall():
            result = dict(zip(columns, row))
            # Parse JSONB parameters
            if 'parameters' in result and result['parameters']:
                result['parameters'] = json.loads(result['parameters'])
            results.append(result)
            
        return results

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.cur.close()
            self.conn.close()


class PerformanceDashboard:
    """
    Interactive dashboard for visualizing performance metrics.
    """
    
    def __init__(self, title: str = "Performance Dashboard"):
        """
        Initialize the performance dashboard.
        
        Args:
            title: Dashboard title
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")
            
        if not DASH_AVAILABLE:
            raise ImportError("Dash is required for interactive dashboards")
            
        self.title = title
        self.data = []
        self.app = None

    def add_data(self, data: List[Dict[str, Any]]):
        """
        Add performance data to the dashboard.
        
        Args:
            data: List of performance metrics
        """
        self.data.extend(data)

    def create_dashboard(self, port: int = 8050):
        """
        Create and run the interactive dashboard.
        
        Args:
            port: Port to run the dashboard on
        """
        # Create Dash app
        self.app = dash.Dash(__name__)
        
        # Define layout
        self.app.layout = html.Div([
            html.H1(self.title),
            
            html.Div([
                html.Label("Select Model:"),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': model, 'value': model} 
                            for model in set(d.get('model_name', '') for d in self.data)],
                    value=None,
                    multi=True
                ),
                
                html.Label("Select Metric:"),
                dcc.Dropdown(
                    id='metric-dropdown',
                    options=[
                        {'label': 'Throughput', 'value': 'throughput'},
                        {'label': 'Latency', 'value': 'latency_ms'},
                        {'label': 'Memory Usage', 'value': 'memory_mb'}
                    ],
                    value='throughput'
                ),
                
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=datetime.datetime.now() - datetime.timedelta(days=30),
                    end_date=datetime.datetime.now()
                )
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),
            
            dcc.Graph(id='performance-graph'),
            
            html.Div([
                html.H3("Performance Statistics"),
                html.Div(id='performance-stats')
            ], style={'padding': '20px'})
        ])
        
        # Define callbacks
        @self.app.callback(
            [Output('performance-graph', 'figure'),
             Output('performance-stats', 'children')],
            [Input('model-dropdown', 'value'),
             Input('metric-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_graph(selected_models, selected_metric, start_date, end_date):
            # Filter data based on selections
            filtered_data = self.data
            
            if selected_models:
                if isinstance(selected_models, list):
                    filtered_data = [d for d in filtered_data if d.get('model_name') in selected_models]
                else:
                    filtered_data = [d for d in filtered_data if d.get('model_name') == selected_models]
            
            if start_date and end_date:
                start_date = datetime.datetime.fromisoformat(start_date.split('T')[0])
                end_date = datetime.datetime.fromisoformat(end_date.split('T')[0])
                filtered_data = [d for d in filtered_data if start_date <= d.get('timestamp', datetime.datetime.now()) <= end_date]
            
            # Create figure
            fig = go.Figure()
            
            # Group by model
            model_groups = {}
            for d in filtered_data:
                model = d.get('model_name', 'Unknown')
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append(d)
            
            # Add traces for each model
            for model, data in model_groups.items():
                x = [d.get('timestamp', datetime.datetime.now()) for d in data]
                y = [d.get(selected_metric, 0) for d in data]
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    name=model
                ))
            
            # Update layout
            metric_labels = {
                'throughput': 'Throughput (samples/second)',
                'latency_ms': 'Latency (ms)',
                'memory_mb': 'Memory Usage (MB)'
            }
            
            fig.update_layout(
                title=f"{metric_labels.get(selected_metric, selected_metric)} Over Time",
                xaxis_title="Date",
                yaxis_title=metric_labels.get(selected_metric, selected_metric),
                legend_title="Model",
                hovermode="closest"
            )
            
            # Calculate statistics
            stats_html = []
            for model, data in model_groups.items():
                values = [d.get(selected_metric, 0) for d in data]
                if values:
                    avg = sum(values) / len(values)
                    max_val = max(values)
                    min_val = min(values)
                    
                    stats_html.append(html.Div([
                        html.H4(model),
                        html.P(f"Average: {avg:.2f}"),
                        html.P(f"Maximum: {max_val:.2f}"),
                        html.P(f"Minimum: {min_val:.2f}")
                    ]))
            
            return fig, stats_html
        
        # Run the app
        self.app.run_server(debug=True, port=port)

    def create_static_dashboard(self) -> Dict[str, Any]:
        """
        Create a static dashboard (for embedding in notebooks or reports).
        
        Returns:
            Dictionary with figures
        """
        if not self.data:
            return {"error": "No data available"}
            
        # Create figures
        figures = {}
        
        # Throughput over time
        fig_throughput = go.Figure()
        
        # Group by model
        model_groups = {}
        for d in self.data:
            model = d.get('model_name', 'Unknown')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(d)
        
        # Add traces for each model
        for model, data in model_groups.items():
            x = [d.get('timestamp', datetime.datetime.now()) for d in data]
            y = [d.get('throughput', 0) for d in data]
            
            fig_throughput.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=model
            ))
        
        fig_throughput.update_layout(
            title="Throughput Over Time",
            xaxis_title="Date",
            yaxis_title="Throughput (samples/second)",
            legend_title="Model",
            hovermode="closest"
        )
        
        figures["throughput"] = fig_throughput
        
        # Latency over time
        fig_latency = go.Figure()
        
        # Add traces for each model
        for model, data in model_groups.items():
            x = [d.get('timestamp', datetime.datetime.now()) for d in data]
            y = [d.get('latency_ms', 0) for d in data]
            
            fig_latency.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=model
            ))
        
        fig_latency.update_layout(
            title="Latency Over Time",
            xaxis_title="Date",
            yaxis_title="Latency (ms)",
            legend_title="Model",
            hovermode="closest"
        )
        
        figures["latency"] = fig_latency
        
        # Memory usage over time
        fig_memory = go.Figure()
        
        # Add traces for each model
        for model, data in model_groups.items():
            x = [d.get('timestamp', datetime.datetime.now()) for d in data]
            y = [d.get('memory_mb', 0) for d in data]
            
            fig_memory.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=model
            ))
        
        fig_memory.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Date",
            yaxis_title="Memory Usage (MB)",
            legend_title="Model",
            hovermode="closest"
        )
        
        figures["memory"] = fig_memory
        
        return figures


class DataQualityDashboard:
    """
    Interactive dashboard for visualizing data quality metrics.
    """
    
    def __init__(self, title: str = "Data Quality Dashboard"):
        """
        Initialize the data quality dashboard.
        
        Args:
            title: Dashboard title
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")
            
        if not DASH_AVAILABLE:
            raise ImportError("Dash is required for interactive dashboards")
            
        self.title = title
        self.data = []
        self.app = None

    def add_data(self, data: List[Dict[str, Any]]):
        """
        Add data quality metrics to the dashboard.
        
        Args:
            data: List of data quality metrics
        """
        self.data.extend(data)

    def create_dashboard(self, port: int = 8051):
        """
        Create and run the interactive dashboard.
        
        Args:
            port: Port to run the dashboard on
        """
        # Create Dash app
        self.app = dash.Dash(__name__)
        
        # Define layout
        self.app.layout = html.Div([
            html.H1(self.title),
            
            html.Div([
                html.Label("Select Dataset:"),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=[{'label': dataset, 'value': dataset} 
                            for dataset in set(d.get('dataset_name', '') for d in self.data)],
                    value=None,
                    multi=True
                ),
                
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=datetime.datetime.now() - datetime.timedelta(days=30),
                    end_date=datetime.datetime.now()
                )
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),
            
            dcc.Graph(id='quality-radar-chart'),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='completeness-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='uniqueness-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='consistency-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='accuracy-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
        
        # Define callbacks
        @self.app.callback(
            [Output('quality-radar-chart', 'figure'),
             Output('completeness-chart', 'figure'),
             Output('uniqueness-chart', 'figure'),
             Output('consistency-chart', 'figure'),
             Output('accuracy-chart', 'figure')],
            [Input('dataset-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_graphs(selected_datasets, start_date, end_date):
            # Filter data based on selections
            filtered_data = self.data
            
            if selected_datasets:
                if isinstance(selected_datasets, list):
                    filtered_data = [d for d in filtered_data if d.get('dataset_name') in selected_datasets]
                else:
                    filtered_data = [d for d in filtered_data if d.get('dataset_name') == selected_datasets]
            
            if start_date and end_date:
                start_date = datetime.datetime.fromisoformat(start_date.split('T')[0])
                end_date = datetime.datetime.fromisoformat(end_date.split('T')[0])
                filtered_data = [d for d in filtered_data if start_date <= d.get('timestamp', datetime.datetime.now()) <= end_date]
            
            # Group by dataset
            dataset_groups = {}
            for d in filtered_data:
                dataset = d.get('dataset_name', 'Unknown')
                if dataset not in dataset_groups:
                    dataset_groups[dataset] = []
                dataset_groups[dataset].append(d)
            
            # Create radar chart
            fig_radar = go.Figure()
            
            for dataset, data in dataset_groups.items():
                # Use the most recent data point for the radar chart
                if data:
                    latest = max(data, key=lambda x: x.get('timestamp', datetime.datetime.now()))
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[
                            latest.get('completeness', 0),
                            latest.get('uniqueness', 0),
                            latest.get('consistency', 0),
                            latest.get('accuracy', 0)
                        ],
                        theta=['Completeness', 'Uniqueness', 'Consistency', 'Accuracy'],
                        fill='toself',
                        name=dataset
                    ))
            
            fig_radar.update_layout(
                title="Data Quality Metrics",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                )
            )
            
            # Create time series charts
            fig_completeness = go.Figure()
            fig_uniqueness = go.Figure()
            fig_consistency = go.Figure()
            fig_accuracy = go.Figure()
            
            for dataset, data in dataset_groups.items():
                x = [d.get('timestamp', datetime.datetime.now()) for d in data]
                
                fig_completeness.add_trace(go.Scatter(
                    x=x,
                    y=[d.get('completeness', 0) for d in data],
                    mode='lines+markers',
                    name=dataset
                ))
                
                fig_uniqueness.add_trace(go.Scatter(
                    x=x,
                    y=[d.get('uniqueness', 0) for d in data],
                    mode='lines+markers',
                    name=dataset
                ))
                
                fig_consistency.add_trace(go.Scatter(
                    x=x,
                    y=[d.get('consistency', 0) for d in data],
                    mode='lines+markers',
                    name=dataset
                ))
                
                fig_accuracy.add_trace(go.Scatter(
                    x=x,
                    y=[d.get('accuracy', 0) for d in data],
                    mode='lines+markers',
                    name=dataset
                ))
            
            fig_completeness.update_layout(
                title="Completeness Over Time",
                xaxis_title="Date",
                yaxis_title="Completeness",
                yaxis=dict(range=[0, 1])
            )
            
            fig_uniqueness.update_layout(
                title="Uniqueness Over Time",
                xaxis_title="Date",
                yaxis_title="Uniqueness",
                yaxis=dict(range=[0, 1])
            )
            
            fig_consistency.update_layout(
                title="Consistency Over Time",
                xaxis_title="Date",
                yaxis_title="Consistency",
                yaxis=dict(range=[0, 1])
            )
            
            fig_accuracy.update_layout(
                title="Accuracy Over Time",
                xaxis_title="Date",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1])
            )
            
            return fig_radar, fig_completeness, fig_uniqueness, fig_consistency, fig_accuracy
        
        # Run the app
        self.app.run_server(debug=True, port=port)

    def create_static_dashboard(self) -> Dict[str, Any]:
        """
        Create a static dashboard (for embedding in notebooks or reports).
        
        Returns:
            Dictionary with figures
        """
        if not self.data:
            return {"error": "No data available"}
            
        # Create figures
        figures = {}
        
        # Group by dataset
        dataset_groups = {}
        for d in self.data:
            dataset = d.get('dataset_name', 'Unknown')
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(d)
        
        # Create radar chart
        fig_radar = go.Figure()
        
        for dataset, data in dataset_groups.items():
            # Use the most recent data point for the radar chart
            if data:
                latest = max(data, key=lambda x: x.get('timestamp', datetime.datetime.now()))
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=[
                        latest.get('completeness', 0),
                        latest.get('uniqueness', 0),
                        latest.get('consistency', 0),
                        latest.get('accuracy', 0)
                    ],
                    theta=['Completeness', 'Uniqueness', 'Consistency', 'Accuracy'],
                    fill='toself',
                    name=dataset
                ))
        
        fig_radar.update_layout(
            title="Data Quality Metrics",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            )
        )
        
        figures["radar"] = fig_radar
        
        # Create time series chart for all metrics
        fig_time_series = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Completeness", "Uniqueness", "Consistency", "Accuracy")
        )
        
        for dataset, data in dataset_groups.items():
            x = [d.get('timestamp', datetime.datetime.now()) for d in data]
            
            fig_time_series.add_trace(
                go.Scatter(
                    x=x,
                    y=[d.get('completeness', 0) for d in data],
                    mode='lines+markers',
                    name=f"{dataset} - Completeness"
                ),
                row=1, col=1
            )
            
            fig_time_series.add_trace(
                go.Scatter(
                    x=x,
                    y=[d.get('uniqueness', 0) for d in data],
                    mode='lines+markers',
                    name=f"{dataset} - Uniqueness"
                ),
                row=1, col=2
            )
            
            fig_time_series.add_trace(
                go.Scatter(
                    x=x,
                    y=[d.get('consistency', 0) for d in data],
                    mode='lines+markers',
                    name=f"{dataset} - Consistency"
                ),
                row=2, col=1
            )
            
            fig_time_series.add_trace(
                go.Scatter(
                    x=x,
                    y=[d.get('accuracy', 0) for d in data],
                    mode='lines+markers',
                    name=f"{dataset} - Accuracy"
                ),
                row=2, col=2
            )
        
        fig_time_series.update_layout(
            title="Data Quality Metrics Over Time",
            height=800
        )
        
        figures["time_series"] = fig_time_series
        
        return figures
