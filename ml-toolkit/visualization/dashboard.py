from flask import Flask, render_template
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data_quality')
def data_quality():
    # Load data quality metrics from a JSON file or other source
    with open(os.path.join('data_quality_metrics.json')) as f:
        metrics = json.load(f)
    return render_template('data_quality.html', metrics=metrics)

@app.route('/performance_metrics')
def performance_metrics():
    # Load performance metrics from a JSON file or other source
    with open(os.path.join('performance_metrics.json')) as f:
        metrics = json.load(f)
    return render_template('performance_metrics.html', metrics=metrics)

@app.route('/model_training')
def model_training():
    # Load model training results from a JSON file or other source
    with open(os.path.join('model_training_results.json')) as f:
        results = json.load(f)
    return render_template('model_training.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)