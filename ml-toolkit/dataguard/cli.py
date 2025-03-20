import click
import os
import pandas as pd
import time
from pathlib import Path
import json

from ml_toolkit.dataguard.data_quality import DataQualityChecker
from ml_toolkit.dataguard.preprocessing import DataPreprocessor
from ml_toolkit.dataguard.visualization import create_quality_dashboard
from ml_toolkit.dataguard.reporters.html_reporter import generate_html_report
from ml_toolkit.dataguard.reporters.pdf_reporter import generate_pdf_report

@click.group()
def cli():
    """Data quality and preprocessing toolkit."""
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', type=click.Path(), default='./reports', 
              help='Directory to save reports (default: ./reports)')
@click.option('--target', type=str, default=None, 
              help='Target column for supervised learning tasks')
@click.option('--sensitive', type=str, multiple=True, 
              help='Sensitive attributes for bias/privacy analysis (can specify multiple)')
@click.option('--format', type=click.Choice(['html', 'pdf', 'json']), default='html',
              help='Report format (default: html)')
@click.option('--interactive/--no-interactive', default=True, 
              help='Create interactive visualization dashboard (default: True)')
def analyze(input_file, output_dir, target, sensitive, format, interactive):
    """Analyze data quality and generate comprehensive report."""
    click.echo(f"Analyzing data quality for {input_file}...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    start_time = time.time()
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(input_file)
    elif file_ext == '.parquet':
        df = pd.read_parquet(input_file)
    else:
        click.echo(f"Unsupported file format: {file_ext}", err=True)
        return
    
    # Run analysis
    click.echo(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    data_checker = DataQualityChecker(
        df=df,
        target_column=target,
        protected_attributes=list(sensitive) if sensitive else None
    )
    
    # Run comprehensive analysis
    results = data_checker.run_comprehensive_analysis()
    
    # Generate report
    output_path = output_dir / f"data_quality_report_{time.strftime('%Y%m%d_%H%M%S')}"
    
    if format == 'html':
        report_path = generate_html_report(results, output_path.with_suffix('.html'))
        click.echo(f"HTML report generated: {report_path}")
    elif format == 'pdf':
        report_path = generate_pdf_report(results, output_path.with_suffix('.pdf'))
        click.echo(f"PDF report generated: {report_path}")
    elif format == 'json':
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"JSON report generated: {output_path.with_suffix('.json')}")
    
    # Create interactive dashboard
    if interactive:
        dashboard_path = create_quality_dashboard(results, output_dir)
        click.echo(f"Interactive dashboard created: {dashboard_path}")
        click.echo("Run 'python -m http.server' in the output directory to view it")
    
    elapsed_time = time.time() - start_time
    click.echo(f"Analysis completed in {elapsed_time:.2f} seconds")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--config', type=click.Path(exists=True), 
              help='JSON config file with preprocessing steps')
@click.option('--auto/--no-auto', default=True, 
              help='Automatically detect and fix issues (default: True)')
@click.option('--target', type=str, default=None, 
              help='Target column to preserve during preprocessing')
@click.option('--log-file', type=click.Path(), default=None,
              help='Log file to record preprocessing steps')
def preprocess(input_file, output_file, config, auto, target, log_file):
    """Preprocess data and fix quality issues."""
    click.echo(f"Preprocessing {input_file}...")
    
    # Load data
    start_time = time.time()
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(input_file)
    elif file_ext == '.parquet':
        df = pd.read_parquet(input_file)
    else:
        click.echo(f"Unsupported file format: {file_ext}", err=True)
        return
    
    click.echo(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Load config if provided
    preprocessing_config = None
    if config:
        with open(config, 'r') as f:
            preprocessing_config = json.load(f)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        df=df,
        target_column=target,
        auto_fix=auto,
        config=preprocessing_config
    )
    
    # Run preprocessing
    processed_df = preprocessor.run_preprocessing_pipeline()
    
    # Save preprocessing logs if requested
    if log_file:
        preprocessor.save_preprocessing_log(log_file)
    
    # Save processed data
    output_ext = os.path.splitext(output_file)[1].lower()
    if output_ext == '.csv':
        processed_df.to_csv(output_file, index=False)
    elif output_ext in ['.xls', '.xlsx']:
        processed_df.to_excel(output_file, index=False)
    elif output_ext == '.parquet':
        processed_df.to_parquet(output_file, index=False)
    else:
        click.echo(f"Unsupported output format: {output_ext}. Saving as CSV instead.")
        processed_df.to_csv(output_file + '.csv', index=False)
    
    # Output summary
    elapsed_time = time.time() - start_time
    rows_diff = len(df) - len(processed_df)
    cols_diff = len(df.columns) - len(processed_df.columns)
    
    click.echo(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    click.echo(f"Output data dimensions: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
    click.echo(f"Rows removed: {rows_diff}, Columns modified/removed: {cols_diff}")
    click.echo(f"Processed data saved to: {output_file}")

if __name__ == '__main__':
    cli()