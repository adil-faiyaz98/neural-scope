# aiml_complexity/examples/example_usage.py

import os
from aiml_complexity import ComplexityAnalyzer, CostAnalyzerAWS, AnalysisReportStorage

def dummy_knn(data):
    # naive kNN: nested loop to compute distances
    best_dist = float('inf')
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                dist = abs(data[i] - data[j])  # trivial distance
                if dist < best_dist:
                    best_dist = dist
    return best_dist

def main():
    # 1) Analyze a function
    analyzer = ComplexityAnalyzer(analyze_runtime=True)
    report_func = analyzer.analyze_function(dummy_knn)
    print("Function Analysis Report:", report_func)

    # 2) Analyze a Python file
    script_path = os.path.join(os.path.dirname(__file__), "test_script.py")
    if os.path.exists(script_path):
        report_file = analyzer.analyze_file(script_path)
        print("File Analysis Report:", report_file)
    else:
        print("No test_script.py found to analyze.")

    # 3) Optional AWS Cost Analysis
    cost_analyzer = CostAnalyzerAWS()  # uses default credentials if available
    cost_report = cost_analyzer.analyze_spend(days_back=3)
    print("AWS Cost Report:", cost_report)

    # 4) Store results in JSON
    storage = AnalysisReportStorage(db_type="json", db_path="analysis_data.json")
    storage.save_report(report_func)
    storage.save_report({"aws_cost": cost_report})
    print("Reports stored. Current stored reports:")
    all_reports = storage.get_all_reports()
    for r in all_reports:
        print(r)

if __name__ == "__main__":
    main()
