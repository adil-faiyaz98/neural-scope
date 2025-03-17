# algocomplex/output.py

import matplotlib.pyplot as plt
import io
import base64
from PyQt5 import QtWidgets, QtCore
import sys

def generate_text_report(static_analysis, dynamic_analysis, memory_analysis):
    """
    Return a simple textual report summarizing results.
    """
    lines = []
    lines.append("=== Complexity Analysis Report ===\n")
    # static
    lines.append("** Static Analysis **\n")
    for func, result in static_analysis.items():
        lines.append(f"Function: {func}\n")
        time_complexities = ', '.join(result['time'])
        space_complexities = ', '.join(result['space'])
        lines.append(f"  Theoretical Time: {time_complexities}\n")
        lines.append(f"  Theoretical Space: {space_complexities}\n")
        lines.append(f"  Details: {result['detail']}\n")
        lines.append("")

    # dynamic
    lines.append("** Dynamic Analysis **\n")
    for func, classification in dynamic_analysis.items():
        lines.append(f"Function: {func}\n")
        lines.append(f"  Empirical Time Class: {classification['time_class']} (reason: {classification['time_details']})\n")

    # memory
    lines.append("\n** Memory Analysis **\n")
    for func, memdata in memory_analysis.items():
        lines.append(f"Function: {func}\n")
        lines.append(f"  Memory Usage: {memdata}\n")
    return "\n".join(lines)


def generate_html_report(static_analysis, dynamic_analysis, memory_analysis):
    """
    Return HTML-formatted report with potential inline base64 plots.
    """
    html_parts = []
    html_parts.append("<html><head><title>Complexity Analysis Report</title></head><body>")
    html_parts.append("<h1>Complexity Analysis Report</h1>")

    # static
    html_parts.append("<h2>Static Analysis</h2>")
    for func, result in static_analysis.items():
        time_complexities = ', '.join(result['time'])
        space_complexities = ', '.join(result['space'])
        html_parts.append(f"<h3>Function: {func}</h3>")
        html_parts.append(f"<p><strong>Theoretical Time:</strong> {time_complexities}</p>")
        html_parts.append(f"<p><strong>Theoretical Space:</strong> {space_complexities}</p>")
        html_parts.append(f"<pre>{result['detail']}</pre>")

    # dynamic
    html_parts.append("<h2>Dynamic Analysis</h2>")
    for func, classification in dynamic_analysis.items():
        html_parts.append(f"<h3>Function: {func}</h3>")
        html_parts.append(f"<p><strong>Empirical Time Class:</strong> {classification['time_class']}")
        html_parts.append(f" (reason: {classification['time_details']})</p>")

    # memory
    html_parts.append("<h2>Memory Analysis</h2>")
    for func, memdata in memory_analysis.items():
        html_parts.append(f"<h3>Function: {func}</h3>")
        html_parts.append(f"<p><strong>Memory Usage:</strong> {memdata}</p>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


class AnalysisGUI(QtWidgets.QMainWindow):
    """
    Simple PyQt5-based GUI that displays analysis results.
    """
    def __init__(self, static_analysis, dynamic_analysis, memory_analysis):
        super().__init__()
        self.setWindowTitle("Complexity Analysis")
        self.resize(800, 600)

        main_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        text_area = QtWidgets.QTextEdit()
        text_report = generate_text_report(static_analysis, dynamic_analysis, memory_analysis)
        text_area.setText(text_report)
        layout.addWidget(text_area)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

def show_gui_report(static_analysis, dynamic_analysis, memory_analysis):
    """
    Show PyQt GUI with analysis results.
    """
    app = QtWidgets.QApplication(sys.argv)
    gui = AnalysisGUI(static_analysis, dynamic_analysis, memory_analysis)
    gui.show()
    sys.exit(app.exec_())


def plot_runtime_graph(time_data):
    """
    Generate a plot from time_data: {size -> [list_of_times]}
    Return base64 PNG or show directly.
    """
    for func, data in time_data.items():
        sizes = sorted(data.keys())
        avg_times = [sum(data[s]) / len(data[s]) for s in sizes]
        plt.plot(sizes, avg_times, marker='o', label=func)

    plt.xlabel("Input Size (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Runtime Performance")
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    return img_base64
