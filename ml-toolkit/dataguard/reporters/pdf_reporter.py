from fpdf import FPDF

class PDFReporter:
    def __init__(self, title="Data Quality Report"):
        self.title = title
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 10, self.title, ln=True, align='C')
        self.pdf.ln(10)

    def add_section(self, title, content):
        self.pdf.set_font("Arial", 'B', 14)
        self.pdf.cell(0, 10, title, ln=True)
        self.pdf.set_font("Arial", '', 12)
        self.pdf.multi_cell(0, 10, content)
        self.pdf.ln(5)

    def save(self, filename):
        self.pdf.output(filename)

    def generate_report(self, data_quality_results):
        self.add_section("Data Quality Summary", data_quality_results['summary'])
        self.add_section("Missing Values", data_quality_results['missing_values'])
        self.add_section("Duplicate Entries", data_quality_results['duplicate_entries'])
        self.add_section("Outliers", data_quality_results['outliers'])
        self.add_section("Feature Scaling", data_quality_results['feature_scaling'])
        self.add_section("Feature Encoding", data_quality_results['feature_encoding'])
        self.add_section("Class Imbalance", data_quality_results['class_imbalance'])
        self.add_section("Bias and Fairness", data_quality_results['bias_fairness'])