# algocomplex/examples/example_usage.py

from algocomplex import Analyzer

sample_code = r'''
def example_linear(n):
    s = 0
    for i in range(n):
        s += i
    return s

def example_quadratic(n):
    cnt = 0
    for i in range(n):
        for j in range(n):
            cnt += (i+j)
    return cnt

def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
'''

def main():
    analyzer = Analyzer(gui=False, html=True, plot=False)
    html_report = analyzer.analyze_code(sample_code)
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_report)
    print("HTML report generated as report.html")

if __name__ == '__main__':
    main()
