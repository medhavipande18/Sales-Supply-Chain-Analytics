import subprocess
import sys

steps = [
    ("Generate synthetic data", ["python", "src/generate_data.py"]),
    ("Descriptive analytics", ["python", "src/descriptive.py"]),
    ("Diagnostic analytics", ["python", "src/diagnostic.py"]),
    ("Predictive analytics", ["python", "src/predictive.py"]),
    ("Prescriptive analytics", ["python", "src/prescriptive.py"]),
]

def main():
    for name, cmd in steps:
        print(f"\n=== {name} ===")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"Failed step: {name}")
            sys.exit(r.returncode)
    print("\nAll steps completed. Check reports/")

if __name__ == "__main__":
    main()