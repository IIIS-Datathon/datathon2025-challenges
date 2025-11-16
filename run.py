import subprocess
import sys

def run_script(script_name):
    """Executes a python script and handles errors."""
    try:
        print(f"--- Running {script_name} ---")
        # Using sys.executable ensures we use the same python interpreter
        result = subprocess.run(["python", script_name], cwd="./core", check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("--- Stderr ---")
            print(result.stderr)
        print(f"--- Successfully finished {script_name} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR running {script_name} ---")
        print(e.stdout)
        print(e.stderr)
        sys.exit(f"Script {script_name} failed. Exiting.")

if __name__ == "__main__":
    print("Starting the model training and SHAP generation pipeline...")
    run_script("trainRF.py")
    run_script("generateShap.py")
    print("Pipeline completed successfully!")