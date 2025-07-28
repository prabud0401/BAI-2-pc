import sys
import os

print(f"--- Python Environment ---")
print(f"Executable: {sys.executable}")
print(f"Version: {sys.version}")
print("-" * 20)

try:
    print("\n--- PyTorch Check ---")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch is not installed.")
except Exception as e:
    print(f"An error occurred with PyTorch: {e}")

print("-" * 20)

try:
    print("\n--- CuPy Check ---")
    import cupy
    print(f"CuPy version: {cupy.__version__}")
    print(f"CuPy found at: {cupy.__file__}")
    # Use getDriverVersion to be safe across versions
    driver_version = cupy.cuda.runtime.getDriverVersion()
    print(f"CuPy using CUDA Driver Version: {driver_version}")
    device_count = cupy.cuda.runtime.getDeviceCount()
    print(f"CuPy can see {device_count} GPU(s).")
    if device_count > 0:
        cupy.cuda.Device(0).synchronize()
        print("Successfully synchronized with GPU 0 via CuPy.")
except ImportError:
    print("CuPy is NOT installed or visible to this Python environment.")
except Exception as e:
    print(f"An error occurred with CuPy: {e}")

print("-" * 20)

try:
    print("\n--- spaCy GPU Check ---")
    import spacy
    print(f"spaCy version: {spacy.__version__}")
    # Use spacy.prefer_gpu() for a check, require_gpu() can exit
    is_gpu_available = spacy.prefer_gpu()
    print(f"spaCy GPU preference check: {'GPU available' if is_gpu_available else 'GPU NOT available'}")
    if is_gpu_available:
        print("Attempting to load a model to GPU...")
        nlp = spacy.load("en_core_web_sm", exclude=["parser", "tagger", "lemmatizer"])
        print("Model loaded to GPU successfully.")
    else:
        # Re-check with require_gpu to get the specific error
        try:
            spacy.require_gpu()
        except Exception as e:
            print(f"spaCy require_gpu() failed with error: {e}")

except ImportError:
    print("spaCy is not installed.")
except Exception as e:
    print(f"An error occurred with spaCy: {e}") 