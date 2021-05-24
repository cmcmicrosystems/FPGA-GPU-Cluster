# Check Pytorch and GPU availability
import sys
import torch
import pandas as pd
import sklearn as sk

print(f"- PyTorch: {torch.__version__}")
print(f"- Python {sys.version}")
print(f"- Pandas {pd.__version__}")
print(f"- Scikit-Learn {sk.__version__}")
print("- GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")
