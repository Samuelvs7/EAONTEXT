"""Environment check script - checks all required packages."""
import importlib
import sys

packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'sklearn',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'nltk': 'nltk',
    'transformers': 'transformers',
    'torch': 'torch',
    'joblib': 'joblib',
    'pyyaml': 'yaml',
    'tqdm': 'tqdm',
}

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print()

missing = []
installed = []
for name, module in packages.items():
    try:
        m = importlib.import_module(module)
        ver = getattr(m, '__version__', 'installed')
        print(f"  {name:20s} -> {ver}")
        installed.append(name)
    except ImportError:
        print(f"  {name:20s} -> MISSING")
        missing.append(name)

print(f"\nInstalled: {len(installed)}")
print(f"Missing:   {len(missing)}")
if missing:
    print(f"Missing packages: {missing}")
