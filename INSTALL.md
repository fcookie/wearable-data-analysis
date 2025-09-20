# Installation and Setup Guide

This guide helps you set up the wearable data analysis pipeline correctly.

## 🚀 Quick Start (Recommended)

### Option 1: Use the Run Script (Simplest)

```bash
# 1. Clone the repository
git clone https://github.com/fcookie/wearable-data-analysis.git
cd wearable-data-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis using the helper script
python run_analysis.py --generate-data
python run_analysis.py --use-uci-har
python run_analysis.py --help  # See all options
```

### Option 2: Install as Package

```bash
# 1. Clone and install
git clone https://github.com/fcookie/wearable-data-analysis.git
cd wearable-data-analysis
pip install -e .

# 2. Run from anywhere
python main.py --generate-data
```

### Option 3: Run Main Script Directly

```bash
# Make sure you're in the project root directory
cd wearable-data-analysis

# Run main script
python main.py --generate-data
```

## 🐍 Python Environment Setup

### Using Conda (Recommended)

```bash
# Create new environment
conda create -n wearable-analysis python=3.9
conda activate wearable-analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python run_analysis.py --generate-data
```

### Using venv

```bash
# Create virtual environment
python -m venv wearable-env

# Activate (Windows)
wearable-env\Scripts\activate

# Activate (macOS/Linux)
source wearable-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run analysis
python run_analysis.py --generate-data
```

## 🔧 Troubleshooting Common Issues

### Import Errors

If you get import errors like `ImportError: attempted relative import with no known parent package`:

**Solution 1: Use run_analysis.py**
```bash
python run_analysis.py --generate-data
```

**Solution 2: Set PYTHONPATH**
```bash
# Windows
set PYTHONPATH=%cd%\src;%PYTHONPATH%
python main.py --generate-data

# macOS/Linux
export PYTHONPATH=$PWD/src:$PYTHONPATH
python main.py --generate-data
```

**Solution 3: Install as editable package**
```bash
pip install -e .
python main.py --generate-data
```

### Missing Dependencies

If you get `ModuleNotFoundError`:

```bash
# Install missing packages
pip install requests tqdm joblib

# Or reinstall all requirements
pip install -r requirements.txt --upgrade
```

### Permission Errors (Windows)

If you get permission errors on Windows:

```bash
# Run as administrator or use:
pip install --user -r requirements.txt
```

## 📁 Project Structure Verification

After cloning, your directory should look like this:

```
wearable-data-analysis/
├── main.py                    # Main pipeline script
├── run_analysis.py            # Helper script (use this if imports fail)
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
├── README.md                  # Main documentation
├── INSTALL.md                 # This file
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineer.py
│   ├── model.py
│   ├── evaluator.py
│   ├── visualizer.py
│   └── load_uci_har.py
├── tests/
│   └── test_model.py
├── data/                      # Will be created
├── results/                   # Will be created
└── notebooks/
    └── uci_har_analysis.ipynb
```

## 🧪 Verify Installation

Test your installation:

```bash
# Test 1: Run with synthetic data
python run_analysis.py --generate-data --n-samples 1000 --no-plots

# Test 2: Run tests
python -m pytest tests/ -v

# Test 3: Test UCI HAR download
python run_analysis.py --use-uci-har --no-plots
```

## 📊 Usage Examples

### Basic Usage

```bash
# Generate synthetic data and analyze
python run_analysis.py --generate-data

# Use UCI HAR dataset
python run_analysis.py --use-uci-har

# Analyze your own data
python run_analysis.py --data-file your_data.csv
```

### Advanced Usage

```bash
# Custom parameters
python run_analysis.py --generate-data \
    --n-samples 20000 \
    --window-size 50 \
    --overlap 0.3 \
    --models "Random Forest" "SVM"

# Save results
python run_analysis.py --use-uci-har \
    --save-model models/best_model.joblib \
    --save-features data/processed/features.csv \
    --save-report results/report.txt
```

### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/uci_har_analysis.ipynb
# Run all cells
```

## 💡 Tips

1. **Always use `run_analysis.py`** if you encounter import issues
2. **Use virtual environments** to avoid dependency conflicts
3. **Check Python version** (requires Python 3.8+)
4. **Run from project root** directory
5. **Install in editable mode** (`pip install -e .`) for development

## ❓ Getting Help

If you still have issues:

1. Check that you're in the project root directory
2. Verify Python version: `python --version` (should be 3.8+)
3. Try the run_analysis.py script
4. Create a new virtual environment
5. Open an issue on GitHub with your error message

## 🎯 Quick Test Commands

```bash
# Minimal test (fastest)
python run_analysis.py --generate-data --n-samples 500 --no-plots --models "Random Forest"

# Full test (comprehensive)
python run_analysis.py --use-uci-har --models "Random Forest" "Logistic Regression"

# Custom data test
python run_analysis.py --data-file your_data.csv --window-size 25
```

Choose the method that works best for your setup! 🚀