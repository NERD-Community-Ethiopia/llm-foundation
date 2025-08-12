# Checkpoint 0: Environment Setup

## Objectives
- Set up the development environment
- Initialize the project structure
- Install all required dependencies
- Set up version control with Git branches

## Tasks

### 1. Environment Setup
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Initialize Poetry project and install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### 2. Git Branch Structure
```bash
# Initialize Git repository (if not already done)
git init

# Create and switch to development branches for each checkpoint
git checkout -b checkpoint-1-env-setup
git checkout -b checkpoint-1-neural-nets
git checkout -b checkpoint-2-rnn-lstm
git checkout -b checkpoint-3-attention
git checkout -b checkpoint-4-transformer-core
git checkout -b checkpoint-5-complete-transformer
git checkout -b checkpoint-6-data-preparation
git checkout -b checkpoint-7-training-infrastructure
git checkout -b checkpoint-8-model-trainning
git checkout -b checkpoint-9-fastapi-backend
git checkout -b checkpoint-10-gradio-frontend
git checkout -b checkpoint-11-deployment

# Return to main branch
git checkout main
```

### 3. Project Structure
```
llm-foundation/
├── src/                          # Source code directory
│   ├── neural_nets/             # Neural Networks & Backpropagation
│   ├── rnn_lstm/                # RNN & LSTM implementations
│   ├── attention/               # Attention mechanisms
│   ├── transformer/             # Transformer architecture
│   ├── data/                    # Data preprocessing
│   ├── training/                # Training loops
│   ├── models/                  # Model implementations
│   ├── api/                     # FastAPI web service
│   ├── interface/               # Gradio interface
│   └── utils/                   # Utility functions
├── data/                        # Dataset files
├── models/                      # Saved model files
├── tests/                       # Test files
├── pyproject.toml              # Poetry configuration
├── requirements.txt            # Legacy pip requirements (optional)
└── README.md                   # This file
```

### 4. Development Workflow
```bash
# Switch to checkpoint branch
git checkout checkpoint-1-neural-nets

# Make changes and commit
git add .
git commit -m "Implement neural network basics"

# Push branch to remote (if using remote repository)
git push origin checkpoint-1-neural-nets

# Merge back to main when checkpoint is complete
git checkout main
git merge checkpoint-1-neural-nets
```

### 5. Development Commands
```bash
# Run tests
poetry run pytest

# Format code
poetry run black .

# Lint code
poetry run flake8

# Add new dependency
poetry add package_name

# Add development dependency
poetry add --group dev package_name
```

### 6. Verification
Run the verification script to ensure everything is set up correctly:
```bash
poetry run python src/utils/verify_setup.py
```

## Next Steps
Once this checkpoint is complete, proceed to Checkpoint 1: Neural Networks & Backpropagation.

```python:src/utils/verify_setup.py
#!/usr/bin/env python3
"""
Verification script for Checkpoint 0 setup
"""

import sys
import importlib
import os

def check_package(package_name):
    """Check if a package is installed and importable"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        'src',
        'src/neural_nets',
        'src/rnn_lstm', 
        'src/attention',
        'src/transformer',
        'src/data',
        'src/training',
        'src/models',
        'src/api',
        'src/interface',
        'src/utils',
        'data',
        'models',
        'tests'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory {dir_name} exists")
        else:
            print(f"❌ Directory {dir_name} is missing")
            all_exist = False
    
    return all_exist

def main():
    print("🔍 Verifying Checkpoint 0 Setup...\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check required packages
    print("\n📦 Checking required packages:")
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 
        'fastapi', 'uvicorn', 'gradio', 'transformers',
        'tokenizers', 'datasets', 'tqdm', 'wandb'
    ]
    
    all_packages_installed = True
    for package in required_packages:
        if not check_package(package):
            all_packages_installed = False
    
    # Check directory structure
    print("\n📁 Checking directory structure:")
    dirs_exist = check_directory_structure()
    
    # Summary
    print("\n" + "="*50)
    if all_packages_installed and dirs_exist:
        print("✅ Setup verification PASSED!")
        print("You can proceed to Checkpoint 1")
    else:
        print("❌ Setup verification FAILED!")
        print("Please fix the issues above before proceeding")
    
    return all_packages_installed and dirs_exist

if __name__ == "__main__":
    main()
