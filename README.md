# Machine Learning Lab

## Introduction
This repository contains Python implementations of experiments for the Machine Learning Lab Exam. The experiments cover core machine learning concepts including Linear Regression, Regularization (Ridge & Lasso), Naive Bayes Classification, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).

---

## Cloning Setup

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/MabelMoncy/ml_lab.git
cd ml_lab
```

---

## Virtual Environment Setup and Activation

### Linux / macOS
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Windows
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment (Command Prompt)
venv\Scripts\activate

# Or in PowerShell:
# venv\Scripts\Activate.ps1
```

---

## Pip Installing Dependencies

Install the required packages using pip:

```bash
python -m pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## Running Each Experiment

Ensure your virtual environment is activated before running the scripts:

### Experiment 1: Simple Linear Regression
Implementation of simple linear regression from scratch using the housing dataset.
```bash
python exp1.py
```

### Experiment 3: Linear, Ridge, and Lasso Regression
Comparison of Linear, Ridge, and Lasso regression models on the Diabetes dataset.
```bash
python exp3.py
```

### Experiment 6: Naive Bayes Classification
Multinomial and Bernoulli Naive Bayes classification on the 20 Newsgroups dataset.
```bash
python exp6.py
```

### Experiment 7: K-Nearest Neighbors (KNN)
KNN image classification on the Fashion MNIST dataset using varying $K$ values.
```bash
python exp7.py
```

### Experiment 9: Support Vector Machine (SVM)
Linear SVM classification and decision boundary visualization on the Iris dataset.
```bash
python exp9.py
```
