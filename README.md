# Machine Learning Lab

A collection of simple, exam-oriented Python programs for the Machine Learning Lab. This repository includes experiments on regression, classification, neural networks, model evaluation, and clustering.

The programs are intended for learning and lab-exam practice. Read the experiment description and dataset requirements before running each script.

## Experiments at a Glance

| No. | Experiment | Dataset |
|---:|---|---|
| 1 | Simple Linear Regression | `housing.csv` |
| 3 | Ridge and Lasso Regression | Diabetes dataset (built into scikit-learn) |
| 5 | Logistic Regression using MLE and MAP (L1 and L2) | Breast Cancer dataset (built into scikit-learn) |
| 6 | Text Classification using Multinomial and Bernoulli Naive Bayes | 20 Newsgroups dataset |
| 7 | K-Nearest Neighbors (KNN) Classification | Fashion-MNIST |
| 9 | Linear SVM Classification with Decision Boundary Visualization | Iris dataset (built into scikit-learn) |
---

## 1. Clone the Repository

```bash
git clone https://github.com/MabelMoncy/ml_lab.git
cd ml_lab
```

## 2. Create and Activate a Virtual Environment

Using a virtual environment keeps the project's Python packages separate from other projects.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows — Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate
```

### Windows — PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

## 3. Install Required Packages

With the virtual environment activated, install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn seaborn tensorflow openpyxl
```

- `numpy`, `pandas`, `matplotlib`, and `scikit-learn` are used across the experiments.
- `seaborn` is used to load the MPG dataset in Experiment 2.
- `tensorflow` provides Keras and the Fashion-MNIST/MNIST datasets used in Experiments 7, 10, and 12.
- `openpyxl` is needed by pandas to read the Excel file in Experiment 8.

TensorFlow installation and model training may take some time. If TensorFlow reports that CUDA/GPU libraries are unavailable, the program can still run using the CPU, although it may take longer.

## 4. Dataset Requirements

Download the following files and place them in the repository's main folder (`ml_lab`), alongside the Python scripts:

| Dataset file | Required for |
|---|---|
| `housing.csv` | Experiment 1 |

**Important:** Keep the filenames exactly as shown, including spaces and capital letters. The scripts expect these files in the project root.

### Datasets that do not need manual file placement

- **Experiment 3:** Uses scikit-learn's built-in Diabetes dataset.
- **Experiment 5:** Uses scikit-learn's built-in Breast Cancer dataset.
- **Experiment 6:** Fetches the 20 Newsgroups dataset through scikit-learn. An internet connection is required for the initial download.
- **Experiment 7:** Keras downloads Fashion-MNIST the first time it is used.
- **Experiment 9:** Uses scikit-learn's built-in Iris dataset.

Built-in datasets do not need to be added to the repository as separate CSV files. Some libraries may download and cache datasets automatically on the first run.

## 5. Run the Experiments

Activate the virtual environment and run commands from the repository's root directory.

| Experiment | Command |
|---:|---|
| 1 | `python exp01.py` |
| 3 | `python exp03.py` |
| 5 | `python exp05.py` |
| 6 | `python exp06.py` |
| 7 | `python exp07.py` |
| 9 | `python exp09.py` |

For example, to run Experiment 14:

```bash
python exp14.py
```

## 6. Troubleshooting

**A required Python module is missing**

Make sure the virtual environment is activated, then install the missing package:

```bash
python -m pip install package-name
```

**A dataset file is not found**

Check that the required CSV or Excel file is in the `ml_lab` root folder and that its filename matches the spelling in the table above.

**A dataset download fails**

Check your internet connection. Some datasets are downloaded automatically by Seaborn, scikit-learn, or Keras the first time the corresponding experiment runs.

**TensorFlow shows CUDA or GPU warnings**

If TensorFlow cannot find CUDA/GPU libraries, it may use the CPU instead. This does not necessarily mean the program has failed. Allow extra time for training.

**All the best for your Machine Learning Lab Exam!**

Prepare well, stay confident, and do your best. Good luck, everyone!
