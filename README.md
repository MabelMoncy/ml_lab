# Machine Learning Lab Experiments

Python programs for selected Machine Learning lab experiments.

## Experiments

| Experiment | Topic | Main output |
|---|---|---|
| 1 | Simple Linear Regression | Slope, intercept, MSE, R², scatter plot and regression line |
| 2 (optional) | Linear vs Polynomial Regression | MSE, R² and regression curves; requires `seaborn` |
| 3 | Linear, Ridge and Lasso Regression | MSE, R², prediction comparison, coefficient comparison and best alpha values |
| 6 | Multinomial vs Bernoulli Naïve Bayes | Accuracy, macro F1-score and classification reports |
| 7 | KNN on Fashion MNIST | Accuracy and classification report for K = 1, 3 and 5 |
| 9 | Linear SVM on Iris | Decision boundary, margins and support vectors |

## Requirements

- Python 3.10 or newer recommended
- Git
- Internet for cloning and first-time downloads of 20 Newsgroups and Fashion MNIST
- Experiment 1 requires the `housing.csv` dataset
- Experiment 4, if used, requires `diabetes.csv`

## Clone the repository

### Windows (PowerShell or Command Prompt)

```powershell
git clone https://github.com/MabelMoncy/ml_lab.git
cd ml_lab
python -m venv .venv
```

Activate the virtual environment:

PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:
```bat
.venv\Scripts\activate.bat
```

If PowerShell blocks activation, use Command Prompt or run this in the current PowerShell session if permitted:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
git clone https://github.com/MabelMoncy/ml_lab.git
cd ml_lab
python3 -m venv .venv
source .venv/bin/activate
```

If creating the environment fails because `venv` is unavailable, install your system's Python venv package. For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3-venv
```

## Install libraries

With the virtual environment activated, run this on Windows or Linux:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn seaborn tensorflow
```

The packages cover Experiments 1, 2, 3, 6, 7 and 9. `seaborn` is needed for Experiment 2; TensorFlow/Keras is used to load Fashion MNIST for Experiment 7.

## Verify installation

```bash
python -c "import numpy, pandas, matplotlib, sklearn, seaborn, tensorflow; print('All libraries installed successfully!')"
```

Expected output:

```text
All libraries installed successfully!
```

## Run experiments

Use the actual filename in your repository. Example filenames:

```bash
python experiment1.py
python experiment2.py
python experiment3.py
python experiment6.py
python experiment7.py
python experiment9.py
```

The filenames above are examples; change them to match your saved Python files.

### Experiment 1 — Simple Linear Regression

**Dataset:** `housing.csv`.

The source program uses a Google Colab path:

```python
df = pd.read_csv("/content/sample_data/housing.csv")
```

For a local project, place `housing.csv` in the project directory and change the path to:

```python
df = pd.read_csv("housing.csv")
```

The program prints:

```text
Slope (b1): ...
Intercept (b0): ...
MSE: ...
R²: ...
```

It also displays a scatter plot and fitted regression line.

### Experiment 2 — Linear vs Polynomial Regression (optional)

Uses Seaborn's built-in `mpg` dataset. It prints MSE and R² for Linear Regression and Polynomial Regression (degrees 2, 3 and 5), and displays the model curves.

### Experiment 3 — Linear, Ridge and Lasso Regression

Uses Scikit-learn's built-in Diabetes dataset; no separate CSV is needed.

Output includes a model performance table with MSE and R², an actual-vs-predicted scatter plot, a coefficient comparison table and bar chart, and the selected Ridge/Lasso alpha values. The source program prints:

```text
Model Performance:
Model           MSE             R2
...
Coefficients:
...
Best Alpha Values:
Ridge Alpha: ...
Lasso Alpha: ...
```

### Experiment 6 — Multinomial vs Bernoulli Naïve Bayes

Uses the 20 Newsgroups dataset and may download it on the first run.

Output includes accuracy, macro F1-score and a classification report for each model:

```text
=== Multinomial NB ===
Accuracy: ...
F1 Score: ...
...

=== Bernoulli NB ===
Accuracy: ...
F1 Score: ...
...
```

Exact numeric values depend on the run and installed library versions.

### Experiment 7 — KNN on Fashion MNIST

Uses TensorFlow/Keras to load Fashion MNIST; the first run may download the dataset. The supplied code trains on the first 10,000 training samples and tests on the first 2,000 test samples, for `k_values = [1, 3, 5]`.

For each K, it prints:

```text
===== K = 1 =====
Accuracy: ...
...
===== K = 3 =====
Accuracy: ...
...
===== K = 5 =====
Accuracy: ...
...
```

Each section includes a classification report.

### Experiment 9 — Linear SVM on Iris

Uses Scikit-learn's built-in Iris dataset; no separate dataset file is required. The plot shows the data points, linear decision boundary, margins and support vectors, with axis labels and the title `Linear SVM on Iris Dataset`.

## Deactivate the environment

When finished:

```bash
deactivate
```

## Keep the virtual environment out of Git

Add this to a `.gitignore` file in the project root:

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
```

## Troubleshooting

- **`ModuleNotFoundError`:** Activate `.venv` and reinstall the packages.
- **Dataset download fails:** Check the internet connection and retry.
- **`housing.csv` not found:** Put the file in the project folder and use `pd.read_csv("housing.csv")`, or update the path.
- **TensorFlow installation fails:** Check that your Python version and operating system are supported by the installed TensorFlow version.
- **Plots do not appear:** Run in a Python environment with Matplotlib display support.

Good luck with your Machine Learning lab exam!
