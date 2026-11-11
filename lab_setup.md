# Setting Up a Python Virtual Environment for ML on Ubuntu

## Step 1: Update your package list and install the required tools

Open your terminal (`Ctrl` + `Alt` + `T`) and run:

```bash
sudo apt update
```

Depending on your Ubuntu version, install the Python 3 virtual environment package (`python3-venv` is standard for Ubuntu 18.04 and 20.04):

* **For Ubuntu 18.04 and 20.04:**
  ```bash
  sudo apt install python3-venv python3-pip
  ```

* **For Ubuntu 16.04** (where `venv` is less common or requires `virtualenv`):
  ```bash
  sudo apt install python3-pip
  pip3 install virtualenv
  ```

---

## Step 2: Create a project directory and navigate to it

Create a folder for your machine learning project and move into it:

```bash
mkdir ml_project
cd ml_project
```

---

## Step 3: Create the virtual environment

Run the command to generate an isolated virtual environment named `venv` (you can replace `venv` with any name you prefer, such as `ml_env`):

* **Using the standard `venv` module (Ubuntu 18.04 & 20.04):**
  ```bash
  python3 -m venv venv
  ```

* **Using `virtualenv` (common on Ubuntu 16.04 or if preferred):**
  ```bash
  virtualenv venv
  ```

---

## Step 4: Activate the virtual environment

Activate the environment so that any packages you install stay localized to this project:

```bash
source venv/bin/activate
```

Once activated, your terminal prompt will be prefixed with the environment name, like this: `(venv) user@ubuntu:~/ml_project$`.

---

## Step 5: Install lightweight ML libraries (for simple programs)

With the virtual environment active, you can install basic machine learning libraries like NumPy, Pandas, or Scikit-learn without affecting your global system Python:

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib
```