# <img src="./docs/azure_logo.png" alt="Azure Logo" style="width:30px;height:30px;"/>  ML Library

## Streamlining Machine Learning Pipelines

The **Machine Learning** class is a versatile Python tool designed to simplify the creation, optimization, and evaluation of machine learning pipelines. It integrates seamlessly with popular libraries such as scikit-learn, Optuna (for hyperparameter tuning), and SHAP (for model interpretation).

---

### Example Usage

For a detailed example using **MLCore**, please refer to the `xxxx.ipynb` notebook in this repository. This notebook demonstrates how to leverage the module to streamline the development and optimization of machine learning models.

---

### Dataset

Start your experimentation with a practical dataset like the [Mobile Phone Price Prediction Cleaned Dataset](https://www.kaggle.com/datasets/ganjerlawrence/mobile-phone-price-prediction-cleaned-dataset/data) from Kaggle. This dataset offers a real-world scenario for applying MLCore to predictive modeling tasks.

---

### Setting Up Your Azure ML Workspace

This project includes a Makefile to automate common tasks. Use `make <command>` to execute specific tasks defined in the Makefile.

#### Step 1: Create a Compute Instance
Begin by creating a compute instance to run your notebooks.

#### Step 2: Install Poetry and Dependencies
Install `poetry` and other dependencies by running:

```bash
make setup
```

This command triggers the setup process defined in the Makefile. For a full list of commands and their functionalities, review the Makefile.

> **Note:** To run your notebooks in this virtual environment, add the created environment to your list of available kernels.

#### Step 3: Push Code to Notebooks
Push your code to the notebooks using SSH or HTTP:
1. Generate an SSH key on the compute instance.
2. Add this SSH key to your Git repository platform.

Once configured, continue with the `Getting_Your_Data.ipynb` notebook to retrieve data, set up your environment, and execute jobs.

---

## Features

This project framework includes the following features:

- Simplified pipeline creation
- Integrated hyperparameter optimization
- Model interpretation support
- Extensible and modular design

---

## Getting Started

### Prerequisites
- Operating System: Windows, Linux, or macOS
- Python (version 3.7 or later)
- Familiarity with command line operations

### Installation
To clone and set up the project:

1. Clone the repository:
    ```bash
    git clone [repository URL]
    ```
2. Change to the repository directory:
    ```bash
    cd [repository name]
    ```
3. Install required dependencies via Poetry:
    ```bash
    make setup
    ```

---

## Quickstart

Follow these steps to get up and running quickly:

1. Clone the repository and navigate into the folder.
2. Run `make setup` to install dependencies.
3. Open the provided Jupyter notebooks and start experimenting.

---

## Demo

A demo application is included to showcase the usage of ML Library. To run the demo, follow the steps below:

1. Set up your compute instance and install dependencies.
2. Run the demo using the provided script/notebook.
3. Enjoy exploring machine learning pipeline creation with ease.

---

## Resources

- [Documentation](#)
- [Related Projects](#)
- [Community Discussions](#)

For additional resources and guidance, visit our repository linked above.

