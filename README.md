# Drought Modeling in the Murray-Darling Basin using Artificial Neural Networks

This repository contains a project focused on modeling drought conditions in the **Murray-Darling Basin**, a vital agricultural region in Australia. The project applies **artificial neural networks (ANNs)** to predict drought occurrence and intensity using climate data derived from the ERA5 dataset. Two tasks are addressed in this project:

- **Task 1: Classification** – Predicting whether there is a drought or no drought based on climate conditions.
- **Task 2: Regression** – Predicting the intensity of a drought based on the same climate conditions.

## 1. Problem Overview

The **Murray-Darling Basin** is one of the most significant agricultural regions in Australia, producing a large portion of the nation's food and fiber (as shown in Figure 1). The basin supports a variety of crops and livestock, making it critical for the country's food production. However, the region is highly prone to droughts, which significantly impact water availability and agricultural productivity, thereby threatening food security. Therefore, monitoring drought conditions in the Murray-Darling Basin is crucial for managing water resources efficiently.

### Tasks:
- **Classification Task**: The goal is to predict whether a grid cell is experiencing a drought (binary classification: 'drought' or 'no drought').
- **Regression Task**: The aim is to predict the intensity of drought conditions represented by the **Standardised Precipitation Index (SPI)**, a proxy for drought intensity.

## 2. Dataset

The dataset used for this project is derived from the **ERA5 global climate dataset**, which combines various climate measurements and simulations using advanced numerical weather models. It contains monthly data for several climate variables associated with droughts, covering the years from **1979 to 2020**.

The dataset used in this project, named `Climate_SPI.csv`, was processed specifically for the Murray-Darling Basin. It contains:

- **Time Information**: Year and Month.
- **Climate Variables**: 12 climate variables are provided, representing temperature, humidity, pressure, wind, and cloud cover, which are known to influence drought conditions. A description of these variables is provided in Table 1.
- **SPI**: The Standardised Precipitation Index (SPI), which serves as a measure of drought intensity.

### Table 1: List of Climate Variables in `Climate_SPI.csv`

| Variable | Description |
|----------|-------------|
| `mn2t`   | Minimum temperature at 2 meters (°K) |
| `msl`    | Mean sea level pressure (Pa) |
| `mx2t`   | Maximum temperature at 2 meters (°K) |
| `q`      | Specific humidity (kg/kg) |
| `t`      | Average temperature at 850 hPa pressure level (°K) |
| `t2`     | Average temperature at 2 meters (°K) |
| `tcc`    | Total cloud cover (0-1) |
| `u`      | U wind component at 850 hPa pressure level (m/s) |
| `u10`    | U wind component at 10 meters (m/s) |
| `v`      | V wind component at 850 hPa pressure level (m/s) |
| `v10`    | V wind component at 10 meters (m/s) |
| `z`      | Geopotential (m^2/s^2) |
| `SPI`    | Standardised Precipitation Index (Unitless) |

Each grid cell in the dataset is represented by a **Grid ID** and contains climate data for that specific location over time. **SPI** values are used to label drought conditions, with low values indicating dry periods and high values indicating wet periods.

## 3. Target Variable

The target variable for both tasks is **SPI** (Standardised Precipitation Index), which is used to measure the intensity of drought. For the classification task, we derive a binary **Drought** variable from the SPI values:

- If **SPI ≤ -1**, the grid cell is classified as experiencing a drought (`Drought = 1`).
- If **SPI > -1**, the grid cell is classified as having no drought (`Drought = 0`).

For the regression task, the objective is to predict the actual SPI value, which provides the drought intensity for the grid cells.

## 4. Methodology

The project involves the following steps for each task:

### Task 1: Classification (Drought or No Drought)
- **Preprocessing**: Climate data is preprocessed by normalizing values and splitting into training, validation, and test sets.
- **Model Architecture**: A neural network is built using Keras and TensorFlow for binary classification.
- **Training**: The model is trained using the training data and validated on the validation set. Hyperparameters such as learning rate, batch size, and epochs are tuned.
- **Evaluation**: The model performance is evaluated on the test set using metrics such as accuracy, confusion matrix, precision, and balanced accuracy.

### Task 2: Regression (Drought Intensity Prediction)
- **Preprocessing**: Similar to the classification task, the dataset is split and normalized.
- **Model Architecture**: A neural network is used for regression, predicting the SPI values.
- **Training**: The model is trained to minimize mean squared error (MSE) using the training data, with validation during training to prevent overfitting.
- **Evaluation**: The model's performance is evaluated on the test set using metrics such as Mean Absolute Error (MAE) and Pearson Correlation Coefficient, along with a scatter plot comparing predicted and true SPI values.

## 5. Results

The project produces two sets of models:
- **Classification Model**: Predicts whether a grid cell is experiencing a drought.
- **Regression Model**: Predicts the intensity of drought conditions (SPI).

The evaluation results for each task include performance metrics such as accuracy, precision, MAE, and correlation coefficients.

## 6. Repository Structure

- **data/**: Contains the `Climate_SPI.csv` dataset.
- **notebooks/**: Jupyter notebooks with detailed implementations of both classification and regression tasks.
- **models/**: Saved models for both tasks.
- **README.md**: This file.

