# Tensorflow Core

We go through the tensorflow core APIs mostly in Tensorflow eager mode and perform Linear regression, Logistic Regression and implement the common optimizers used in machine learning

## Files

root
|
+-- data
|   |-- Auto_MPG.csv                            # Fuel Consumption data of cars in the US from the 1970s-1980s
|   |-- Wisconsin_Brest_Cancer_Data.csv         # Breast Cancer data with information if the tumour was malignant or benign
|
+-- notebooks
|   |-- core_lin_reg.ipynb                      # Notebook going through the data pipeline and linear regression.
|   |-- core_log_reg.ipynb                      # Notebook going through the data pipeline and logistic regression.
|
+-- src
|   +-- optimizers                              # C implementions of the common optimizers used in machine learning
|   |   |-- adam (.h) (.c)                      # Implemention for the Adam (Adaptive Momentum) optimizer
|   |   |-- gradient_descent (.h) (.c)          # Implemention for the classic gradient descent algorithm.
|   |   |-- momentum (.h) (.c)                  # Implemention for the Momentum optimizer
|   |   |-- rms_prop (.h) (.c)                  # Implemention for the RMSprop (Root Mean Square propogation) optimizer
|
|   +-- tests
|   |   |-- convergence (.h) (.c)               # Perform convergence tests on a pre defined loss function
|   |   |-- main.c                              # Execute tests
|
|-- README.md

## TODO

Add CMake config file.