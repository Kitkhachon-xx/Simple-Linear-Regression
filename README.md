# Simple Linear Regression

Learning project implementing **Linear Regression**, **Polynomial Regression**, and **KNN Regressor** from scratch using only NumPy (no `scikit-learn`), built as part of the PIM Machine Learning course.

## Project Structure

```
.
├── 01-Lecture/                          # Notes and worked math examples
│   ├── KNN_Regressor_Math_Summary.md
│   ├── KNN_Regressor_Numeric_Example.md
│   └── KNN_Regressor_Error_Calculation.md
└── 02-Code-Station/
    ├── Jupyter/
    │   └── Linear_Regression.ipynb      # Exploratory notebook (linear, polynomial, KNN)
    └── Python/
        ├── Linear_Regression_NP.py       # Simple linear regression via the Normal Equation
        ├── None_Linear_Regression_NP.py  # Polynomial regression (degree 1, 2, 4)
        └── Knn-Regressor.py              # K-Nearest Neighbors regression
```

## Methods

### Linear Regression (`Linear_Regression_NP.py`)

Fits $y = w_0 x + w_1$ by solving the **Normal Equation** directly:

$$R = Y X^T (X X^T)^{-1}$$

where $X$ is the design matrix (data stacked with a row of ones for the bias term).

### Polynomial Regression (`None_Linear_Regression_NP.py`)

Extends the same Normal Equation approach to a design matrix of stacked powers $[X^M, \dots, X^2, X, 1]$, fitting a degree-$M$ polynomial to non-linear (sine-based) data. Demonstrates degree 1, 2, and 4 fits to show underfitting vs. a better fit.

### KNN Regressor (`Knn-Regressor.py`)

Non-parametric regression: for a query point, finds the $k$ nearest training points by absolute distance and predicts the **mean** of their $Y$ values:

$$\hat{y}(x_0) = \frac{1}{k}\sum_{n \in \mathcal{N}_k(x_0)} Y_n$$

Also includes error evaluation (MAE, MSE, RMSE) using a leave-one-out prediction over the training data.

## Running

Each script is self-contained and runnable directly:

```bash
python 02-Code-Station/Python/Linear_Regression_NP.py
python 02-Code-Station/Python/None_Linear_Regression_NP.py
python 02-Code-Station/Python/Knn-Regressor.py
```

Requires `numpy` and `matplotlib`.

## Notes

- `01-Lecture/` contains detailed, step-by-step numeric walkthroughs of the KNN Regressor math (distance calculation, neighbor selection, averaging, and error metrics) written in Thai for study reference.
- All models are implemented from first principles to illustrate the underlying linear algebra rather than relying on a ready-made ML library.
