# Polynomial Regression from Scratch (Feature Engineering + Closed Form)

## Overview
This project implements **Polynomial Regression from scratch in Python**, including manual feature engineering and a closed-form linear regression solver.

It demonstrates how nonlinear relationships can be modeled using linear regression by transforming input features into polynomial space.

Additionally, it compares the custom implementation with `scikit-learn`’s built-in `PolynomialFeatures`.

---

## Core Idea

Polynomial Regression transforms input features:

\[
x \rightarrow [x, x^2, x^3, ..., x^n]
\]

Then applies **Linear Regression** on the transformed feature space.

This allows a linear model to fit nonlinear patterns.

---

## Implemented Components

### 1. Linear Regression (Closed Form)
Implemented from scratch using the Normal Equation:

\[
\theta = (X^T X)^{-1} X^T y
\]

Features:
- Manual bias term handling
- Matrix-based solution using NumPy
- Direct computation without iteration

---

### 2. Polynomial Feature Engineering (From Scratch)
Implemented manual polynomial expansion:

- Expands features to higher degrees
- Example:
  - Input: `x`
  - Output: `[x, x², x³, x⁴]`

No sklearn preprocessing used in custom version.

---

## Dataset
- Position Salaries dataset (`Position_Salaries.csv`)

This dataset is used because it has a clear nonlinear relationship between position level and salary.

---

## Custom Implementation Workflow

1. Load dataset
2. Extract feature `X`
3. Manually construct polynomial features
4. Fit closed-form linear regression on transformed data
5. Predict and visualize results

---

## Comparison with scikit-learn

Also includes implementation using:

- `PolynomialFeatures` from `sklearn`
- `LinearRegression`

Used for validation and comparison.

---

## Key Features

- Polynomial feature expansion from scratch
- Closed-form linear regression implementation
- Visualization of nonlinear curve fitting
- Comparison with sklearn implementation
- Understanding of feature space transformation

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn (for comparison only)


```bash
git clone https://github.com/your-username/polynomial-regression-from-scratch.git
cd polynomial-regression-from-scratch
