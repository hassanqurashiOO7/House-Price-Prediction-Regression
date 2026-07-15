# House Price Prediction 🏠

A Multiple Linear Regression project predicting house sale prices using the Ames Housing dataset — built by deliberately narrowing 81 columns down to a small, high-impact feature set rather than using everything.

## Dataset

- **Source:** Kaggle — [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Original size:** 1,460 rows, 81 columns
- **Target:** `SalePrice`

## Why Not All 81 Columns?

This dataset is famous for being large and messy — many columns are barely populated (e.g. `PoolQC` had data for only 7 of 1,460 houses), and using all of them would require heavy missing-value handling and encoding for very little gain. Instead of processing every column, the approach here was:

1. Find which **numeric** features actually correlate with `SalePrice`
2. Check those features against each other for **multicollinearity** (features that duplicate the same information)
3. Add a small number of **categorical** features known to matter in real estate, verified visually rather than assumed
4. Drop anything that didn't clear these checks

This kept the project focused on features that actually move the needle, instead of spending time cleaning 70+ columns with marginal impact.

## Feature Selection Process

### Step 1 — Correlation with SalePrice
Checked correlation of all numeric columns against `SalePrice` and kept the top candidates:

`OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt`

### Step 2 — Removing Multicollinearity
Some of these features were highly correlated *with each other*, meaning they carried duplicate information:

| Pair | Correlation | Kept | Dropped |
|---|---|---|---|
| GarageCars vs GarageArea | 0.81 | GarageCars (higher correlation with SalePrice) | GarageArea |
| GrLivArea vs TotRmsAbvGrd | 0.83 | GrLivArea (higher correlation with SalePrice) | TotRmsAbvGrd |
| TotalBsmtSF vs 1stFlrSF | 0.81 | TotalBsmtSF (higher correlation with SalePrice) | 1stFlrSF |

### Step 3 — Final Numeric Features
```
OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt
```

### Step 4 — Adding Categorical Features (Verified, Not Assumed)
Rather than guessing which text columns mattered, each candidate was checked with a boxplot against `SalePrice` to confirm a real price difference existed between categories:

```
Neighborhood, ExterQual, KitchenQual, BsmtQual, CentralAir
```

## Handling Outliers

Two houses stood out sharply from the rest: very large living area (`GrLivArea` > 4000 sq ft) and top overall quality, but priced far below where the trend predicted (~$160k–185k instead of an expected $400k+). These are well-documented outliers in the Ames dataset (often linked to non-market sale conditions) and were removed since they were pulling the regression line off the true trend — most other high-value points followed the expected pattern and were kept.

## Handling Missing Values

Missing values weren't blindly dropped — each case was checked for a pattern first:

- **`BsmtQual` (34 missing):** Verified these belonged to houses with no basement at all — filled with `'None'` rather than dropped, since the missingness itself is meaningful information (not an error).
- **`FireplaceQu`, `MasVnrType`:** Same approach — checked against a related column (`Fireplaces`, `MasVnrArea`) to confirm "missing" meant "feature doesn't exist," then filled with `'None'`.
- **`Alley`, `PoolQC`, `Fence`, `MiscFeature`:** Dropped entirely — over 80% missing, too sparse to be useful.

## Workflow

1. Load data, inspect shape/structure (`.info()`, `.isnull().sum()`)
2. Correlation analysis → shortlist numeric features
3. Multicollinearity check → drop redundant features
4. Boxplot verification → shortlist categorical features
5. Missing value handling (pattern-based fill vs drop)
6. Outlier removal (verified against target, not just feature magnitude)
7. One-Hot Encoding (`pd.get_dummies`) for categorical features
8. Feature scaling (`StandardScaler`) on numeric features only — target left unscaled
9. Train/test split (80/20)
10. Linear Regression model
11. Evaluation — R² and Adjusted R²

## Results

| Metric | Score |
|---|---|
| R² Score | 0.852 |
| Adjusted R² | 0.828 |

The model explains roughly **85% of the variation** in house prices using just **6 numeric + 5 categorical features** — a small fraction of the original 81 columns, with strong performance for a simple linear model.

## Tools & Libraries

- Python, pandas, NumPy
- Matplotlib, Seaborn (EDA, boxplots, correlation heatmaps)
- scikit-learn (StandardScaler, train_test_split, LinearRegression, r2_score)

## Key Takeaways

- More features isn't always better — a focused set of well-verified features can perform close to a full model with far less complexity.
- Correlation alone isn't enough — checking features *against each other* (multicollinearity) avoids redundant information confusing the model.
- Missing data should be investigated before deciding to drop or fill — a missing value often has a specific, meaningful cause.
- Outliers should be removed only when verified against the target variable, not just because a feature value looks large.

## Future Improvements

- Test Ridge/Lasso Regression for additional regularization
- Expand to more categorical features using automated selection (Chi-square, ANOVA)
- Try tree-based models (Random Forest, XGBoost) for comparison
