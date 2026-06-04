# FUNCTIONS FOR ENSEMBLE MODELS

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE,SequentialFeatureSelector
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import OLSInfluence

"""
Variables
"""
TARGET = 'encon'
TEMPORAL_COL = 'timestamp'
ID_COLS = ['vehid', 'trip']
NUMERIC_COLS = ["speed_kmh", "temp_degc", "displacement", "cylinders", "elapsed_min"]
CATEGORICAL_COLS = ["inferred_class", "transmission", "day_of_week", "road_class"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

RF_NUMERIC_COLS = ["speed_kmh", "temp_degc", "displacement", "cylinders", "elapsed_min", "accel", "speed_roll_mean_10", "accel_roll_std_10", "day_sin", "day_cos"]
RF_CATEGORICAL_COLS = ["inferred_class", "transmission", "road_class"]

RF_FEATURE_COLS = RF_NUMERIC_COLS + RF_CATEGORICAL_COLS

"""
Ensemble Models
"""
# Temporal Train/Test/Dev Split
def temporal_split(df, train_frac=0.98, dev_frac=0.01):
    df_sorted = df.sort_values(TEMPORAL_COL).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    dev_end = int(n * (train_frac + dev_frac))
    train = df_sorted.iloc[:train_end]
    dev = df_sorted.iloc[train_end:dev_end]
    test = df_sorted.iloc[dev_end:]
    print(f"Total rows: {n:,}")
    print(f"Train: {len(train):,}, {len(train)/n:.1%}")
    print(f"Dev: {len(dev):,}, {len(dev)/n:.1%}")
    print(f"Test: {len(test):,}, {len(test)/n:.1%}")
    return train, dev, test

# Preprocessing for OLS (use One Hot instead of Label)
def preprocess_for_ols(train, dev, test):
    X_train = train[FEATURE_COLS].copy()
    X_dev = dev[FEATURE_COLS].copy()
    X_test = test[FEATURE_COLS].copy()
    
    y_train = train[TARGET].copy()
    y_dev = dev[TARGET].copy()
    y_test = test[TARGET].copy()

    # One-hot encoding
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=True)
    X_dev = pd.get_dummies(X_dev, columns=CATEGORICAL_COLS, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, drop_first=True)
    
    # Align columns
    X_dev = X_dev.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Scale numeric columns
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_dev[NUMERIC_COLS] = scaler.transform(X_dev[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    X_train = X_train.astype(float)
    X_dev = X_dev.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_dev, X_test, y_train, y_dev, y_test, scaler

# Stepwise Selection (Forward)
def stepwise_selection(X, y, initial_features=[],
                       threshold_in=0.01,
                       threshold_out=0.05,
                       verbose=True):
    included = list(initial_features)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pvalues = pd.Series(index=excluded, dtype=float)

        for feature in excluded:
            model = sm.OLS(y, sm.add_constant(
                    pd.DataFrame(X[included + [feature]]))).fit()
            new_pvalues[feature] = model.pvalues[feature]
        if not new_pvalues.empty:
            best_pval = new_pvalues.min()

            if best_pval < threshold_in:
                best_feature = new_pvalues.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f"Add {best_feature:30} p={best_pval:.6f}")
        # Backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # Exclude intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f"Drop {worst_feature:30} p={worst_pval:.6f}")
        if not changed:
            break

    return included
    
# Preprocessing for Random Forest and Boosting
def safe_transform(series, encoder):
            known = set(encoder.classes_)
            return series.astype(str).apply(
                lambda x: encoder.transform([x])[0] if x in known else -1)
    
def preprocess(train, dev, test):
    encoders = {}
    # Label-encode each categorical column
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(train[col].astype(str))
        encoders[col] = le        
        train[col] = le.transform(train[col].astype(str))
        dev[col] = safe_transform(dev[col],  le)
        test[col] = safe_transform(test[col], le)
 
    X_train = train[FEATURE_COLS].values
    X_dev = dev[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values
 
    y_train = train[TARGET].values
    y_dev = dev[TARGET].values
    y_test = test[TARGET].values
 
    num_idx = [FEATURE_COLS.index(c) for c in NUMERIC_COLS]
    scaler = StandardScaler()
    X_train_scaled = X_train.astype(float)
    X_dev_scaled = X_dev.astype(float)
    X_test_scaled = X_test.astype(float)
    X_train_scaled[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
    X_dev_scaled[:, num_idx] = scaler.transform(X_dev[:, num_idx])
    X_test_scaled[:, num_idx] = scaler.transform(X_test[:, num_idx])
 
    print(f"Feature matrix shapes: (train: {X_train.shape}, dev: {X_dev.shape}, test: {X_test.shape})")
    print(f"Target range for train set: [{y_train.min():.3f}, {y_train.max():.3f}], mean={y_train.mean():.3f}")
 
    return (X_train, X_dev, X_test, X_train_scaled, X_dev_scaled, X_test_scaled, y_train, y_dev, y_test, scaler, encoders)
    
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 0.01      
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float("nan")
    print(f"{name}")
    print(f"MAE: {mae:.4f} L/hr")
    print(f"RMSE: {rmse:.4f} L/hr")
    print(f"R-squared: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    #return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def rf_plot(preds, y_dev, X_dev, TARGET_UNIT):
    C_BLUE = "#378ADD"
    C_CORAL = "#D85A30"
    C_TEAL = "#1D9E75"
    C_AMBER = "#BA7517"
    C_GRAY = "#888780"

    print("Residual diagnostics")
 
    residuals = preds - y_dev
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
     
    # 3a. Residuals vs predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(preds, residuals, alpha=0.15, s=4, color=C_BLUE)
    ax1.axhline(0, color=C_CORAL, linewidth=1)
    ax1.set_xlabel(f"Predicted ({TARGET_UNIT})")
    ax1.set_ylabel(f"Residual ({TARGET_UNIT})")
    ax1.set_title("Residuals vs predicted\n(fan shape = heteroscedasticity)")
     
    # 3b. Residuals vs actual
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_dev, residuals, alpha=0.15, s=4, color=C_TEAL)
    ax2.axhline(0, color=C_CORAL, linewidth=1)
    ax2.set_xlabel(f"Actual ({TARGET_UNIT})")
    ax2.set_ylabel(f"Residual ({TARGET_UNIT})")
    ax2.set_title("Residuals vs actual\n(curve = nonlinearity not captured)")
     
    # 3c. Predicted vs actual scatter
    ax3 = fig.add_subplot(gs[0, 2])
    lim = [min(y_dev.min(), preds.min()), max(y_dev.max(), preds.max())]
    ax3.scatter(y_dev, preds, alpha=0.15, s=4, color=C_AMBER)
    ax3.plot(lim, lim, color=C_CORAL, linewidth=1, linestyle="--")
    ax3.set_xlabel(f"Actual ({TARGET_UNIT})")
    ax3.set_ylabel(f"Predicted ({TARGET_UNIT})")
    ax3.set_title("Predicted vs actual\n(points should hug red line)")
     
    # 3d. Residual histogram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(residuals, bins=80, color=C_BLUE, alpha=0.7, edgecolor="none")
    ax4.axvline(0, color=C_CORAL, linewidth=1.5, label="zero")
    ax4.axvline(residuals.mean(),color=C_AMBER, linewidth=1.5,
                linestyle="--", label=f"mean={residuals.mean():.3f}")
    ax4.set_xlabel(f"Residual ({TARGET_UNIT})")
    ax4.set_title("Residual distribution\n(should be centred at 0)")
    ax4.legend(fontsize=9)
     
    # 3e. Absolute error by speed decile 
    ax5 = fig.add_subplot(gs[1, 1])
    speed_idx = RF_FEATURE_COLS.index("speed_kmh") if "speed_kmh" in RF_FEATURE_COLS else None
    if speed_idx is not None:
        speed_vals = X_dev[:, speed_idx]
        deciles = pd.qcut(speed_vals, q=10, duplicates="drop")
        abs_err = np.abs(residuals)
        err_by_dec = pd.Series(abs_err).groupby(deciles).mean()
        ax5.bar(range(len(err_by_dec)), err_by_dec.values, color=C_TEAL, alpha=0.8)
        ax5.set_xticks(range(len(err_by_dec)))
        ax5.set_xticklabels([f"{iv.mid:.0f}" for iv in err_by_dec.index],
                            rotation=45, fontsize=8)
        ax5.set_xlabel("Speed (km/h) — decile midpoints")
        ax5.set_ylabel(f"Mean |error| ({TARGET_UNIT})")
        ax5.set_title("MAE by speed decile\n(spikes = regime the model struggles with)")
    else:
        ax5.text(0.5, 0.5, "speed_kmh not found\nin FEATURE_NAMES",
                 ha="center", va="center", transform=ax5.transAxes)
        ax5.set_title("MAE by speed decile")
     
    # 3f. Absolute error by target decile  (does model fail at high/low encon?)
    ax6 = fig.add_subplot(gs[1, 2])
    target_dec = pd.qcut(y_dev, q=10, duplicates="drop")
    err_by_tgt = pd.Series(np.abs(residuals)).groupby(target_dec).mean()
    ax6.bar(range(len(err_by_tgt)), err_by_tgt.values, color=C_AMBER, alpha=0.8)
    ax6.set_xticks(range(len(err_by_tgt)))
    ax6.set_xticklabels([f"{iv.mid:.1f}" for iv in err_by_tgt.index],
                        rotation=45, fontsize=8)
    ax6.set_xlabel(f"Actual encon ({TARGET_UNIT}) — decile midpoints")
    ax6.set_ylabel(f"Mean |error| ({TARGET_UNIT})")
    ax6.set_title("MAE by target decile\n(high error at extremes = range compression)")
     
    plt.suptitle("Random Forest - Residual diagnostics (dev set)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
     
    # Print residual summary
    print(f"Residual mean : {residuals.mean():.4f}, (bias; should be ~0)")
    print(f"Residual std : {residuals.std():.4f}")
    print(f"  Max overpredict : {residuals.max():.4f}")
    print(f"  Max underpredict: {residuals.min():.4f}")
    pct_within_1 = (np.abs(residuals) < 1.0).mean() * 100
    print(f"% within 1 L/hr : {pct_within_1:.1f}%")


def rf_tree_diagnostic(model, max_depth, min_depth):
    C_BLUE = "#378ADD"
    C_CORAL = "#D85A30"
    C_TEAL = "#1D9E75"
    C_AMBER = "#BA7517"
    C_GRAY = "#888780"
    
    depths = [t.get_depth() for t in model.estimators_]
    leaf_counts = [t.get_n_leaves() for t in model.estimators_]
     
    print(f"Tree depth mean: {np.mean(depths)} "
          f"min: {np.min(depths)}, max: {np.max(depths)}")
    print(f"Leaf nodes mean: {np.mean(leaf_counts)}  "
          f"min: {np.min(leaf_counts)}, max: {np.max(leaf_counts)}")
     
    if np.mean(depths) >= min_depth:
        print("(!) Most trees hit max_depth= => they want to grow deeper.")
        print("=> Either raise max_depth or accept the current regularisation.")
    elif np.mean(depths) < 10:
        print("Trees are shallow: model may be underfitting.")
     
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(depths, bins=20, color=C_BLUE,  alpha=0.8, edgecolor="none")
    axes[0].axvline(30, color=C_CORAL, linestyle="--", linewidth=1, label="max_depth=30")
    axes[0].set_title("Distribution of tree depths")
    axes[0].set_xlabel("Depth"); axes[0].legend(fontsize=9)
     
    axes[1].hist(leaf_counts, bins=20, color=C_TEAL, alpha=0.8, edgecolor="none")
    axes[1].set_title("Distribution of leaf counts per tree")
    axes[1].set_xlabel("Number of leaves")
     
    plt.tight_layout()
    plt.show()


def plot_ols_diagnostics(residuals, fitted):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # (a) Residuals vs Fitted
    axes[0,0].scatter(fitted, residuals,  s=3)
    axes[0,0].axhline(0, color="red", linewidth=1)
    axes[0,0].set_xlabel("Fitted values")
    axes[0,0].set_ylabel("Residuals")
    axes[0,0].set_title("Residuals vs Fitted")
    
    # (b) Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title("Q-Q plot (normality of residuals)")
    
    # (c) Scale-Location (sqrt |residuals| vs fitted)
    axes[1,0].scatter(fitted, np.sqrt(np.abs(residuals)), s=3)
    axes[1,0].set_xlabel("Fitted values")
    axes[1,0].set_ylabel("√|Residuals|")
    axes[1,0].set_title("Scale-Location (homoscedasticity)")
    
    # (d) Residuals distribution
    axes[1,1].hist(residuals, bins=80, edgecolor="none")
    axes[1,1].set_xlabel("Residual")
    axes[1,1].set_ylabel("Frequency")
    axes[1,1].set_title("Residual distribution")
    
    plt.tight_layout()
    plt.show()

def shap_analysis_xgb(model, X, feature_names, max_display=15, sample_n=2000):
    if len(X) > sample_n:
        idx = np.random.choice(len(X), sample_n, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)         
 
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("XGBoost - SHAP Beeswarm", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.close()
 
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feature_names[i] for i in order[::-1]], mean_abs[order[::-1]], color="#E05C5C")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("XGBoost — Feature Importance (SHAP)", fontweight="bold")
    plt.tight_layout()
    plt.close() 
    return shap_values

def plot_regression_diagnostics(model, sample_size=50000, random_state=42):
    # Fitted values and residuals
    fitted = model.fittedvalues
    residuals = model.resid
    # Standardized residuals
    influence = OLSInfluence(model)
    standardized_residuals = influence.resid_studentized_internal
    # Leverage
    leverage = influence.hat_matrix_diag
    # Cook's distance
    cooks = influence.cooks_distance[0]
    # Create DataFrame
    diag_df = pd.DataFrame({
        "fitted": fitted,
        "residuals": residuals,
        "std_resid": standardized_residuals,
        "leverage": leverage,
        "cooks": cooks})

    # Sample for plotting if dataset is large
    if len(diag_df) > sample_size:
        diag_df = diag_df.sample(n=sample_size, random_state=random_state)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(diag_df["fitted"], diag_df["residuals"], alpha=0.3)
    axes[0, 0].axhline(y=0, linestyle='--')
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")

    # 2. Normal Q-Q
    qq = ProbPlot(diag_df["std_resid"])
    qq.qqplot(line='45', ax=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q")


    # 3. Scale-Location
    sqrt_std_resid = np.sqrt(np.abs(diag_df["std_resid"]))
    axes[1, 0].scatter(diag_df["fitted"], sqrt_std_resid, alpha=0.3)
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("Sqrt(|Standardized Residuals|)")

    # 4. Residuals vs Leverage
    axes[1, 1].scatter(diag_df["leverage"], diag_df["std_resid"], s=1000 * diag_df["cooks"], alpha=0.3)
    axes[1, 1].axhline(y=0, linestyle='--')
    axes[1, 1].set_title("Residuals vs Leverage")
    axes[1, 1].set_xlabel("Leverage")
    axes[1, 1].set_ylabel("Standardized Residuals")
    plt.tight_layout()
    plt.show()

