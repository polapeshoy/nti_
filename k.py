import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGET_COLUMN   = "SalePrice"
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
NAN_DROP_THRESH = 0.02
MODEL_SAVE_PATH = "best_model.joblib"

ORDINAL_COLUMNS = {
    "Exter Qual":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "Exter Cond":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Qual":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Cond":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Exposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFin Type 1":["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFin Type 2":["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Heating QC":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "Kitchen Qual":  ["Po", "Fa", "TA", "Gd", "Ex"],
    "Fireplace Qu":  ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Finish": ["NA", "Unf", "RFn", "Fin"],
    "Garage Qual":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Cond":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Paved Drive":   ["N", "P", "Y"],
    "Lot Shape":     ["IR3", "IR2", "IR1", "Reg"],
    "Land Slope":    ["Sev", "Mod", "Gtl"],
    "Functional":    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "Fence":         ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Pool QC":       ["NA", "Fa", "TA", "Gd", "Ex"],
    "Utilities":     ["ELO", "NoSeWa", "NoSewr", "AllPub"],
}

DROP_COLUMNS = ["Order", "PID"]

print("=" * 65)
print("   Ames Housing — ML Pipeline")
print("=" * 65)

try:
    df = pd.read_csv('../data/AmesHousing.csv.xls')
    print(f"✓ Data loaded — {df.shape[0]:,} rows × {df.shape[1]} columns\n")
except FileNotFoundError:
    print("⚠  'ames_housing.csv' not found.")
    print("   Place your CSV next to this script and re-run.\n")
    raise SystemExit(1)

print("── Step 1 · Drop irrelevant columns ─────────────────────────")
cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"   Dropped: {cols_to_drop}\n")

print("── Step 2 · Remove duplicates ────────────────────────────────")
n_before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"   Removed {n_before - len(df)} duplicate row(s). Rows remaining: {len(df):,}\n")

print("── Step 3 · Separate target ──────────────────────────────────")
y = df[TARGET_COLUMN].copy()
X = df.drop(columns=[TARGET_COLUMN])
print(f"   Target '{TARGET_COLUMN}' isolated. Feature matrix: {X.shape}\n")

print("── Step 4 · Handle missing values ────────────────────────────")
n_rows = len(X)
low_nan_cols = [
    col for col in X.columns
    if 0 < X[col].isna().sum() / n_rows < NAN_DROP_THRESH
]
print(f"   Columns with <{NAN_DROP_THRESH*100:.0f}% NaNs (drop affected rows): {low_nan_cols}")

if low_nan_cols:
    mask_keep = X[low_nan_cols].notna().all(axis=1)
    X = X[mask_keep].reset_index(drop=True)
    y = y[mask_keep].reset_index(drop=True)
    print(f"   Rows after dropping: {len(X):,}")

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_nan_cols = [c for c in num_cols if X[c].isna().any()]
if num_nan_cols:
    num_imputer = SimpleImputer(strategy="median")
    X[num_nan_cols] = num_imputer.fit_transform(X[num_nan_cols])
    print(f"   Median-imputed numeric cols: {num_nan_cols}")

cat_nan_cols = [c for c in cat_cols if X[c].isna().any()]
if cat_nan_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_nan_cols] = cat_imputer.fit_transform(X[cat_nan_cols])
    print(f"   Mode-imputed categorical cols: {cat_nan_cols}")

print(f"   Remaining NaNs: {X.isna().sum().sum()}\n")

print("── Step 5 · Encode categorical variables ─────────────────────")
ordinal_cols_present = [c for c in ORDINAL_COLUMNS if c in X.columns]
if ordinal_cols_present:
    categories = [ORDINAL_COLUMNS[c] for c in ordinal_cols_present]
    ord_encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    X[ordinal_cols_present] = ord_encoder.fit_transform(X[ordinal_cols_present])
    print(f"   Ordinal-encoded ({len(ordinal_cols_present)} cols): {ordinal_cols_present}")

nominal_cols = [
    c for c in X.select_dtypes(include=["object"]).columns
    if c not in ordinal_cols_present
]
if nominal_cols:
    X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
    print(f"   One-hot encoded {len(nominal_cols)} nominal column(s).")

print(f"   Final feature count after encoding: {X.shape[1]}\n")
X = X.astype(float)

print("── Step 6 · Train/Test split (80/20) ────────────────────────")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"   Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}\n")

print("── Step 7 · Scale features (StandardScaler) ─────────────────")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("   Scaling complete.\n")

print("── Step 8 · Train 5 optimized regression models ──────────────")
models = {
    "Ridge Regression":       Ridge(alpha=10.0, random_state=RANDOM_STATE),
    "Decision Tree":          DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest":          RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=RANDOM_STATE),
    "XGBoost Regressor":      XGBRegressor(n_estimators=150, learning_rate=0.08, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"   ✓ {name} trained")

print()

print("── Step 9 · Evaluate on test set ────────────────────────────")
results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({
        "Model":  name,
        "R² Score": round(r2, 4),
        "MAE ($)":  round(mae, 2),
        "RMSE ($)": round(rmse, 2),
    })

results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False).reset_index(drop=True)
results_df.index += 1

print("\n   ┌─ Model Performance (sorted by R² Score) ─────────────────┐")
print(results_df.to_string())
print("   └──────────────────────────────────────────────────────────┘\n")

print("── Step 10 · Save the best model ────────────────────────────")
best_model_name = results_df.iloc[0]["Model"]
best_model      = trained_models[best_model_name]
best_r2         = results_df.iloc[0]["R² Score"]

joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"   🏆 Best model : {best_model_name}  (R² = {best_r2})")
print(f"   💾 Saved to   : '{MODEL_SAVE_PATH}'\n")

print("── How to reload and predict later ──────────────────────────")
print("   import joblib, numpy as np")
print('   model = joblib.load("best_model.joblib")')
print("   predictions = model.predict(X_new_scaled)")
print("   print(predictions)")
print("=" * 65)
print("   Pipeline complete!")
print("=" * 65)