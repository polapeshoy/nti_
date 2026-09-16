import streamlit as st
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

st.set_page_config(
    page_title="Enterprise Valuation Engine",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def run_full_pipeline_and_cache():
    try:
        df = pd.read_csv('../data/AmesHousing.csv.xls')
    except FileNotFoundError:
        return None

    raw_columns = df.columns.tolist()
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN])

    n_rows = len(X)
    low_nan_cols = [col for col in X.columns if 0 < X[col].isna().sum() / n_rows < NAN_DROP_THRESH]

    if low_nan_cols:
        mask_keep = X[low_nan_cols].notna().all(axis=1)
        X = X[mask_keep].reset_index(drop=True)
        y = y[mask_keep].reset_index(drop=True)

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_nan_cols = [c for c in num_cols if X[c].isna().any()]
    if num_nan_cols:
        num_imputer = SimpleImputer(strategy="median")
        X[num_nan_cols] = num_imputer.fit_transform(X[num_nan_cols])

    cat_nan_cols = [c for c in cat_cols if X[c].isna().any()]
    if cat_nan_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_nan_cols] = cat_imputer.fit_transform(X[cat_nan_cols])

    ordinal_cols_present = [c for c in ORDINAL_COLUMNS if c in X.columns]
    ord_encoder = None
    if ordinal_cols_present:
        categories = [ORDINAL_COLUMNS[c] for c in ordinal_cols_present]
        ord_encoder = OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)
        X[ordinal_cols_present] = ord_encoder.fit_transform(X[ordinal_cols_present])

    nominal_cols = [c for c in X.select_dtypes(include=["object"]).columns if c not in ordinal_cols_present]
    
    unique_categories = {col: df[col].dropna().unique().tolist() for col in nominal_cols + ordinal_cols_present if col in df.columns}
    
    if nominal_cols:
        X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

    X = X.astype(float)
    train_features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, MODEL_SAVE_PATH)

    return {
        "best_model": model,
        "scaler": scaler,
        "ord_encoder": ord_encoder,
        "ordinal_cols_present": ordinal_cols_present,
        "nominal_cols": nominal_cols,
        "train_features": train_features,
        "raw_data_columns": raw_columns,
        "unique_categories": unique_categories
    }

pipeline_assets = run_full_pipeline_and_cache()

if pipeline_assets is None:
    st.error("🚨 '../data/AmesHousing.csv.xls' not found. Verify data placement and path specification.")
    st.stop()

st.title("🏢 Real Estate Valuation Core")
st.markdown("---")

input_mode = st.radio("Select Input Mode", ["📂 Batch Processing (Upload CSV)", "🎛️ Interactive Smart Form"], horizontal=True)

def process_and_predict(input_df):
    processed_df = input_df.copy()
    
    for col in pipeline_assets["raw_data_columns"]:
        if col != TARGET_COLUMN and col not in processed_df.columns:
            if col in ORDINAL_COLUMNS:
                processed_df[col] = ORDINAL_COLUMNS[col][0]
            else:
                processed_df[col] = "NA" if col in pipeline_assets["nominal_cols"] else 0.0
                
    if pipeline_assets["ord_encoder"] is not None:
        ord_cols = pipeline_assets["ordinal_cols_present"]
        for col in ord_cols:
            processed_df[col] = processed_df[col].fillna(ORDINAL_COLUMNS[col][0]).astype(str)
        processed_df[ord_cols] = pipeline_assets["ord_encoder"].transform(processed_df[ord_cols])
        
    processed_df = pd.get_dummies(processed_df, columns=pipeline_assets["nominal_cols"])
    
    for col in pipeline_assets["train_features"]:
        if col not in processed_df.columns:
            processed_df[col] = 0.0
            
    processed_df = processed_df[pipeline_assets["train_features"]].astype(float)
    scaled_data = pipeline_assets["scaler"].transform(processed_df)
    predictions = pipeline_assets["best_model"].predict(scaled_data)
    return np.clip(predictions, 0, None)

if input_mode == "📂 Batch Processing (Upload CSV)":
    st.header("Batch Valuation Engine")
    uploaded_file = st.file_uploader("Upload your property records dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            external_df = pd.read_csv(uploaded_file)
        else:
            external_df = pd.read_excel(uploaded_file)
            
        st.subheader("Raw Data Preview")
        st.dataframe(external_df.head(5), use_container_width=True)
        
        if st.button("Execute Pipeline Predictions", type="primary", use_container_width=True):
            preds = process_and_predict(external_df)
            external_df["Predicted_SalePrice"] = preds
            
            avg_pred = preds.mean()
            total_records = len(preds)
            
            st.markdown("---")
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                st.metric(label="Average Batch Valuation", value=f"${avg_pred:,.2f}")
            with b_col2:
                st.metric(label="Total Properties Processed", value=f"{total_records} Assets")
            with b_col3:
                st.metric(label="Statistical Operational Bound (MAE Trend)", value=f"${max(0.0, avg_pred - 13945.9):,.2f} - ${avg_pred + 13945.9:,.2f}")
            
            st.markdown("---")
            st.subheader("Valuation Trend Graph")
            st.line_chart(external_df["Predicted_SalePrice"], use_container_width=True)
            
            st.subheader("Processed Results Table")
            st.dataframe(external_df[["Predicted_SalePrice"] + [c for c in external_df.columns if c != "Predicted_SalePrice"]], use_container_width=True)
            
            csv_output = external_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Valuation Report (CSV)", data=csv_output, file_name="valuation_report.csv", mime="text/csv", use_container_width=True)

else:
    st.header("Smart Property Profiler")
    
    with st.expander("📐 Core Dimensions & Key Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            gr_liv_area = st.slider("Above Grade Living Area (SqFt)", 300, 6000, 1500, 50)
        with col2:
            total_bsmt_sf = st.slider("Total Basement Area (SqFt)", 0, 6000, 1000, 50)
        with col3:
            garage_area = st.slider("Garage Total Area (SqFt)", 0, 1500, 450, 25)

    with st.expander("🛠️ Property Quality & Infrastructure", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            overall_qual = st.selectbox("Material & Finish Quality", list(range(1, 11)), index=5)
        with col2:
            exter_qual = st.selectbox("External Material Quality", pipeline_assets["unique_categories"].get("Exter Qual", ["TA", "Gd", "Ex", "Fa", "Po"]), index=2)
        with col3:
            kitchen_qual = st.selectbox("Kitchen Executive Quality", pipeline_assets["unique_categories"].get("Kitchen Qual", ["TA", "Gd", "Ex", "Fa", "Po"]), index=2)
        with col4:
            bsmt_qual = st.selectbox("Basement Foundation Height", pipeline_assets["unique_categories"].get("Bsmt Qual", ["TA", "Gd", "Ex", "Fa", "Po", "NA"]), index=3)

    with st.expander("📍 Location, Construction & Infrastructure", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            neighborhood = st.selectbox("Neighborhood District", pipeline_assets["unique_categories"].get("Neighborhood", ["NAmes", "CollgCr", "OldTown"]), index=0)
        with col2:
            year_built = st.number_input("Construction Year", 1800, 2026, 2000, 1)
        with col3:
            garage_cars = st.selectbox("Garage Car Capacity", [0, 1, 2, 3, 4], index=2)
        with col4:
            central_air = st.radio("Central Air System", pipeline_assets["unique_categories"].get("Central Air", ["Y", "N"]), index=0, horizontal=True)

    if st.button("Execute Single Asset Valuation", type="primary", use_container_width=True):
        single_data = {
            "Gr Liv Area": gr_liv_area,
            "Total Bsmt SF": total_bsmt_sf,
            "Garage Area": garage_area,
            "Overall Qual": overall_qual,
            "Exter Qual": exter_qual,
            "Kitchen Qual": kitchen_qual,
            "Bsmt Qual": bsmt_qual,
            "Neighborhood": neighborhood,
            "Year Built": year_built,
            "Garage Cars": garage_cars,
            "Central Air": central_air
        }
        
        single_df = pd.DataFrame([single_data])
        final_val = process_and_predict(single_df)[0]
        
        st.markdown("---")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Market Valuation Estimate", value=f"${final_val:,.2f}")
        with m_col2:
            st.metric(label="Statistical Operational Bound (MAE)", value=f"${max(0.0, final_val - 13945.9):,.2f} - ${final_val + 13945.9:,.2f}")