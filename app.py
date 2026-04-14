import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import time

# Page Config
st.set_page_config(page_title="AutoML Pipeline Pro", layout="wide")
st.title("🚀 Advanced ML Pipeline Dashboard")

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = "Regression"
if 'final_features' not in st.session_state:
    st.session_state.final_features = None

# Horizontal Tabs for Workflow
tabs = st.tabs([
    "1. Config", "2. Data Input & PCA", "3. EDA", 
    "4. Cleaning", "5. Feature Selection", "6. Model Training", "7. Hyper-Tuning"
])

# --- TAB 1: Configuration ---
with tabs[0]:
    st.header("Step 1: Problem Definition")
    st.session_state.problem_type = st.radio(
        "Select Problem Type", 
        ["Classification", "Regression"], 
        index=1 if st.session_state.problem_type == "Regression" else 0
    )
    st.info(f"The pipeline is currently configured for: **{st.session_state.problem_type}**")

# --- TAB 2: Data Input & PCA ---
with tabs[1]:
    st.header("Step 2: Data Input & Visualization")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file or st.session_state.df is not None:
        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
        
        df = st.session_state.df
        st.write("### Data Preview", df.head())
        
        target = st.selectbox("Select Target Feature", df.columns)
        st.session_state.target = target
        
        # PCA Visualization
        st.subheader("PCA Data Shape Visualization")
        available_features = [c for c in df.columns if c != target]

        features = st.multiselect("Select Features for PCA", available_features, default=available_features[:5])
        
        if len(features) >= 2:
            try:
                # Basic preprocessing for PCA (numeric only)
                numeric_df = df[features].select_dtypes(include=[np.number]).dropna()
                if not numeric_df.empty:
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(StandardScaler().fit_transform(numeric_df))
                    fig = px.scatter(components, x=0, y=1, color=df.loc[numeric_df.index, target],
                                    title="2D PCA Projection", labels={'0': 'PC1', '1': 'PC2'})
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Please select numeric features or clean the data first for PCA.")
            except Exception as e:
                st.error(f"PCA Error: {e}")

# --- TAB 3: EDA ---
with tabs[2]:
    if st.session_state.df is not None:
        st.header("Step 3: Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Statistics")
            st.write(st.session_state.df.describe())
        
        with col2:
            st.write("#### Missing Values")
            st.write(st.session_state.df.isnull().sum())
            
        st.write("#### Correlation Heatmap")
        numeric_only = st.session_state.df.select_dtypes(include=[np.number])
        if not numeric_only.empty:
            corr = numeric_only.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation")
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Please upload data in Tab 2 first.")

# --- Helper: get string/object columns (pandas 3.0 compatible) ---
def _get_string_columns(dataframe):
    """Select columns with string/object dtype, compatible with pandas 2 and 3."""
    cols = set()
    for col in dataframe.columns:
        if pd.api.types.is_string_dtype(dataframe[col]) or pd.api.types.is_object_dtype(dataframe[col]):
            cols.add(col)
    return list(cols)

# --- TAB 4: Data Engineering & Cleaning ---
with tabs[3]:
    if st.session_state.df is not None:
        st.header("Step 4: Cleaning & Outliers")
        df_clean = st.session_state.df.copy()
        
        # Imputation
        st.subheader("1. Missing Value Imputation")
        imp_method = st.selectbox("Imputation Method", ["Mean", "Median", "Most Frequent"])
        
        # Outlier Detection
        st.subheader("2. Outlier Detection")
        outlier_method = st.selectbox("Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        
        # Clean specific columns for Car dataset if present
        # Mileage: remove ' km' suffix and convert to float
        if 'Mileage' in df_clean.columns and pd.api.types.is_string_dtype(df_clean['Mileage']):
            df_clean['Mileage'] = (
                df_clean['Mileage']
                .str.replace(' km', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            df_clean['Mileage'] = pd.to_numeric(df_clean['Mileage'], errors='coerce')

        # Engine volume: strip unit suffix if present and convert to float
        if 'Engine volume' in df_clean.columns and pd.api.types.is_string_dtype(df_clean['Engine volume']):
            df_clean['Engine volume'] = (
                df_clean['Engine volume']
                .str.split()
                .str[0]
            )
            df_clean['Engine volume'] = pd.to_numeric(df_clean['Engine volume'], errors='coerce')

        # Levy: replace '-' with NaN and convert to float
        if 'Levy' in df_clean.columns and pd.api.types.is_string_dtype(df_clean['Levy']):
            df_clean['Levy'] = df_clean['Levy'].replace('-', np.nan)
            df_clean['Levy'] = pd.to_numeric(df_clean['Levy'], errors='coerce')

        # Logic for removal
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found. Please clean your data or upload a different dataset.")
        else:
            selected_col = st.selectbox("Select column to check outliers", numeric_cols)
            
            col_data = df_clean[selected_col].dropna()
            outliers = pd.Series([False] * len(df_clean), index=df_clean.index)

            if outlier_method == "IQR":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (df_clean[selected_col] < (Q1 - 1.5 * IQR)) | (df_clean[selected_col] > (Q3 + 1.5 * IQR))
            elif outlier_method == "Isolation Forest":
                iso = IsolationForest(contamination=0.05, random_state=42)
                preds = iso.fit_predict(df_clean[[selected_col]].fillna(0))
                outliers = pd.Series(preds == -1, index=df_clean.index)
                
            st.warning(f"Detected {outliers.sum()} outliers using {outlier_method}")
            
            if st.button("Remove Outliers & Apply Imputation"):
                df_clean = df_clean[~outliers]
                # Map user-friendly name to sklearn strategy
                strategy_map = {"Mean": "mean", "Median": "median", "Most Frequent": "most_frequent"}
                strategy = strategy_map[imp_method]
                imputer = SimpleImputer(strategy=strategy)
                num_cols_in_clean = df_clean.select_dtypes(include=[np.number]).columns
                df_clean[num_cols_in_clean] = imputer.fit_transform(df_clean[num_cols_in_clean])
                st.session_state.df = df_clean
                st.success("Data Cleaned!")
    else:
        st.info("Please upload data in Tab 2 first.")

# --- TAB 5: Feature Selection ---
with tabs[4]:
    if st.session_state.df is not None:
        st.header("Step 5: Feature Selection")
        df_fs = st.session_state.df.copy()
        target = st.session_state.target
        
        if target is None:
            st.warning("Please select a target feature in Tab 2 first.")
        elif target not in df_fs.columns:
            st.warning(f"Target column '{target}' not found in data. Please re-select in Tab 2.")
        else:
            # Label Encoding for categorical (pandas 3.0 compatible)
            str_cols = _get_string_columns(df_fs)
            for col in str_cols:
                df_fs[col] = LabelEncoder().fit_transform(df_fs[col].astype(str))
            
            # Ensure all columns are numeric after encoding
            df_fs = df_fs.apply(pd.to_numeric, errors='coerce')
            df_fs = df_fs.dropna(axis=1, how='all')  # Drop cols that are entirely NaN

            if target not in df_fs.columns:
                st.error("Target column was lost during encoding. Please check your data.")
            else:
                X = df_fs.drop(columns=[target])
                y = df_fs[target]
                
                # Drop rows where target is NaN
                valid_mask = y.notna()
                X = X[valid_mask]
                y = y[valid_mask]
                
                # Fill remaining NaNs in X
                X = X.fillna(0)
                
                method = st.radio("Selection Method", ["Variance Threshold", "Correlation", "Information Gain"])
                
                selected_features = X.columns.tolist()
                
                if method == "Variance Threshold":
                    threshold = st.slider("Threshold", 0.0, 1.0, 0.01)
                    vt = VarianceThreshold(threshold=threshold)
                    vt.fit(X)
                    selected_features = X.columns[vt.get_support()].tolist()
                elif method == "Information Gain":
                    try:
                        if st.session_state.problem_type == "Classification":
                            scores = mutual_info_classif(X, y, random_state=42)
                        else:
                            scores = mutual_info_regression(X, y, random_state=42)
                        feat_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
                        st.bar_chart(feat_scores)
                        top_k = st.number_input("Top K Features", 1, len(X.columns), min(5, len(X.columns)))
                        selected_features = feat_scores.head(top_k).index.tolist()
                    except Exception as e:
                        st.error(f"Information Gain error: {e}")
                elif method == "Correlation":
                    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
                    st.bar_chart(corr_with_target)
                    corr_threshold = st.slider("Minimum correlation", 0.0, 1.0, 0.1)
                    selected_features = corr_with_target[corr_with_target >= corr_threshold].index.tolist()
                    
                st.write("Selected Features:", selected_features)
                st.session_state.final_features = selected_features
    else:
        st.info("Please upload data in Tab 2 first.")

# --- TAB 6: Model Training ---
with tabs[5]:
    if st.session_state.df is not None and st.session_state.final_features and st.session_state.target:
        st.header("Step 6: Model Selection & Training")
        
        df_model = st.session_state.df.copy()

        # Encode string columns for training
        str_cols = _get_string_columns(df_model)
        le_dict = {}
        for col in str_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le

        # Validate features exist
        valid_features = [f for f in st.session_state.final_features if f in df_model.columns]
        if not valid_features:
            st.error("No valid features found. Please go back to Feature Selection.")
        else:
            X = df_model[valid_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            y = df_model[st.session_state.target]
            y = pd.to_numeric(y, errors='coerce')

            # Drop rows where target is NaN
            valid_mask = y.notna()
            X = X[valid_mask]
            y = y[valid_mask]

            # --- Performance: optional subsampling for large datasets ---
            n_rows = len(X)
            st.caption(f"Dataset size: **{n_rows:,}** rows × **{len(valid_features)}** features")
            use_sampling = False
            sample_size = n_rows
            if n_rows > 5000:
                use_sampling = st.checkbox("Subsample data for faster training", value=True)
                if use_sampling:
                    sample_size = st.slider("Sample size (rows)", 1000, n_rows, min(5000, n_rows), step=500)
                    X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=sample_size, random_state=42, shuffle=True)
                    X = X_sampled
                    y = y_sampled
                    st.info(f"Using {sample_size:,} sampled rows for training.")

            split_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
            
            if st.session_state.problem_type == "Classification":
                model_choices = ["Logistic Regression", "SVM", "Random Forest"]
            else:
                model_choices = ["Linear Regression", "SVM", "Random Forest"]
            
            model_choice = st.selectbox("Select Model", model_choices)
            k_fold = st.number_input("K-Fold Cross Validation (K)", 2, 10, 3)

            # SVM performance warning
            if model_choice == "SVM" and len(X_train) > 5000:
                st.warning(f"⚠️ SVM is very slow on large datasets ({len(X_train):,} training rows). Consider subsampling or using Random Forest.")
            
            # Model mapping (with n_jobs for parallelism where supported)
            model = None
            if model_choice in ("Linear Regression", "Logistic Regression"):
                model = LogisticRegression(max_iter=1000, n_jobs=-1) if st.session_state.problem_type == "Classification" else LinearRegression(n_jobs=-1)
            elif model_choice == "SVM":
                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf"])
                model = SVC(kernel=kernel) if st.session_state.problem_type == "Classification" else SVR(kernel=kernel)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) if st.session_state.problem_type == "Classification" else RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            if st.button("Train & Validate"):
                with st.spinner("Training model... this may take a moment."):
                    try:
                        t_start = time.time()

                        # Step 1: Fit model
                        model.fit(X_train, y_train)
                        fit_time = time.time() - t_start

                        # Step 2: Cross-validation (parallel)
                        cv_start = time.time()
                        scores = cross_val_score(model, X_train, y_train, cv=k_fold, n_jobs=-1)
                        cv_time = time.time() - cv_start
                        
                        # Step 3: Predictions
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)
                        total_time = time.time() - t_start

                        st.write(f"### K-Fold Mean Score: {scores.mean():.4f}")

                        # Timing info
                        st.caption(f"⏱️ Fit: {fit_time:.1f}s | CV: {cv_time:.1f}s | Total: {total_time:.1f}s")
                        
                        if st.session_state.problem_type == "Regression":
                            train_r2 = r2_score(y_train, y_pred_train)
                            test_r2 = r2_score(y_test, y_pred_test)
                            col_m1, col_m2 = st.columns(2)
                            col_m1.metric("Train R²", f"{train_r2:.4f}")
                            col_m2.metric("Test R²", f"{test_r2:.4f}")
                            
                            if train_r2 > test_r2 + 0.1:
                                st.error("Potential Overfitting Detected!")
                            elif train_r2 < 0.5:
                                st.warning("Potential Underfitting Detected!")
                        else:
                            train_acc = accuracy_score(y_train, y_pred_train)
                            test_acc = accuracy_score(y_test, y_pred_test)
                            col_m1, col_m2 = st.columns(2)
                            col_m1.metric("Train Accuracy", f"{train_acc:.4f}")
                            col_m2.metric("Test Accuracy", f"{test_acc:.4f}")
                    except Exception as e:
                        st.error(f"Training Error: {e}")
    else:
        st.info("Please complete Steps 1-5 first (upload data, select target, and run feature selection).")

# --- TAB 7: Hyperparameter Tuning ---
with tabs[6]:
    st.header("Step 7: Hyperparameter Tuning")
    
    if st.session_state.df is not None and st.session_state.final_features and st.session_state.target:
        tune_method = st.radio("Search Strategy", ["GridSearch", "RandomSearch"])
        
        # Subsampling for hyper-tuning
        tune_sample = st.slider("Max rows for tuning (subsampling speeds up search)", 1000, 19000, 3000, step=500,
                                key="tune_sample")

        if st.button("Start Tuning"):
            st.info("Tuning Random Forest parameters...")
            try:
                df_tune = st.session_state.df.copy()
                
                # Encode string columns
                str_cols = _get_string_columns(df_tune)
                for col in str_cols:
                    df_tune[col] = LabelEncoder().fit_transform(df_tune[col].astype(str))
                
                valid_features = [f for f in st.session_state.final_features if f in df_tune.columns]
                X = df_tune[valid_features].apply(pd.to_numeric, errors='coerce').fillna(0)
                y = pd.to_numeric(df_tune[st.session_state.target], errors='coerce')
                
                valid_mask = y.notna()
                X = X[valid_mask]
                y = y[valid_mask]

                # Subsample for faster tuning
                if len(X) > tune_sample:
                    X, _, y, _ = train_test_split(X, y, train_size=tune_sample, random_state=42)
                    st.caption(f"Using {tune_sample:,} sampled rows for tuning.")

                param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}
                base_model = RandomForestRegressor(n_jobs=-1, random_state=42) if st.session_state.problem_type == "Regression" else RandomForestClassifier(n_jobs=-1, random_state=42)
                
                if tune_method == "GridSearch":
                    search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1)
                else:
                    search = RandomizedSearchCV(base_model, param_grid, cv=3, n_iter=4, random_state=42, n_jobs=-1)

                t_start = time.time()
                search.fit(X, y)
                elapsed = time.time() - t_start
                
                st.success(f"Best Params: {search.best_params_}")
                st.write("Best Score:", search.best_score_)
                st.caption(f"⏱️ Tuning completed in {elapsed:.1f}s")
            except Exception as e:
                st.error(f"Tuning Error: {e}")
    else:
        st.info("Please complete Steps 1-5 first (upload data, select target, and run feature selection).")