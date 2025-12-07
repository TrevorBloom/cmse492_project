import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class CreditPipeline:

    def __init__(self, model_dir="trained_models"):
        model_dir = Path(model_dir)

        # Load models
        self.lr = joblib.load(model_dir / "logreg_credit_default.pkl")
        self.rf = joblib.load(model_dir / "rf_credit_default.pkl")

        self.cat = CatBoostClassifier()
        self.cat.load_model(model_dir / "catboost_credit_default.cbm")

        self.scaler = joblib.load(model_dir / "scaler.pkl")

        # Encoders and imputers
        self.bool_cols = ["owns_car", "owns_house"]
        self.cat_cols = ["occupation_type"]
        self.imputer = IterativeImputer(random_state=42)
        self.ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        # Feature list (must match training exactly)
        self.features = [
            'credit_limit_used(%)', 'credit_score', 'prev_defaults',
            'default_in_last_6months', 'no_of_children', 'owns_car',
            'no_of_days_employed', 'yearly_debt_payments', 'migrant_worker',
            'total_family_members', 'credit_score_squared',
            'credit_limit_used_squared', 'credit_score_x_credit_limit_used',
            'credit_ratio_limit'
        ]

        # LR uses a subset
        self.lr_features = [
            'credit_limit_used(%)', 'credit_score', 'prev_defaults',
            'default_in_last_6months','default_x_default_last_6', 
            'prev_defaults_squared', 'credit_ratio_limit', 'credit_score_x_credit_limit_used',
            'credit_limit_used_squared', 'credit_score_squared'
        ]


    # ----------------------------
    # BOOLEAN / CATEGORICAL ENCODING
    # ----------------------------
    def encode_booleans(self, df):
        bool_map = {
            "Y": 1, "N": 0,
            "Yes": 1, "No": 0,
            "True": 1, "False": 0,
            True: 1, False: 0
        }
        for col in self.bool_cols:
            if col in df.columns:
                df[col] = df[col].map(bool_map).fillna(df[col])
        return df

    def encode_categoricals(self, df):
        for col in self.cat_cols:
            if col in df.columns:
                # reshape for encoder
                values = df[[col]].astype(str)
                df[col] = self.ord_encoder.fit_transform(values)
        return df

    # ----------------------------
    # ITERATIVE IMPUTATION
    # ----------------------------
    def impute_missing(self, df):
        df_numeric = df.select_dtypes(include=[np.number])
        df[df_numeric.columns] = self.imputer.fit_transform(df_numeric)
        return df

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------
    def add_features(self, df):
        # existing engineered features
        df["credit_score_squared"] = df["credit_score"] ** 2
        df["credit_limit_used_squared"] = df["credit_limit_used(%)"] ** 2
        df["credit_score_x_credit_limit_used"] = df["credit_score"] * df["credit_limit_used(%)"]
        df["credit_ratio_limit"] = df["credit_limit_used(%)"] / df["credit_score"].replace(0, np.nan)
        df["credit_ratio_limit"] = df["credit_ratio_limit"].fillna(0)

        # missing features for LR
        df["prev_defaults_squared"] = df["prev_defaults"] ** 2

        # create default_x_default_last_6 if not already present
        if "default_x_default_last_6" not in df.columns:
            df["default_x_default_last_6"] = df["default_in_last_6months"] * df["prev_defaults"]

        return df



    # ----------------------------
    # PREP FOR LR (SCALING)
    # ----------------------------
    def preprocess_for_lr(self, df):
        df_lr = df[self.lr_features].copy()
        df_lr[self.lr_features] = self.scaler.transform(df_lr)
        return df_lr

    # ----------------------------
    # COMPLETE PREPROCESSING PIPELINE
    # ----------------------------
    def preprocess(self, df):
        df = df.copy()
        df = self.encode_booleans(df)
        df = self.encode_categoricals(df)
        df = self.impute_missing(df)
        df = self.add_features(df)
        # Return the full df with all engineered features, so LR gets its columns
        return df

    # ----------------------------
    # PREDICTION
    # ----------------------------
    def predict(self, df_raw):
        df = self.preprocess(df_raw)  # full dataframe, all features present
        df_lr = self.preprocess_for_lr(df)  # selects LR-specific features and scales
        df_rf_cat = df[self.features].copy()  # select features RF/CatBoost expect

        results = pd.DataFrame(index=df.index)
        results["lr_proba"] = self.lr.predict_proba(df_lr)[:, 1]
        results["lr_pred"] = self.lr.predict(df_lr)

        results["rf_proba"] = self.rf.predict_proba(df_rf_cat)[:, 1]
        results["rf_pred"] = self.rf.predict(df_rf_cat)

        results["cat_proba"] = self.cat.predict_proba(df_rf_cat)[:, 1]
        results["cat_pred"] = self.cat.predict(df_rf_cat)

        return results


def main():
    parser = argparse.ArgumentParser(description="Run credit risk predictions")
    parser.add_argument("--input", "-i", required=True, help="Input CSV")
    parser.add_argument("--output", "-o", default="predictions.csv", help="Output CSV")
    parser.add_argument("--models", "-m", default="trained_models", help="Models directory")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    pipeline = CreditPipeline(model_dir=args.models)

    results = pipeline.predict(df)
    results.to_csv(args.output, index=False)
    print(f"Saved predictions → {args.output}")


if __name__ == "__main__":
    main()
