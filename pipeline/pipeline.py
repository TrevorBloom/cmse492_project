import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler


class CreditPipeline:

    def __init__(self, model_dir="trained_models"):
        model_dir = Path(model_dir)

        # Load models
        self.lr = joblib.load(model_dir / "logreg_credit_default.pkl")
        self.rf = joblib.load(model_dir / "rf_credit_default.pkl")

        self.cat = CatBoostClassifier()
        self.cat.load_model(model_dir / "catboost_credit_default.cbm")

        self.scaler = joblib.load(model_dir / "scaler.pkl")

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
    # 1. BOOLEAN / CATEGORY CLEANING
    # ----------------------------
    def normalize_booleans(self, df):
        bool_map = {
            "Y": 1, "N": 0,
            "Yes": 1, "No": 0,
            "True": 1, "False": 0,
            True: 1, False: 0
        }

        for col in ["owns_car", "migrant_worker", "default_in_last_6months"]:
            if col in df.columns:
                df[col] = df[col].map(bool_map).fillna(df[col])
        return df

    # ----------------------------
    # 2. IMPUTATION
    # ----------------------------
    def impute_missing(self, df):
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

    # ----------------------------
    # 3. FEATURE ENGINEERING
    # ----------------------------
    def add_features(self, df):
        df["credit_score_squared"] = df["credit_score"] ** 2
        df["credit_limit_used_squared"] = df["credit_limit_used(%)"] ** 2
        df["credit_score_x_credit_limit_used"] = (
            df["credit_score"] * df["credit_limit_used(%)"]
        )
        df["credit_ratio_limit"] = df["credit_limit_used(%)"] / df["credit_score"].replace(0, np.nan)
        df["credit_ratio_limit"] = df["credit_ratio_limit"].fillna(0)

        # New features
        df["prev_defaults_squared"] = df["prev_defaults"] ** 2
        df["default_x_default_last_6"] = df["prev_defaults"] * df["default_in_last_6months"]

        return df



    # ----------------------------
    # 4. PREP FOR LR (SCALING)
    # ----------------------------
    def preprocess_for_lr(self, df):
        df_lr = df[self.lr_features].copy()
        df_lr[self.lr_features] = self.scaler.transform(df_lr)
        return df_lr

    # ----------------------------
    # 5. COMPLETE PREPROCESSING PIPELINE
    # ----------------------------
    def preprocess(self, df):
        df = df.copy()
        df = self.normalize_booleans(df)
        df = self.impute_missing(df)
        df = self.add_features(df)
        return df

    # ----------------------------
    # 6. PREDICTION
    # ----------------------------
    def predict(self, df_raw):
        df = self.preprocess(df_raw)
        df_lr = self.preprocess_for_lr(df)
        df = df[self.features].copy()

        results = pd.DataFrame(index=df.index)

        results["lr_proba"] = self.lr.predict_proba(df_lr)[:, 1]
        results["lr_pred"] = self.lr.predict(df_lr)

        results["rf_proba"] = self.rf.predict_proba(df)[:, 1]
        results["rf_pred"] = self.rf.predict(df)

        results["cat_proba"] = self.cat.predict_proba(df)[:, 1]
        results["cat_pred"] = self.cat.predict(df)

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
