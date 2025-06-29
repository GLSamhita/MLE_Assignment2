import os
import glob
import pandas as pd
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark

from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse

def main(snapshotdate_str):
    try:
        """
        Main function to train and save the XGBoost churn model.
        """
        print(f"Starting XGBoost model training for date: {snapshotdate_str}")

        # --- 1. Setup Spark and Config ---
        print("Setting up Spark session...")
        spark = pyspark.sql.SparkSession.builder \
            .appName("xgboost_training") \
            .master("local[*]") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        print("Spark session created.")

        print("Building configuration based on provided training date...")
        # Build config based on the provided training date
        config = {}
        config["model_train_date_str"] = snapshotdate_str
        config["model_train_date"] = datetime.strptime(snapshotdate_str, "%Y-%m-%d")
        config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
        config["oot_start_date"] = config['model_train_date'] - relativedelta(months=1)
        config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
        config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=3)
        config["train_test_ratio"] = 0.8
        print("Configuration set:")
        print(config)

        # --- 2. Load Data ---
        print("Loading data from label store...")
        # Load Label Store
        label_store_path = "datamart/gold/label_store/"
        label_files = glob.glob(os.path.join(label_store_path, '*.parquet'))
        label_store_sdf = spark.read.option("header", "true").parquet(*label_files)
        labels_sdf = label_store_sdf.filter(
            (col("snapshot_date") >= config["train_test_start_date"]) &
            (col("snapshot_date") <= config["oot_end_date"])
        )
        print(f"Loaded label store with {labels_sdf.count()} records.")

        print("Loading data from feature store...")
        # Load Feature Store
        feature_store_path = "datamart/gold/feature_store/"
        feature_files = glob.glob(os.path.join(feature_store_path, '*.parquet'))
        valid_feature_files = [f for f in feature_files if os.path.getsize(f) > 0]
        features_store_sdf = spark.read.option("header", "true").parquet(*valid_feature_files)
        features_sdf = features_store_sdf.filter(
            (col("snapshot_date") >= config["train_test_start_date"]) &
            (col("snapshot_date") <= config["oot_end_date"])
        )
        print(f"Loaded feature store with {features_sdf.count()} records.")

        # --- 3. Prepare Data for Modeling ---
        print("Preparing data for modeling...")
        data_pdf = labels_sdf.join(features_store_sdf.drop('snapshot_date'), on=["customer_id"], how="left").toPandas()
        data_pdf['snapshot_Date'] = pd.to_datetime(data_pdf['snapshot_date']).dt.date
        print(f"Joined data loaded with {data_pdf.shape[0]} rows.")

        # Split data into train, test, oot
        oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
        train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]

        print("Coolumns")
        print(data_pdf.columns.tolist())
        feature_cols = data_pdf.columns.tolist()[5:-1]

        X_oot = oot_pdf[feature_cols]
        y_oot = oot_pdf["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            train_test_pdf[feature_cols], train_test_pdf["label"], 
            test_size= 1 - config["train_test_ratio"],
            random_state=88,     # Ensures reproducibility
            shuffle=True,        # Shuffle the data before splitting
            stratify=train_test_pdf["label"]           # Stratify based on the label column
        )
        print(f"Train-test split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}, OOT size: {X_oot.shape[0]}")

        # --- 4. Preprocess Data ---
        print("Preprocessing data...")
        scaler = StandardScaler()

        transformer_stdscaler = scaler.fit(X_train)  # Fit on training data
        print("StandardScaler fitted on training data.")

        # Transform data
        X_train_processed = transformer_stdscaler.transform(X_train)
        X_test_processed = transformer_stdscaler.transform(X_test)
        X_oot_processed = transformer_stdscaler.transform(X_oot)
        print("Data transformed using StandardScaler.")

        # --- 5. Train XGBoost Model ---
        print("Training XGBoost model...")
        xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)

        # Define the hyperparameter space to search
        param_dist = {
            'n_estimators': [25, 50],
            'max_depth': [2, 3],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }

        auc_scorer = make_scorer(roc_auc_score)
        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_dist,
            scoring=auc_scorer,
            n_iter=100,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        print("RandomizedSearchCV setup complete.")

        # Perform the random search
        random_search.fit(X_train_processed, y_train)
        print("Random search complete.")

        # Output the best parameters and best score
        print("Best parameters found: ", random_search.best_params_)
        print("Best AUC score: ", random_search.best_score_)

        # --- 6. Evaluate Model ---
        print("Evaluating model on training data...")
        best_model = random_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
        train_auc_score = roc_auc_score(y_train, y_pred_proba)
        print("Train AUC score: ", train_auc_score)

        print("Evaluating model on test data...")
        y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
        test_auc_score = roc_auc_score(y_test, y_pred_proba)
        print("Test AUC score: ", test_auc_score)

        print("Evaluating model on OOT data...")
        y_pred_proba = best_model.predict_proba(X_oot_processed)[:, 1]
        oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
        print("OOT AUC score: ", oot_auc_score)

        print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
        print("Test GINI score: ", round(2*test_auc_score-1,3))
        print("OOT GINI score: ", round(2*oot_auc_score-1,3))

        # --- 7. Save Model ---
        print("Saving model...")
        model_artefact = {}

        model_artefact['model'] = best_model
        model_artefact['model_version'] = "model_"+config["model_train_date_str"].replace('-','_')
        model_artefact['preprocessing_transformers'] = {'stdscaler': transformer_stdscaler}
        model_artefact['data_dates'] = config
        model_artefact['data_stats'] = {
            'X_train': X_train.shape[0],
            'X_test': X_test.shape[0],
            'X_oot': X_oot.shape[0],
            'y_train': round(y_train.mean(),2),
            'y_test': round(y_test.mean(),2),
            'y_oot': round(y_oot.mean(),2)
        }
        model_artefact['results'] = {
            'auc_train': train_auc_score,
            'auc_test': test_auc_score,
            'auc_oot': oot_auc_score,
            'gini_train': round(2*train_auc_score-1,3),
            'gini_test': round(2*test_auc_score-1,3),
            'gini_oot': round(2*oot_auc_score-1,3)
        }
        model_artefact['hp_params'] = random_search.best_params_

        # Create model bank directory
        model_bank_directory = "model_bank/"
        if not os.path.exists(model_bank_directory):
            os.makedirs(model_bank_directory)
        print(f"Model bank directory created at {model_bank_directory}.")

        # Full path to the file
        file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

        # Write the model to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(model_artefact, file)
        print(f"Model saved to {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()
    main(args.snapshotdate)