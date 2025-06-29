import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI)
    
    Args:
        expected: Reference distribution (baseline)
        actual: Current distribution 
        buckets: Number of buckets for binning
    
    Returns:
        PSI value and binning details
    """
    def scale_range(input_array, new_min, new_max):
        input_min = np.min(input_array)
        input_max = np.max(input_array)
        return ((input_array - input_min) / (input_max - input_min)) * (new_max - new_min) + new_min
    
    # Scale both arrays to 0-1 range for consistent binning
    expected_scaled = scale_range(expected, 0, 1)
    actual_scaled = scale_range(actual, 0, 1)
    
    # Create bins based on expected distribution
    breakpoints = np.arange(0, buckets + 1) / buckets
    
    # Calculate frequencies
    expected_freq = np.histogram(expected_scaled, bins=breakpoints)[0]
    actual_freq = np.histogram(actual_scaled, bins=breakpoints)[0]
    
    # Convert to percentages and handle zeros
    expected_pct = expected_freq / len(expected_scaled)
    actual_pct = actual_freq / len(actual_scaled)
    
    # Replace zeros with small value to avoid log(0)
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_values)
    
    # Create summary dataframe
    psi_df = pd.DataFrame({
        'bucket': range(1, buckets + 1),
        'expected_pct': expected_pct,
        'actual_pct': actual_pct,
        'psi_value': psi_values
    })
    
    return psi, psi_df

def plot_psi_chart(psi_df, psi_value, feature_name, snapshot_date, save_path):
    """Create PSI visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart comparing distributions
    x = psi_df['bucket']
    width = 0.35
    ax1.bar(x - width/2, psi_df['expected_pct'], width, label='Expected (Baseline)', alpha=0.7)
    ax1.bar(x + width/2, psi_df['actual_pct'], width, label='Actual', alpha=0.7)
    ax1.set_xlabel('Bucket')
    ax1.set_ylabel('Percentage')
    ax1.set_title(f'Distribution Comparison - {feature_name}\nPSI: {psi_value:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PSI contribution by bucket
    colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in psi_df['psi_value']]
    ax2.bar(psi_df['bucket'], psi_df['psi_value'], color=colors, alpha=0.7)
    ax2.set_xlabel('Bucket')
    ax2.set_ylabel('PSI Contribution')
    ax2.set_title('PSI Contribution by Bucket')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def update_auc_trend(auc_value, snapshot_date, model_name, monitor_directory):
    """Update AUC trend CSV and create plot"""
    auc_csv_path = os.path.join(monitor_directory, f"{model_name[:-4]}_auc_trend.csv")
    
    # Create new record
    new_record = pd.DataFrame({
        'snapshot_date': [snapshot_date],
        'auc_score': [auc_value],
        'gini_score': [2 * auc_value - 1]
    })
    
    # Load existing data if exists, otherwise create new
    if os.path.exists(auc_csv_path):
        auc_df = pd.read_csv(auc_csv_path)
        auc_df['snapshot_date'] = pd.to_datetime(auc_df['snapshot_date'])
        
        # Remove existing record for same date if exists
        auc_df = auc_df[auc_df['snapshot_date'] != pd.to_datetime(snapshot_date)]
        
        # Append new record
        auc_df = pd.concat([auc_df, new_record], ignore_index=True)
    else:
        auc_df = new_record
    
    # Sort by date
    auc_df['snapshot_date'] = pd.to_datetime(auc_df['snapshot_date'])
    auc_df = auc_df.sort_values('snapshot_date').reset_index(drop=True)
    
    # Save updated CSV
    auc_df.to_csv(auc_csv_path, index=False)
    
    # Create AUC trend plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # AUC trend
    ax1.plot(auc_df['snapshot_date'], auc_df['auc_score'], marker='o', linewidth=2, markersize=6)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random (0.5)')
    ax1.set_ylabel('AUC Score')
    ax1.set_title(f'Model Performance Trend - {model_name[:-4]}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0.4, 1.0)
    
    # GINI trend
    ax2.plot(auc_df['snapshot_date'], auc_df['gini_score'], marker='s', linewidth=2, markersize=6, color='orange')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Random (0.0)')
    ax2.set_xlabel('Snapshot Date')
    ax2.set_ylabel('GINI Score')
    ax2.set_title('GINI Coefficient Trend')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(monitor_directory, f"{model_name[:-4]}_auc_trend.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return auc_df

def create_kde_plots(feature_data, prediction_data, feature_name, snapshot_date, model_name, monitor_directory):
    """Create KDE plots for feature and prediction distributions"""
    
    # Feature KDE plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=feature_data, fill=True, alpha=0.7)
    plt.title(f'Distribution of {feature_name}\nSnapshot: {snapshot_date}')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Prediction score KDE plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=prediction_data, fill=True, alpha=0.7, color='orange')
    plt.title(f'Prediction Score Distribution\nSnapshot: {snapshot_date}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    kde_path = os.path.join(monitor_directory, f"{model_name[:-4]}_kde_{snapshot_date.replace('-', '_')}.png")
    plt.savefig(kde_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(snapshotdate, modelname):
    print('\n\n---Starting Model Monitoring Job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_monitoring") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Configuration
    config = {
        "snapshot_date_str": snapshotdate,
        "snapshot_date": datetime.strptime(snapshotdate, "%Y-%m-%d"),
        "model_name": modelname,
        "monitor_directory": f"model_monitor/{modelname[:-4]}/"
    }
    
    print(f"Monitoring model: {config['model_name']}")
    print(f"Snapshot date: {config['snapshot_date_str']}")
    
    # Create monitor directory
    if not os.path.exists(config["monitor_directory"]):
        os.makedirs(config["monitor_directory"])
        print(f"Created monitor directory: {config['monitor_directory']}")
    
    try:
        # --- Load Model Predictions ---
        print("Loading model predictions...")
        predictions_path = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
        prediction_files = glob.glob(os.path.join(predictions_path, '*.parquet'))
        
        if not prediction_files:
            print(f"No prediction files found in {predictions_path}")
            return
        
        predictions_sdf = spark.read.option("header", "true").parquet(*prediction_files)
        current_predictions_sdf = predictions_sdf.filter(col("snapshot_date") == config["snapshot_date_str"])
        
        if current_predictions_sdf.count() == 0:
            print(f"No predictions found for date {config['snapshot_date_str']}")
            return
        
        current_predictions_pdf = current_predictions_sdf.toPandas()
        print(f"Loaded {len(current_predictions_pdf)} current predictions")
        
        # --- Load Feature Store for Current Date ---
        print("Loading feature store...")
        feature_store_path = "datamart/gold/feature_store/"
        feature_files = glob.glob(os.path.join(feature_store_path, '*.parquet'))
        valid_feature_files = [f for f in feature_files if os.path.getsize(f) > 0]
        
        features_store_sdf = spark.read.option("header", "true").parquet(*valid_feature_files)
        # current_features_sdf = features_store_sdf.filter(col("snapshot_date") == config["snapshot_date_str"])
        current_features_sdf = features_store_sdf
        current_features_pdf = current_features_sdf.toPandas()
        print(f"Loaded {len(current_features_pdf)} current feature records")
        
       
        label_store_path = "datamart/gold/label_store/"
        label_files = glob.glob(os.path.join(label_store_path, '*.parquet'))
        
        if label_files:
            labels_store_sdf = spark.read.option("header", "true").parquet(*label_files)
            current_labels_sdf = labels_store_sdf.filter(col("snapshot_date") == config["snapshot_date_str"])
            
            if current_labels_sdf.count() > 0:
                current_labels_pdf = current_labels_sdf.toPandas()
                
                # Merge predictions with labels for AUC calculation
                perf_data = current_predictions_pdf.merge(
                    current_labels_pdf[['customer_id', 'label']], 
                    on='customer_id', 
                    how='inner'
                )
                
                if len(perf_data) > 0:
                    # Calculate AUC
                    auc_score = roc_auc_score(perf_data['label'], perf_data['model_predictions'])
                    print(f"Current AUC Score: {auc_score:.4f}")
                    print(f"Current GINI Score: {2*auc_score-1:.4f}")
                    
                    # Update AUC trend
                    auc_df = update_auc_trend(auc_score, config["snapshot_date_str"], 
                                            config["model_name"], config["monitor_directory"])
                    print("AUC trend updated and plotted")
                else:
                    print("No matching records between predictions and labels")
            else:
                print(f"No labels found for date {config['snapshot_date_str']}")
        else:
            print("No label files found")
        
        # --- PSI Calculation (using baseline from one month ago) ---
        print("Calculating PSI...")
        baseline_date = (config["snapshot_date"] - relativedelta(months=1)).strftime("%Y-%m-%d")
        baseline_features_sdf = features_store_sdf.filter(col("snapshot_date") == baseline_date)
        
        if baseline_features_sdf.count() > 0:
            baseline_features_pdf = baseline_features_sdf.toPandas()
            print(f"Loaded {len(baseline_features_pdf)} baseline feature records from {baseline_date}")
            
            # Calculate PSI for monthly_inhand_salary feature
            if 'monthly_inhand_salary' in current_features_pdf.columns and 'monthly_inhand_salary' in baseline_features_pdf.columns:
                # Remove null values
                baseline_salary = baseline_features_pdf['monthly_inhand_salary'].dropna()
                current_salary = current_features_pdf['monthly_inhand_salary'].dropna()
                
                if len(baseline_salary) > 0 and len(current_salary) > 0:
                    psi_value, psi_df = calculate_psi(baseline_salary, current_salary)
                    print(f"PSI for monthly_inhand_salary: {psi_value:.4f}")
                    
                    # Interpret PSI
                    if psi_value < 0.1:
                        psi_status = "Low risk - No significant change"
                    elif psi_value < 0.2:
                        psi_status = "Medium risk - Some change detected"
                    else:
                        psi_status = "High risk - Significant change detected"
                    
                    print(f"PSI Status: {psi_status}")
                    
                    # Create PSI plot
                    psi_plot_path = os.path.join(config["monitor_directory"], 
                                                f"{config['model_name'][:-4]}_psi_{config['snapshot_date_str'].replace('-', '_')}.png")
                    plot_psi_chart(psi_df, psi_value, 'monthly_inhand_salary', 
                                 config["snapshot_date_str"], psi_plot_path)
                    print("PSI plot created")
                    
                    # Save PSI results to CSV
                    psi_results = pd.DataFrame({
                        'snapshot_date': [config["snapshot_date_str"]],
                        'baseline_date': [baseline_date],
                        'feature_name': ['monthly_inhand_salary'],
                        'psi_value': [psi_value],
                        'psi_status': [psi_status]
                    })
                    
                    psi_csv_path = os.path.join(config["monitor_directory"], 
                                              f"{config['model_name'][:-4]}_psi_results.csv")
                    
                    if os.path.exists(psi_csv_path):
                        existing_psi = pd.read_csv(psi_csv_path)
                        # Remove existing record for same date if exists
                        existing_psi = existing_psi[existing_psi['snapshot_date'] != config["snapshot_date_str"]]
                        psi_results = pd.concat([existing_psi, psi_results], ignore_index=True)
                    
                    psi_results.to_csv(psi_csv_path, index=False)
                    print("PSI results saved to CSV")
                else:
                    print("Insufficient data for PSI calculation")
            else:
                print("monthly_inhand_salary feature not found in data")
        else:
            print(f"No baseline data found for {baseline_date}")
        
        # --- Create KDE Plots ---
        print("Creating KDE plots...")
        if 'monthly_inhand_salary' in current_features_pdf.columns:
            salary_data = current_features_pdf['monthly_inhand_salary'].dropna()
            prediction_data = current_predictions_pdf['model_predictions']
            
            create_kde_plots(salary_data, prediction_data, 'monthly_inhand_salary',
                           config["snapshot_date_str"], config["model_name"], 
                           config["monitor_directory"])
            print("KDE plots created")
        
        # --- Generate Monitoring Summary ---
        print("Generating monitoring summary...")
        summary_path = os.path.join(config["monitor_directory"], 
                                   f"{config['model_name'][:-4]}_monitoring_summary_{config['snapshot_date_str'].replace('-', '_')}.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Model Monitoring Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Model: {config['model_name']}\n")
            f.write(f"Snapshot Date: {config['snapshot_date_str']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'auc_score' in locals():
                f.write(f"Performance Metrics:\n")
                f.write(f"- AUC Score: {auc_score:.4f}\n")
                f.write(f"- GINI Score: {2*auc_score-1:.4f}\n\n")
            
            if 'psi_value' in locals():
                f.write(f"Data Stability (PSI):\n")
                f.write(f"- Feature: monthly_inhand_salary\n")
                f.write(f"- PSI Value: {psi_value:.4f}\n")
                f.write(f"- Status: {psi_status}\n\n")
            
            f.write(f"Data Volumes:\n")
            f.write(f"- Predictions: {len(current_predictions_pdf)}\n")
            f.write(f"- Features: {len(current_features_pdf)}\n")
            
            if 'perf_data' in locals():
                f.write(f"- Labels (for AUC): {len(perf_data)}\n")
        
        print(f"Monitoring summary saved: {summary_path}")
        
    except Exception as e:
        print(f"Error in model monitoring: {str(e)}")
        raise e
        
    finally:
        # Clean up Spark session
        spark.stop()
    
    print('\n\n---Model Monitoring Job Completed---\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Monitoring Script")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="Model name with .pkl extension")
    
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)