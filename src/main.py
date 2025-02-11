from train import compute_results, best_model
from preprocess import load_data
from charts import plot_time_series
import pandas as pd

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

signals = [
    "accel-x",
    "accel-y",
    "accel-z",
    "gyro-x",
    "gyro-y",
    "gyro-z",
]

samples = 60
features_dict = { }
for s in signals:
    features_dict[s] = []
    for i in range(samples):
        features_dict[s].append(s + "-" + str(i))
        
features = []
for feat in features_dict.values():
    features.extend(feat)
features

if __name__ == "__main__":
    

    X_train, y_train, X_validation, y_validation, X_test, y_test = load_data(
        "MotionSense",
        features
    )        
    
    print("Generating charts...")
    plot_time_series(X_train, y_train, features_dict, signals, "MotionSense", total_samples=2)
    print("Charts generated")
    
    # Compute results
    compute_results()
    print("Results computed")
    
    # Best model
    df_results = pd.read_csv("results/results.csv")
    best_model(df_results)