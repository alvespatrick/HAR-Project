from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from umap import UMAP
from preprocess import load_data
from tqdm import tqdm
from itertools import product

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

def initialize_models():
    models = {
        "RandomForestClassifier": RandomForestClassifier(),
        "SVC": SVC(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
    }
    return models
def train_models(X_train, y_train, X_validation, y_validation, X_test, y_test):
    
    models = initialize_models()
    
    results = {
        "model_name": [],
        "acc_validation": [],
        "acc_test": [],
    }
    
    for name, model in models.items():
        results["model_name"].append(name)
        model.fit(X_train, y_train)

        y_true = y_validation
        y_pred = model.predict(X_validation)
        acc_validation = accuracy_score(y_true, y_pred)
        acc_validation = round(acc_validation, 3)

        y_true = y_test
        y_pred = model.predict(X_test)
        acc_test = accuracy_score(y_true, y_pred)
        acc_test = round(acc_test, 3)
        
        results["acc_validation"].append(acc_validation)
        results["acc_test"].append(acc_test)
        
    return results

    
    
def compute_fft(
    df, 
    features_dict
):
    
    size = 30
    signals = [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ]
    
    new_features_dict = {}
    
    new_features = {}
    for signal in signals:
        new_features_dict[signal] = []
        for i in range(size):
            new_features[signal + "_fft-" + str(i)] = []
            new_features_dict[signal].append(signal + "_fft-" + str(i))
    for i, row in df.iterrows():
        for signal in signals:
            sample = row[features_dict[signal]]
            sample_fft = np.fft.fft(sample)
            sample_fft = np.abs(sample_fft)
            # Take half of the sample (real signal)
            sample_fft = sample_fft[:len(sample_fft)//2]
                        
            for col in range(size):
                new_features[signal + "_fft-" + str(col)].append(sample_fft[col])
            
    return pd.DataFrame(new_features), new_features_dict

def compute_stats_features(
    df, 
    features_dict
):
        
    X = {}
    for signal in features_dict.keys():
        X[signal + "_mean"] = []
        X[signal + "_std"] = []
        X[signal + "_amplitude"] = []
    
    for signal in features_dict.keys():
        features = features_dict[signal]
        for sample in df[features].values:
            # Compute mean, std and amplitude
            mean = np.mean(sample)
            std = np.std(sample)
            amplitude = np.max(sample) - np.min(sample)
            
            X[signal + "_mean"].append(mean)
            X[signal + "_std"].append(std)
            X[signal + "_amplitude"].append(amplitude)
            
    return pd.DataFrame(X)
    

def compute_results():

    results = {
        "Dataset": [],
        "model_name": [],
        "Domain": [],
        "Dimension Reduction": [],
        "Feature Extraction": [],
        "Final Dimension": [],
        "Accuracy Validation": [],
        "Accuracy Test": [],
    }

    datasets = [
        # "KuHar",
        "MotionSense",
        # "WISDM",
        # "UCI",
    ]
    
    domains = [
        "Time",
        "Frequency",
    ]
    
    feature_exts = [
        None,
        # "Stats",
    ]
    
    dim_reds = [
        12,
        24,
        None,
    ]    
    

    
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
    
    label = "standard activity code"
    
    print("Computing Combinations")
    combinations = list(product(datasets, domains, dim_reds, feature_exts))
    total_combinations = len(combinations)
    print(f"Total combinations: {total_combinations}")
    for (dataset, domain, dim_red, feature_ext) in tqdm(combinations, total=total_combinations):
        # Load data
        X_train, y_train, X_validation, y_validation, X_test, y_test = load_data(dataset, features)
        
        if domain == "Frequency":
            X_train, new_features_dict = compute_fft(X_train, features_dict)
            X_validation, new_features_dict = compute_fft(X_validation, features_dict)
            X_test, new_features_dict = compute_fft(X_test, features_dict)
        
        if feature_ext == "Stats":
            X_train = compute_stats_features(X_train, features_dict)
            X_validation = compute_stats_features(X_validation, features_dict)
            X_test = compute_stats_features(X_test, features_dict)
            
        if dim_red is not None:
            reducer = UMAP(n_components=dim_red, random_state=42)
            X_train = reducer.fit_transform(X_train)
            X_validation = reducer.transform(X_validation)
            X_test = reducer.transform(X_test)
        
        
        acc_results = train_models(X_train, y_train, X_validation, y_validation, X_test, y_test)
        
        for i, model_name in enumerate(acc_results["model_name"]):
            results["Dataset"].append(dataset)
            results["model_name"].append(model_name)
            results["Domain"].append(domain)
            results["Dimension Reduction"].append(dim_red)
            results["Feature Extraction"].append(feature_ext)
            results["Final Dimension"].append(X_train.shape[1])
            results["Accuracy Validation"].append(acc_results["acc_validation"][i])
            results["Accuracy Test"].append(acc_results["acc_test"][i])
            
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/results.csv", index=False)
    
def best_model(
    df_results
):
    df_results = pd.read_csv("results/results.csv")
    datasets = df_results["Dataset"].unique()
    for dataset in datasets:
        df_dataset = df_results[df_results["Dataset"] == dataset].sort_values(by="Accuracy Validation", ascending=False)
        print(f"Best model for {dataset}:")
        print(df_dataset.iloc[0])
        print("\n")