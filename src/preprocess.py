import pandas as pd

def load_data(dataset_name, features):
    data_path = f"../datasets/standardized_balanced/{dataset_name}/"

    # Load data
    train_data = pd.read_csv(data_path + "train.csv")
    validation_data = pd.read_csv(data_path + "validation.csv")
    test_data = pd.read_csv(data_path + "test.csv")

    # Split features and target
    label = "standard activity code"
    X_train = train_data[features]
    y_train = train_data[label]
    X_validation = validation_data[features]
    y_validation = validation_data[label]
    X_test = test_data[features]
    y_test = test_data[label]

    return X_train, y_train, X_validation, y_validation, X_test, y_test