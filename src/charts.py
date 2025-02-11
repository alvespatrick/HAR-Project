import numpy as np
import plotly.express as px

def plot_time_series(
    data, 
    y, 
    features_dict,
    signals,
    dataset_name,
    total_samples: int = 2
):

    standart_activity_code = {
        0: "Sit",
        1: "Stand",
        2: "Walk",
        3: "Up_Stairs",
        4: "Down_Stairs",
        5: "Jogging",
    }

    # Select a random sample
    np.random.seed(42)
    for idx in np.random.randint(0, len(data), total_samples):
        print(f"Generating chart for sample {idx}")
        sample = data.iloc[[idx]]
        label = y[idx]
        label = standart_activity_code[label]
        
        acc_signal = [
            sample[features_dict["accel-x"]].values[0],
            sample[features_dict["accel-y"]].values[0],
            sample[features_dict["accel-z"]].values[0]
        ]
        
        gyr_signal = [
            sample[features_dict["gyro-x"]].values[0],
            sample[features_dict["gyro-y"]].values[0],
            sample[features_dict["gyro-z"]].values[0]
        ]
        
        # Plot and save accelerometer data
        acc_fig = px.line(title=f"Accelerometer Data - Activity: {label}")
        for i, signal in enumerate(acc_signal):
            acc_fig.add_scatter(y=signal, name=signals[i])
        acc_fig.write_image(f"figures/{dataset_name}_{idx}_acc.png")
        
        # Plot and save gyroscope data
        gyr_fig = px.line(title=f"Gyroscope Data - Activity: {label}")
        for i, signal in enumerate(gyr_signal):
            gyr_fig.add_scatter(y=signal, name=signals[i+3])
        gyr_fig.write_image(f"figures/{dataset_name}_{idx}_gyr.png")