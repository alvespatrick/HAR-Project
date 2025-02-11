# HAR-Project

A Human Activity Recognition (HAR) project that analyzes sensor data (accelerometer and gyroscope) to classify different physical activities using various machine learning models.

## Features

- Supports multiple machine learning models:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- Data processing in both time and frequency domains
- Dimension reduction using UMAP
- Visualization of sensor data and results
- Performance comparison across different models and configurations

## Project Structure

├── src/

│   ├── main.py          # Main execution script

│   ├── preprocess.py    # Data preprocessing functions

│   ├── train.py         # Model training and evaluation

│   └── charts.py        # Visualization functions

├── datasets/            # Dataset directory (not included in repo)

├── datasets/            # Dataset directory (not included in repo)

├── results/             # Model evaluation results

├── figures/             # Generated visualizations

├── results/             # Model evaluation results

├── requirements.txt     # Python dependencies

└── README.md


## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

### Using Python Virtual Environment

1. Create and activate a virtual environment:

```bash
python -m venv venv
source har/bin/activate  # On Windows: har\Scripts\activate
```

### Using Docker

1. Build the Docker image:

```bash
docker build -t har-project .
```

2. Run the container:

```bash
docker run -v $(pwd)/datasets:/app/datasets -v $(pwd)/results:/app/results -v $(pwd)/figures:/app/figures har-project
```

## Dataset Structure

Place your dataset in the `datasets/standardized_balanced/{dataset_name}/` directory with the following structure:
- `train.csv`
- `validation.csv`
- `test.csv`

Each CSV file should contain the following columns:
- Accelerometer data (x, y, z)
- Gyroscope data (x, y, z)
- Activity labels ("standard activity code")

Supported activities:
- 0: Sitting
- 1: Standing
- 2: Walking
- 3: Walking Upstairs
- 4: Walking Downstairs
- 5: Jogging
    
## Usage

Run the main script:
```bash
python src/main.py
```

This will:
1. Load and preprocess the data
2. Generate visualization charts for sensor data
3. Train and evaluate multiple models
4. Save results and identify the best performing model

## Output

- Time series visualizations will be saved in the `figures/` directory
- Model evaluation results will be saved in `results/results.csv`
- The best performing model for each dataset will be printed to console

## Model Configurations

The project evaluates models with different combinations of:
- Data domains: Time and Frequency
- Dimension reduction: UMAP with various components (12, 24, or None)
- Feature extraction methods
- Multiple machine learning algorithms

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

[Add your chosen license here]


