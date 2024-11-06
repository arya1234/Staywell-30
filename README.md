# StayWell 30: Early Readmission Prediction (ERP) Model

The **StayWell 30** ERP Model is a machine learning project designed to predict the likelihood of a patient being readmitted to the hospital within 30 days after discharge. This model leverages patient data, including demographic information, medical history, medication details, and lab results, to provide healthcare providers with insights that can lead to proactive interventions, ultimately reducing readmission rates and improving patient outcomes.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Data](#data)
- [Model](#model)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)

## Problem Statement

Hospital readmissions are costly and can pose additional health risks for patients. Accurately predicting readmissions enables healthcare providers to intervene with targeted care and resources, reducing both costs and patient risk. The StayWell 30 ERP Model addresses this problem by predicting the likelihood of readmission, supporting early interventions to improve patient outcomes.

## Data

The StayWell 30 Model utilizes a dataset collected from electronic health records (EHRs) or healthcare databases. The dataset includes the following types of information:
- **Demographics**: Age, gender, etc.
- **Medical History**: Diagnoses, prior hospitalizations, etc.
- **Medication History**: List of current medications.
- **Lab Results**: Relevant lab measurements.

### Data Preprocessing
The data has been preprocessed to ensure it is ready for modeling, including:
- **Cleaning**: Removing or imputing missing values.
- **Encoding**: Converting categorical variables to numeric representations.
- **Normalization**: Scaling numeric variables for optimal model performance.

## Model

The StayWell 30 ERP Model employs a machine learning algorithm to predict the likelihood of readmission within 30 days of discharge. The model pipeline includes:
1. **Training**: The model is trained on a subset of patient data.
2. **Tuning**: Hyperparameters are tuned to enhance performance.
3. **Evaluation**: The model's accuracy, precision, recall, and F1 score are measured using a testing set.
4. **Deployment**: Once validated, the model is deployed to predict readmission risk for new patients.

## Folder Structure

The project is organized as follows:

```
.StayWell-30/
├── .vscode/                 # VS Code settings
├── data/                    # Raw data files
├── processed_data/          # Preprocessed data ready for modeling
├── models/                  # Saved models for predictions
├── model_trainer/           # Scripts for training the model
├── testing/                 # Test cases and validation data (e.g., JSON files)
├── LICENSE                  # License for the project
├── README.md                # Project documentation
├── main.py                  # Main script to run predictions
├── requirements.txt         # Required packages
```

## Installation

To install and set up the StayWell 30 Model:

1. Clone the repository:
   ```
   git clone https://github.com/arya1234/Staywell-30.git
   cd Staywell-30
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the StayWell 30 Model, import the `erp_model` module and call the `predict` function with a patient's data:

```python
from erp_model import predict

# Sample patient data
patient_data = {
    'age': 60,
    'gender': 'M',
    'diagnosis': 'heart failure',
    'medications': ['furosemide', 'lisinopril'],
    'lab_results': {'sodium': 135, 'potassium': 4.0, 'creatinine': 1.2}
}

# Predict readmission probability
readmission_probability = predict(patient_data)

print(f"Readmission probability: {readmission_probability}")
```

The `predict` function returns a probability between 0 and 1, indicating the likelihood of readmission within 30 days of discharge.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
