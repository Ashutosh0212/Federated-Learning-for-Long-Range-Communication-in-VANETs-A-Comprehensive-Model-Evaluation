# File: preprocess_obd.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json


def validate_obd_data(df, features, target):
    """Validate OBD data for completeness and quality"""
    missing_cols = [col for col in features +
                    [target] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing OBD columns: {missing_cols}")

    # Check for reasonable value ranges
    validations = {
        'OBD_Engine_Load': (0, 100),         # percentage
        'OBD_Engine_RPM': (0, 8000),         # typical RPM range
        'OBD_KPL_Instant': (0, 50),          # reasonable KPL range
        'OBD_Engine_Coolant_Temp_C': (0, 120)  # typical temperature range
    }

    for col, (min_val, max_val) in validations.items():
        if col in df.columns:
            invalid_count = df[col][(df[col] < min_val)
                                    | (df[col] > max_val)].count()
            if invalid_count > 0:
                print(f"Warning: {
                      invalid_count} invalid values found in {col}")


def main():
    # Load the dataset
    file_path = 'VehicularData(anonymized) (1).csv'
    data = pd.read_csv(file_path)

    # Replace infinite values with NaN and fill them
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # Select only OBD-related features for pollution prediction
    features = [
        'OBD_Engine_Load',           # Engine load percentage
        'OBD_Engine_RPM',            # Engine RPM
        'OBD_KPL_Instant',           # Instant fuel consumption
        'OBD_Fuel_Flow_CCmin',       # Fuel flow rate
        'OBD_Air_Pedal',             # Accelerator pedal position
        'OBD_Engine_Coolant_Temp_C',  # Engine temperature
        'OBD_Intake_Air_Temp_C'      # Intake air temperature
    ]
    target = 'OBD_CO2_gkm_Instant'

    # Validate the data
    validate_obd_data(data, features, target)

    # Print basic statistics
    print("\nFeature Statistics:")
    print(data[features + [target]].describe())

    # Scale features to [0, 1] for LSTM input
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features + [target]])

    # Save scaler parameters for later use
    scaler_params = {
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist(),
        'data_range_': scaler.data_range_.tolist(),
        'feature_names': features + [target]
    }

    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)

    # Prepare sequences for time series prediction
    sequence_length = 20  # Increased for better temporal patterns
    X, y = [], []

    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length, :-1])
        y.append(data_scaled[i+sequence_length, -1])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Save the preprocessed data
    np.save('preprocessed_X.npy', X)
    np.save('preprocessed_y.npy', y)

    print(f"\nPreprocessing completed:")
    print(f"Input shape (X): {X.shape}")
    print(f"Target shape (y): {y.shape}")
    print("\nFiles saved: preprocessed_X.npy, preprocessed_y.npy, scaler_params.json")


if __name__ == "__main__":
    main()
