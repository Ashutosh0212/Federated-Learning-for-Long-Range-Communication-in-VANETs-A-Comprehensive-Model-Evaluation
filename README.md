# 🚗 VANET Federated Learning Project

## Project Title
**Federated Learning for Long-Range Communication in VANETs: A Comprehensive Model Evaluation**

## Contributors
- **Ashutosh Gupta** - M23CSE009
- **Pinaq Sharma** - M23EET007
- **Ashutosh Singh Baghel** - M23EET008
- **Shreshth Vatsal Sharma** - B21CS094

---

## 📄 Project Description
This project implements a Federated Learning framework using Flower (`flwr`) for predicting vehicular pollution based on OBD-II data from vehicles in a VANET (Vehicular Ad Hoc Network) setup. The models evaluated include:

1. **Simple LSTM (Long Short-Term Memory)**
2. **Basic CNN (Convolutional Neural Network)**
3. **Simple Hybrid (CNN + RNN)**
4. **Enhanced GRU (Gated Recurrent Unit)**

The goal is to compare the performance of these models for long-range communication and real-time predictions in VANETs, while maintaining data privacy using federated learning.

---

## ⚙️ Features Used
The following OBD-II sensor features were selected for predicting CO₂ emissions:

- **OBD_Engine_Load**: Engine load percentage, indicating power demand.
- **OBD_Engine_RPM**: Engine revolutions per minute (RPM), showing engine speed.
- **OBD_KPL_Instant**: Instant fuel consumption, linked to vehicle efficiency.
- **OBD_Fuel_Flow_CCmin**: Fuel flow rate, providing a measure of fuel usage.
- **OBD_Air_Pedal**: Accelerator pedal position, indicating driver input.
- **OBD_Engine_Coolant_Temp_C**: Engine temperature, affecting combustion efficiency.
- **OBD_Intake_Air_Temp_C**: Intake air temperature, influencing air-fuel mixture.

**Target Variable**:
- **OBD_CO2_gkm_Instant**: Instant CO₂ emissions in grams per kilometer.

---

## 🛠️ Installation and Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Virtual Environment (venv)

### Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Data Preprocessing
Before running the main federated learning script, preprocess the OBD-II data to generate the necessary `.npy` files.

Run the following script:
```bash
python obd_data_preprocess.py
```

This script will:
- Load the raw OBD-II data.
- Extract and normalize the selected features.
- Save the preprocessed features (`preprocessed_X.npy`) and target (`preprocessed_y.npy`) files.
- Store the scaler parameters in `scaler_params.json` for consistency during prediction.

Ensure that the following files are generated in the root directory:
- `preprocessed_X.npy`
- `preprocessed_y.npy`
- `scaler_params.json`

### Step 2: Run the Federated Learning Script
To start the federated learning process, run the main script:
```bash
python one_for_all.py
```

This script will:
- Load the preprocessed data and split it among three clients.
- Initialize the Flower server and clients internally (no separate server/client scripts needed).
- Train the selected models (Simple LSTM, Basic CNN, Simple Hybrid, Enhanced GRU) in a federated learning setup.

---

## 📊 Viewing the Results

### Console Output
The training metrics (loss and MAE) for each model are displayed on the console after each federated learning round.

### Graphical Analysis
The script also generates the following plots:
1. **Training Loss Trend**: Shows the reduction in training loss across epochs.
2. **Validation Loss Trend**: Illustrates the decrease in validation loss.
3. **Validation MAE Trend**: Provides insights into the model's prediction accuracy.

To view the plots, check the output from the `plot_metrics()` function.

---

## 📂 Project Structure
```
vanet-federated-learning/
├── data/                     # Raw and preprocessed data files
├── logs/                     # Training and evaluation logs
├── plots/                    # Generated plots for results
├── obd_data_preprocess.py    # Data preprocessing script
├── one_for_all.py            # Main federated learning script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation                   # Model definitions (LSTM, CNN, Hybrid, GRU)
├── simple_lstm_model.py
├── basic_cnn_model.py
├── simple_hybrid_model.py
|── simple_gru_model.py
├── preprocessed_X.npy        # Preprocessed feature data
├── preprocessed_y.npy        # Preprocessed target data
└── scaler_params.json        # Scaler parameters for data normalization
```

---

## 📝 Notes
- The dataset used in this project is the **OBD-II Vehicular Data** from Belo Horizonte, Brazil.
- The data was anonymized to protect driver privacy, and only relevant features were selected for pollution prediction.

---

## 🔮 Future Work
- Implement advanced models like Transformers for enhanced sequence learning.
- Integrate privacy-preserving techniques like differential privacy in the federated setup.
- Test the models in real-world VANET environments with heterogeneous devices.

---

## 📧 Contact
For any questions or issues, please contact:

- Ashutosh Gupta: ashutoshgupta@iitj.ac.in


Enjoy experimenting and happy learning! 🚀