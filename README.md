# Predictive Maintenance Diagnostic System
This project delivers an end-to-end Machine Learning solution to predict and diagnose industrial machine failures. Using the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), the system classifies specific failure types, enabling proactive maintenance and reducing operational downtime.

## Live Demo
You can access the interactive dashboard here: [Predictive-Maintenance-Dashboard](https://predictive-maintenance-dashboard-atulzfmv2ckqzqclgubmd5.streamlit.app/) <br>
Or you can simulate this project in colab: [Open in Colab](https://colab.research.google.com/drive/1NU4VwO8CnGITqupOTythkP3khHhjQ3lX#scrollTo=IokAQ7MVm2WI)

## Problem Statement
Unplanned machine failures are a significant cost driver in manufacturing. The challenges addressed in this project include:
- Early Detection: Predicting failure before it occurs to prevent secondary damage.
- Multiclass Classification: Distinguishing between specific failure types (HDF, PWF, OSF, TWF) rather than just a binary "Failure/No Failure".
- Data Imbalance: Handling the extreme scarcity of failure cases compared to normal operation (only ~3.39% of the data contains failures).

## Exploratory Data Analysis
Key findings from the sensor data (feature analysis):
- Thermal Relationships: <br>
Air Temperature and Process Temperature are highly correlated. <br>
The difference between them (Delta T) is a primary indicator of Heat Dissipation Failure (HDF).
- Operational Strain: <br>
Failures are often preceded by spikes in Torque or high Tool Wear duration.

## Feature Engineering
To capture the underlying physics of the machine, the following features were engineered:
- Delta T: Process Temperature - Air Temperature.
- Power: Torque × Rotational Speed.
- Strain: Tool Wear × Torque.
These engineered features showed higher importance in the final model than raw sensor inputs.

## Modeling and Evaluation
To ensure high accuracy and operational efficiency, the diagnostic system follows a sequential two-stage prediction pipeline:
1. Stage 1: Binary Detection (The "Gatekeeper")
   - The data is first processed by a Binary Random Forest model.
   - Goal: To determine if the machine is in a Normal state or a Failure state.
   - This stage is optimized for Recall, ensuring that no potential failures are missed.
2. Stage 2: Multi-Diagnostic Classification
   - Only if Stage 1 detects a "Failure", the data is passed to the Multiclass Random Forest model.
   - Goal: To diagnose the specific root cause of the failure (e.g., HDF, PWF, OSF, or TWF).

## Actionable Insights
The system doesn't just predict; it prescribes. The integrated dashboard provides the following logic:
<center>
  
|Failure Type|Diagnosis|Recommended Action|
|:---|:---|:---|
|HDF|Poor heat dissipation.|Inspect cooling systems and clean air filters|
|PWF|Power/Torque instability.|Check torque load or electrical power supply stability.|
|OSF|Excessive structural strain.|Reduce torque load or replace the cutting tool immediately.|
|TWF|Tool wear limit reached.|Schedule routine tool replacement based on wear minutes.|

</center>

## Tech Stack
- Language: Python (Pandas, NumPy, Scikit-Learn)
- Imbalanced-learn: SMOTE for data balancing.
- Deployment: Streamlit, GitHub, Joblib.

## How to Run
- Clone this repository.
- Install dependencies: `pip install -r requirements.txt`.
- Run the dashboard: `streamlit run app.py`.
