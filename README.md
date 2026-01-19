# Predictive Maintenance Diagnostic System
This project delivers an end-to-end Machine Learning solution to predict and diagnose industrial machine failures. Using the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), the system classifies specific failure types, enabling proactive maintenance and reducing operational downtime.

## Problem Statement
Unplanned machine failures are a significant cost driver in manufacturing. The challenges addressed in this project include:
- Early Detection: Predicting failure before it occurs to prevent secondary damage.
- Multiclass Classification: Distinguishing between specific failure types (HDF, PWF, OSF, TWF) rather than just a binary "Failure/No Failure".
- Data Imbalance: Handling the extreme scarcity of failure cases compared to normal operation (only ~3.39% of the data contains failures).

## Exploratory Data Analysis
Key findings from the sensor data (feature analysis):
- Thermal Relationships: Air Temperature and Process Temperature are highly correlated. The difference between them (Delta T) is a primary indicator of Heat Dissipation Failure (HDF).
- Operational Strain: Failures are often preceded by spikes in Torque or high Tool Wear duration.
- Class Imbalance: Visual analysis confirmed that "No Failure" is the majority class, necessitating specialized sampling techniques like SMOTE.

## Feature Engineering
To capture the underlying physics of the machine, the following features were engineered:
- Delta T: Process Temperature - Air Temperature.
- Power: Torque × Rotational Speed.
- Strain: Tool Wear × Torque.
These engineered features showed higher importance in the final model than raw sensor inputs.

## Modeling and Evaluation
- Algorithm: Random Forest Classifier (Multi-output/Multiclass).
- Imbalance Handling: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training set to ensure the model learns the characteristics of rare failure modes.
- Performance: The model achieved high precision and recall across failure types, significantly outperforming a baseline model without SMOTE.

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
- Python: Data processing and modeling (Pandas, NumPy, Scikit-Learn).
- Imbalanced-learn: SMOTE for data balancing.
- Streamlit: Interactive web dashboard for real-time diagnostics.
- Joblib: Model serialization and deployment.

## How to Run
- Clone this repository.
- Install dependencies: pip install -r requirements.txt.
- Run the dashboard: streamlit run app.py.
