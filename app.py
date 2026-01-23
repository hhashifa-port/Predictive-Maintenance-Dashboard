
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# page configuration
st.set_page_config(page_title = 'Machine Health Dashboard', layout = 'wide')

# load assets (scaler and models)
@st.cache_resource
def load_assets():
  scaler = joblib.load('preprocessor.pkl')
  binary_model = joblib.load('best_random_forest.pkl')    # binary model  : 0/1
  multiclass_model = joblib.load('multi_rf_model.pkl')    # multiclass    : failure type
  return scaler, binary_model, multiclass_model

try:
  scaler, binary_model, multiclass_model = load_assets()
  multi_classes = multiclass_model.classes_
except Exception as e:
  st.error(f'Error loading assets: {e}')
  st.stop()

# header
st.title('Machine Failure Diagnostic Dashboard')
st.markdown('This dashboard predicts machine failure risks based on real-time sensor data.')
st.divider()

# sidebar: data sensor input
st.sidebar.header('Data Sensor Input')

air_temp = st.sidebar.slider('Air temperature', 295.0, 305.0, 300.0)
proc_temp = st.sidebar.slider('Process temperature', 304.0, 315.0, 310.0)
rpm = st.sidebar.number_input('Rotational speed', 1000, 3000, 1500)
torque = st.sidebar.number_input('Torque', 3.0, 80.0, 40.0)
tool_wear = st.sidebar.number_input('Tool wear', 0, 250, 100)
type_m = st.sidebar.selectbox('Type', ['L', 'M', 'H'])

st.sidebar.divider()
st.sidebar.subheader('Sensitivity Settings')
# threshold for binary
threshold = st.sidebar.slider('Failure Detection Threshold', 0.1, 0.9, 0.4)

# feature engineering
raw_input = pd.DataFrame([{
    'Type' : type_m,
    'Air temperature': air_temp,
    'Process temperature': proc_temp,
    'Rotational speed': rpm,
    'Torque': torque,
    'Tool wear': tool_wear,
    'Delta T' : proc_temp - air_temp,
    'Power' : torque * (rpm * (2 * np.pi / 60)),
    'Strain' : torque * tool_wear
}])

# make sure the column order match
cols_order = ['Type', 'Air temperature', 'Process temperature',
              'Rotational speed', 'Torque', 'Tool wear',
              'Delta T', 'Power', 'Strain']

raw_input = raw_input[cols_order]

if st.sidebar.button('Run Diagnostics'):

  # feature transformation with Scaler
  with_robust = ['Rotational speed', 'Torque', 'Power', 'Strain']
  with_standard = ['Air temperature', 'Process temperature', 'Delta T', 'Tool wear']
  with_onehot = ['Type']

  input_scaled = scaler.transform(raw_input)

  try:
    after_onehot = list(scaler.named_transformers_['onehot'].get_feature_names_out(['Type']))
  except:
    after_onehot = ['Type_L', 'Type_M']

  col_names = with_standard + with_robust + after_onehot
  data_after_scaled = pd.DataFrame(input_scaled, columns = col_names)

  # feature selection
  # feature_used = ['Delta T', 'Rotational speed', 'Power', 'Strain', 'Type_L', 'Type_M']
  feature_used = ['Air temperature', 'Process temperature', 'Delta T', 'Tool wear', 'Rotational speed', 'Torque', 'Power', 'Strain', 'Type_L', 'Type_M']
  final_data = data_after_scaled[feature_used]

  # binary prediction
  prob_binary = binary_model.predict_proba(final_data)[0][1]
  col1, col2 = st.columns(2)

  if prob_binary < threshold:
    with col1:
      st.subheader('Diagnostic Results')
      st.success('Machine Status: Normal')
      st.metric('Risk Level', f'{prob_binary*100:.2f}%', delta = f'Below Threshold', delta_color = 'inverse')

    with col2:
      st.subheader('Actionable Insights')
      st.write('**Action:** Continue routine monitoring as per maintenance schedule.')

  else:
    # if failure == 1, use Multiclass model
    multi_probs = multiclass_model.predict_proba(final_data)[0]

    # filtering 'No Filter' values
    prob_map = dict(zip(multi_classes, multi_probs))
    prob_map.pop('No Failure', None)

    # re-calculate max probability
    filtered_classes = list(prob_map.keys())
    filtered_probs = np.array(list(prob_map.values()))
    multi_prediction = filtered_classes[np.argmax(filtered_probs)]

    with col1:
      st.subheader('Diagnostic Results')
      st.error(f'Potential Failure Detected: {multi_prediction}')
      st.metric('Risk Level', f'{prob_binary*100:.2f}%', delta = f'Above Threshold', delta_color = 'inverse')

    with col2:
      st.subheader('Actionable Insights')
      if multi_prediction == 'HDF':
        st.write('**Action:** Inspect cooling system. Thermal differential is too narrow.')
      elif multi_prediction == 'PWF':
        st.write('**Action:** Check torque load or electrical power stability.')
      elif multi_prediction == 'OSF':
        st.write('**Action:** Replace tool immediately due to excessive structural strain.')
      else:
        st.write('**Action:** Continue routine monitoring as per maintenance schedule.')

    # probablity visualization
    if prob_binary >= threshold:
      st.divider()
      st.write('**Failure Type Probability Distribution**')

      prob_df = pd.DataFrame({
          'Failure Type': filtered_classes,
          'Probability (%)': filtered_probs * 100
          }).set_index('Failure Type')

      st.bar_chart(prob_df)

    else:
      st.divider()
      st.info('Probability distribution visualization is only available when a **failure is detected**.')
