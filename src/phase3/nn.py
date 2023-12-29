import streamlit as st
import numpy as np
from tensorflow import keras
from preprocess import dataset
import pickle
from colorama import Fore, Back, Style

def show_nn():
  ds = dataset
  df = ds.preprocessed_df
  st.title('Predict salaries (NN)')

  industries = ds.industry_encoder.inverse_transform(np.unique(df['industry'].values))
  selected_industry = st.selectbox("Industry", industries)

  skills = ['ACCT', 'ADM', 'ADVR', 'ANLS', 'ART', 'BD', 'CNSL', 'CUST', 'DIST', 'DSGN', 'EDU', 'ENG', 'FIN', 'GENB', 
    'HCPR', 'HR', 'IT', 'LGL', 'MGMT', 'MNFC', 'MRKT', 'OTHR', 'PR', 'PRCH', 'PRDM', 'PRJM', 'PROD', 'QA', 'RSCH', 'SALE', 'SCI', 'STRA', 'SUPL', 'TRNG', 'WRT']
  selected_skl = st.multiselect('Skills required', skills)

  work_types = ['Internship', 'Part-time', 'Contract', 'Full-time']
  selected_work_type = st.selectbox("Work type", work_types)

  locations = ds.loc_encoder.inverse_transform(np.unique(df['location'].values))
  selected_loc = st.selectbox("Location", locations)

  exp_levels = ['Entry level', 'Associate', 'Mid-Senior level', 'Management']
  selected_exp = st.selectbox("Experience level", exp_levels)

  selected_emp_count = st.number_input("Employee count", value=None, placeholder="Enter an integer...")

  btn_clicked = st.button('Predict')

  # Run NN model and return predictions
  if btn_clicked:
    nn_model = keras.models.load_model('./models/nn.h5', compile=False)

    # Need naive bayes model for the above avg columns
    nb_1, nb_2, nb_3 = pickle.load(open('./models/nb.p', 'rb'))

    # Encoded selected skills
    encoded_skills = [0] * 35
    for skill in selected_skl:
      index = skills.index(skill)
      encoded_skills[index] = 1

    # Get naive bayes predictions for above avg columns
    nb_sample = encoded_skills + [ds.industry_encoder.transform([selected_industry])[0]] 
    nb_res_min = nb_1.predict([nb_sample])
    nb_res_med = nb_2.predict([nb_sample])
    nb_res_max = nb_3.predict([nb_sample])

    print(Fore.GREEN + f'[DEBUG] (NB) {nb_res_min} {nb_res_med} {nb_res_max}' + Style.RESET_ALL) # DEBUG

    # Bin employee count into 10k bins
    mod_emp_count = selected_emp_count // 10000 if selected_emp_count < 100000 else 10

    nn_sample = [[work_types.index(selected_work_type)] + 
                  [ds.loc_encoder.transform([selected_loc])[0]] + 
                  [ds.industry_encoder.transform([selected_industry])[0]] +
                  [mod_emp_count] +
                  [exp_levels.index(selected_exp)]+ 
                  encoded_skills + 
                  [nb_res_min[0]] + [nb_res_med[0]] + [nb_res_max[0]]]
    
    nn_sample = np.array(nn_sample).reshape(1, -1)
    
    print(Fore.GREEN + f'[DEBUG] (NN) {nn_sample}' + Style.RESET_ALL) # DEBUG

    nn_res = nn_model.predict(nn_sample)
    print(Fore.GREEN + f'[DEBUG] (NN) {nn_res}' + Style.RESET_ALL) # DEBUG

    col1, col2, col3 = st.columns(3)
    col1.metric("Minimum salary", f'${round(nn_res[0][0] * 10000, 2)}')
    col2.metric("Median salary", f'${round(nn_res[0][1] * 10000, 2)}')
    col3.metric("Maximum salary", f'${round(nn_res[0][2] * 10000, 2)}')

