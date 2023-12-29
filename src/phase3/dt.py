import streamlit as st
import numpy as np
from preprocess import dataset
import pickle
from fuzzywuzzy import process
from colorama import Fore, Back, Style

def show_dt():
  ds = dataset
  df = ds.preprocessed_df
  st.title('Estimate required experience level (DT)')

  industries = ds.industry_encoder.inverse_transform(np.unique(df['industry'].values))
  selected_industry = st.selectbox("Industry", industries)

  selected_emp_count = st.number_input("Company employee count", value=None, placeholder="Enter an integer...")

  selected_med_salary = st.number_input("Annual salary", value=None, placeholder="Enter an integer...")

  selected_title = st.text_input("Job title", value=None, max_chars=None, key=None, type='default')

  # User input might not match the titles in the dataset, so we need to find the closest match
  if selected_title:
    title_list = ds.title_encoder.classes_
    matched_title = process.extractBests(selected_title, title_list, score_cutoff=50, limit=10)
    print(Fore.GREEN + f'[DEBUG] Input title: {selected_title} \t Matched titles: {matched_title}' + Style.RESET_ALL) # DEBUG
    selected_title = st.selectbox("Closest matching job titles", [title[0] for title in matched_title])

  btn_clicked = st.button('Predict')

  # Run DT model and return predictions
  if btn_clicked:
    dt_model = pickle.load(open('./models/dt.pkl', 'rb'))
    exp_levels = ['Entry level', 'Associate', 'Mid-Senior level', 'Management']

    # Bin employee count into 10k bins
    mod_emp_count = selected_emp_count // 10000 if selected_emp_count < 100000 else 10

    dt_sample = [[ds.industry_encoder.transform([selected_industry])[0]] +
                  [mod_emp_count] +
                  [selected_med_salary] + 
                  [ds.title_encoder.transform([selected_title])[0]]]
    
    dt_sample = np.array(dt_sample).reshape(1, -1)
    dt_res = dt_model.predict(dt_sample)

    print(Fore.GREEN + f'[DEBUG] (DT) Prediction: {dt_res}' + Style.RESET_ALL) # DEBUG
    st.metric(label="Estimated experience level", value=exp_levels[dt_res[0]])