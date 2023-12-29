import streamlit as st
import pandas as pd
import numpy as np
from preprocess import dataset
import pickle

def show_lr():
    ds = dataset
    df = ds.preprocessed_df
    st.title('Predict above avg income based on job (LR)')

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

    # if buttun is click it will preprocess the information that is given and return the prediction
    if btn_clicked:
        lr_min = pickle.load(open('./models/lr1.pkl', 'rb'))
        lr_med = pickle.load(open('./models/lr2.pkl', 'rb'))
        lr_max = pickle.load(open('./models/lr3.pkl', 'rb'))

        # have the empoyee count into bins
        mod_emp_count = selected_emp_count // 10000 if selected_emp_count < 100000 else 10

        # encode selected skills where 0 means it didnt select and 1 means it select
        encoded_skills = [0] * 35
        for skill in selected_skl:
            index = skills.index(skill)
            encoded_skills[index] = 1
        
        feature = [[work_types.index(selected_work_type)] +
                   [ds.loc_encoder.transform([selected_loc])[0]] + 
                   [exp_levels.index(selected_exp)] + 
                   [mod_emp_count] + 
                   [ds.industry_encoder.transform([selected_industry])[0]] + 
                   encoded_skills]
        
        feature = np.array(feature).reshape(1, -1)

        min_pred = lr_min.predict(feature)
        med_pred = lr_med.predict(feature)
        max_pred = lr_max.predict(feature)

        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum salary", 'True' if min_pred else 'False')
        col2.metric("Median salary", 'True' if med_pred else 'False')
        col3.metric("Maximum salary", 'True' if max_pred else 'False')
