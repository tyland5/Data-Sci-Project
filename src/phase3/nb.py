import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from preprocess import dataset
import pickle


def show_nb():
    ds = dataset
    df = ds.preprocessed_df
    st.title('Skills and Industry vs Above Avg Salaries')

    industries = ds.industry_encoder.inverse_transform(np.unique(df['industry'].values))
    selected_industry = st.selectbox("Industry", industries)

    skills = ['ACCT', 'ADM', 'ADVR', 'ANLS', 'ART', 'BD', 'CNSL', 'CUST', 'DIST', 'DSGN', 'EDU', 'ENG', 'FIN', 'GENB', 
        'HCPR', 'HR', 'IT', 'LGL', 'MGMT', 'MNFC', 'MRKT', 'OTHR', 'PR', 'PRCH', 'PRDM', 'PRJM', 'PROD', 'QA', 'RSCH', 'SALE', 'SCI', 'STRA', 'SUPL', 'TRNG', 'WRT']

    selected_skills = st.multiselect('Skills required', skills)

    btn_clicked = st.button('Generate')

    # when you click on generate button, run naive bayes algorithm and show visual
    if btn_clicked:

        # load in the naive bayes models. min, med, max
        nb_classifier, nb_classifier2, nb_classifier3 = pickle.load(open('./models/nb.p', 'rb'))
        
        # create skills array and formatted input sample
        encoded_skills = [0] * 35
        for skill in selected_skills:
            index = skills.index(skill)
            encoded_skills[index] = 1

        sample = encoded_skills + [ds.industry_encoder.transform([selected_industry])[0]] 
        
        # get predictions so they can be shown on page
        res_min = nb_classifier.predict([sample])
        res_med = nb_classifier2.predict([sample])
        res_max = nb_classifier3.predict([sample])

        col1, col2, col3 = st.columns(3)
        col1.metric("Above average minimum salary", 'True' if res_min[0] else 'False')
        col2.metric("Above average median salary", 'True' if res_med[0] else 'False')
        col3.metric("Above average maximum salary", 'True' if res_max[0] else 'False')
    