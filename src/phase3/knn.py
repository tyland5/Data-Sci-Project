import streamlit as st
import pandas as pd
import numpy as np
from preprocess import dataset
from sklearn.preprocessing import MinMaxScaler
import pickle

def show_knn():
    ds = dataset
    st.title('Predict industry (KNN)')

    selected_min_salary = st.number_input("Minimum annual salary", value=None, placeholder="Enter an integer...")
    selected_med_salary = st.number_input("Median annual salary", value=None, placeholder="Enter an integer...")
    selected_max_salary = st.number_input("Maximum annual salary", value=None, placeholder="Enter an integer...")

    experience_lvls = ['Entry level', 'Associate', 'Mid-Senior level', 'Management']
    selected_exp_lvl = st.selectbox("Experience level", experience_lvls)
    # encode the experience level based on the index
    selected_exp_lvl = experience_lvls.index(selected_exp_lvl)

    selected_emp_count = st.number_input("Employee count", value=None, placeholder="Enter an integer...")

    company_size = [1, 2, 3, 4, 5, 6, 7]
    selected_company_size = st.selectbox("Company size", company_size)

    btn_clicked = st.button('Predict')

    # if buttun is click it will preprocess the information that is given and return the prediction
    if btn_clicked:
        knn = pickle.load(open('./models/knn.p', 'rb'))

        # have the empoyee count into bins
        mod_emp_count = selected_emp_count // 10000 if selected_emp_count < 100000 else 10

        # convert the salary in ten thousands
        scale_min_salary = selected_min_salary / 10000
        scale_med_salary = selected_med_salary / 10000
        scale_max_salary = selected_max_salary / 10000

        feature = [[scale_max_salary] +
                   [scale_med_salary] + 
                   [scale_min_salary] + 
                   [selected_company_size]+ 
                   [mod_emp_count] + 
                   [selected_exp_lvl]]
        feature = np.array(feature).reshape(1, -1)

        predict = knn.predict(feature)

        st.metric("Industry", ds.industry_encoder.inverse_transform(predict)[0])