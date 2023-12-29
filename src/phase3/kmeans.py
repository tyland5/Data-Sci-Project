import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from preprocess import dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle

def show_kmeans():
    ds = dataset
    df = ds.preprocessed_df
    st.title('Seeing similar paying jobs')
    
    titles = ds.title_encoder.inverse_transform(np.unique(df['title'].values))
    selected_title = st.selectbox("Job title", titles)

    work_types = ['Internship', 'Part-time', 'Contract', 'Full-time'] 
    selected_work_type = st.selectbox("Work type", work_types)

    states = ds.loc_encoder.inverse_transform(np.unique(df['location'].values))
    selected_state = st.selectbox("State", states)

    experience_lvls = ['Entry level', 'Associate', 'Mid-Senior level', 'Management']
    selected_exp_lvl = st.selectbox("Experience level", experience_lvls)

    selected_emp_count = st.number_input("Employee count", value=None, placeholder="Enter an integer...")
    
    industries = ds.industry_encoder.inverse_transform(np.unique(df['industry'].values))
    selected_industry = st.selectbox("Industry", industries)

    selected_min_sal= st.number_input("Minimum annual salary", value=None, placeholder="Enter an integer...")
    selected_med_sal= st.number_input("Median annual salary", value=None, placeholder="Enter an integer...")
    selected_max_sal= st.number_input("Maximum annual salary", value=None, placeholder="Enter an integer...")

    btn_clicked = st.button('Generate')

    
    if btn_clicked: 

        # get scaler to scale the sample's columns other than salaries just like when training
        scaler = MinMaxScaler()
        original_features = df.loc[:,'title':'industry'].drop(columns=['min_salary', 'max_salary', 'med_salary'])
        scaler.fit(original_features.loc[:, 'title':'industry'])

        # define the centroids and clusters
        kmeans = pickle.load(open('./models/kmeans.p', 'rb'))
        clusters = np.load('./models/clusters.npy', allow_pickle=True)
    
        # make sure to encode and scale the necessary fields of input sample
        mod_employee_count = selected_emp_count // 10000 if selected_emp_count < 100000 else 10 
        mod_employee_count 
        sample = scaler.transform([[ ds.title_encoder.transform([selected_title])[0], work_types.index(selected_work_type), ds.loc_encoder.transform([selected_state])[0], 
                    experience_lvls.index(selected_exp_lvl), mod_employee_count, ds.industry_encoder.transform([selected_industry])[0] ]])
        sample = [list(np.append(sample, [selected_min_sal/10000, selected_med_sal/10000, selected_max_sal/10000 ]))]
       
        # get the proper cluster and put the samples into a dataframe to be displayed
        target_centroid = kmeans.predict(sample)[0] 
        returned_df = clusters[target_centroid]
        returned_df = pd.DataFrame(np.array(returned_df), columns=['Job Title', 'Work Type', 'State', 'Experience Level', 'Employee Count', 
                                                                   'Industry', 'Minimum Salary', 'Median Salary', 'Maximum Salary'])

        # get the real values from encodings. for some reason, pd.dataframe turns int indices to float
        returned_df['Job Title'] = ds.title_encoder.inverse_transform(returned_df['Job Title'].astype(int).values)
        returned_df['Work Type'] = [work_types[int(index)] for index in returned_df['Work Type'].values]
        returned_df['State'] = ds.loc_encoder.inverse_transform(returned_df['State'].astype(int).values)
        returned_df['Experience Level'] = [experience_lvls[int(index)] for index in returned_df['Experience Level'].values]
        returned_df['Employee Count'] = [str(count * 10000) if count < 10 and count > 0 else ('Less than 10000' if count < 10 else '100000+')for count in returned_df['Employee Count'].values]
        returned_df['Industry'] = ds.industry_encoder.inverse_transform(returned_df['Industry'].astype(int).values)
        returned_df['Minimum Salary'] = [min_sal * 10000 for min_sal in returned_df['Minimum Salary'].values]
        returned_df['Median Salary'] = [med_sal * 10000 for med_sal in returned_df['Median Salary'].values]
        returned_df['Maximum Salary'] = [max_sal * 10000 for max_sal in returned_df['Maximum Salary'].values]
        
        st.dataframe(returned_df)