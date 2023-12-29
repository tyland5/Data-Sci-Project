import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class dataset:
    loc_encoder = ''
    title_encoder = ''
    industry_encoder = ''
    preprocessed_df = ''

    def preprocess_dataset():
        df = pd.read_csv("./archive/job_postings.csv")
        columns_to_drop = ['expiry', 'skills_desc', 'posting_domain', 'compensation_type', 'sponsored', 'currency', 'listed_time',
                        'job_posting_url', 'application_url', 'application_type', 'closed_time', 'applies', 'views', 'remote_allowed',
                        'original_listed_time', 'description', 'work_type']

        # drop columns we know we don't want
        df = df.drop(columns=columns_to_drop)

        # joining the different data sets
        emp_count_df = pd.read_csv('./archive/employee_counts.csv')
        emp_count_df = emp_count_df[['company_id', 'employee_count']]
        emp_count_df = emp_count_df.drop_duplicates(subset = ['company_id'])

        comp_ind = pd.read_csv("./archive/company_industries.csv")
        comp_ind = comp_ind.drop_duplicates(subset = ['company_id'])

        comp_size = pd.read_csv("./archive/companies.csv")
        comp_size = comp_size[['company_id', 'company_size']]
        comp_size = comp_size.drop_duplicates(subset = ['company_id'])

        skills_abr = pd.read_csv("./archive/job_skills.csv")
        skills_abr = skills_abr[['skill_abr', 'job_id']]

        # One hot encode skills_abr for each job_id
        skills_abr = pd.get_dummies(skills_abr, columns=['skill_abr'])
        skills_abr = skills_abr.groupby('job_id').sum().reset_index()

        df = df.merge(emp_count_df, how='inner', on='company_id')
        df = df.merge(comp_ind,  how='inner', on='company_id')
        df = df.merge(comp_size,  how='inner', on='company_id')
        df = df.merge(skills_abr,  how='inner', on='job_id')

        # get rid of null experience level
        unique_levels = df['formatted_experience_level'].unique()[1:]
        df = df[df['formatted_experience_level'].isin(unique_levels)]

        #drop null salaries
        df = df.dropna(subset=['pay_period', 'min_salary', 'max_salary'])

        # making all pay the same rates
        df.loc[df['pay_period'] == 'HOURLY', 'max_salary'] = df['max_salary'] * 40 * 52
        df.loc[df['pay_period'] == 'WEEKLY', 'max_salary'] = df['max_salary'] * 52
        df.loc[df['pay_period'] == 'MONTHLY', 'max_salary'] = df['max_salary'] * 12

        df.loc[df['pay_period'] == 'HOURLY', 'min_salary'] = df['min_salary'] * 40 * 52
        df.loc[df['pay_period'] == 'WEEKLY', 'min_salary'] = df['min_salary'] * 52
        df.loc[df['pay_period'] == 'MONTHLY', 'min_salary'] = df['min_salary'] * 12

        df.loc[df['pay_period'] == 'HOURLY', 'med_salary'] = df['med_salary'] * 40 * 52
        df.loc[df['pay_period'] == 'WEEKLY', 'med_salary'] = df['med_salary'] * 52
        df.loc[df['pay_period'] == 'MONTHLY', 'med_salary'] = df['med_salary'] * 12

        # we know all salaries are annual, so we drop the pay_period column
        df = df.drop(columns=['pay_period'])

        # impute missing median salary now we have correct rates for min and max
        df.loc[df['med_salary'].isna(), 'med_salary'] = (df['max_salary'] + df['min_salary']) / 2

        # Drop outliers for min and max salary

        df = df[df['max_salary'] > 5000]
        df = df[df['max_salary'] <= 350000]

        df = df[df['min_salary'] > 5000]
        df = df[df['min_salary'] <= 350000]

        # Drop rows with null columns now that we have salary columns fixed
        df = df.dropna()

        # scaling the salary columns. they'll be in 10s of thousands 
        df["min_salary"] = df["min_salary"].apply(lambda sal: sal / 10000)
        df["max_salary"] = df["max_salary"].apply(lambda sal: sal / 10000)
        df["med_salary"] = df["med_salary"].apply(lambda sal: sal / 10000)

        # here we are trying to standardize the location column so we only get the abbreviated state name

        us_state_to_abbrev = {
            "Alabama": "AL",
            "Alaska": "AK",
            "Arizona": "AZ",
            "Arkansas": "AR",
            "California": "CA",
            "Colorado": "CO",
            "Connecticut": "CT",
            "Delaware": "DE",
            "Florida": "FL",
            "Georgia": "GA",
            "Hawaii": "HI",
            "Idaho": "ID",
            "Illinois": "IL",
            "Indiana": "IN",
            "Iowa": "IA",
            "Kansas": "KS",
            "Kentucky": "KY",
            "Louisiana": "LA",
            "Maine": "ME",
            "Maryland": "MD",
            "Massachusetts": "MA",
            "Michigan": "MI",
            "Minnesota": "MN",
            "Mississippi": "MS",
            "Missouri": "MO",
            "Montana": "MT",
            "Nebraska": "NE",
            "Nevada": "NV",
            "New Hampshire": "NH",
            "New Jersey": "NJ",
            "New Mexico": "NM",
            "New York": "NY",
            "North Carolina": "NC",
            "North Dakota": "ND",
            "Ohio": "OH",
            "Oklahoma": "OK",
            "Oregon": "OR",
            "Pennsylvania": "PA",
            "Rhode Island": "RI",
            "South Carolina": "SC",
            "South Dakota": "SD",
            "Tennessee": "TN",
            "Texas": "TX",
            "Utah": "UT",
            "Vermont": "VT",
            "Virginia": "VA",
            "Washington": "WA",
            "West Virginia": "WV",
            "Wisconsin": "WI",
            "Wyoming": "WY",
            "District of Columbia": "DC"}

        us_city_to_state= {
                'San': 'CA',
                'Atlanta': 'GA',
                'Grand Rapids': 'MI',
                'Detroit': 'MI',
                'Cincinnati': 'OH',
                'Denver': 'CO',
                'Los Angeles': 'CA',
                'Las Vegas': 'NV',
                'Seattle':'WA',
                'Miami-Fort': 'FL',
                'Chicago': 'IL',
                'Tampa': 'FL',
                'Salt Lake': 'UT',
                'Nashville': 'TN',
                'Buffalo-Niagara': 'NY',
                'Raleigh-Durham-Chapel': 'NC',
                'Huntsville-Decatur-Albertville':'AL',
                'Madison' : 'WI',
                'Minneapolis-St.': 'MN',
                'Dallas-Fort': 'TX',
                'Pittsburgh': 'PA',
                'Houston': 'TX',
                'Orlando': 'FL',
                'Memphis': 'TN',
                'Philadelphia': 'PA',
                'Phoenix': 'AZ',
                'AZ': 'AZ',
                'Charlotte': 'NC',
                'Cleveland':'OH'}

        locations = df['location'].values
        counter = 0
        counter2 = 0
        for i in range(len(locations)):
            components = locations[i].split(',')
            
            # looked like always "state, US"
            if len(components) > 1 and components[-1].strip() == 'United States':
                counter += 1
                locations[i] = us_state_to_abbrev[components[-2].strip()]
                continue
            
            # checks for city, state abbrev
            loc = components[-1].strip()
            if len(loc) == 2:
                locations[i] = loc
                continue
            
            
            location = components[-1].split()[0].strip()
            if location in us_state_to_abbrev.keys():
                counter2 +=1
                locations[i] = us_state_to_abbrev[location]
                continue  
            elif location in us_city_to_state.keys():
                counter2 +=1
                locations[i] = us_city_to_state[location]
                continue
            
            if len(components[-1].split()) > 1:
                location = components[-1].split()[0].strip() + ' ' + components[-1].split()[1].strip()
                if location in us_state_to_abbrev.keys():
                    counter2 +=1
                    locations[i] = us_state_to_abbrev[location]
                    continue
                elif location in us_city_to_state.keys():
                    counter2 +=1
                    locations[i] = us_city_to_state[location]
                    continue

                location = components[-1].split()[1].strip()
                if location in us_state_to_abbrev.keys():
                    counter2 +=1
                    locations[i] = us_state_to_abbrev[location]
                    continue
                elif location in us_city_to_state.keys():
                    counter2 +=1
                    locations[i] = us_city_to_state[location]
                    continue
                    
            locations[i] = 'other'

                
        unique_locations = np.unique(locations)
                
        df['location'] = locations

        # encoding location
        loc_encoder = LabelEncoder()
        loc_encoder.fit(unique_locations)
        locations = loc_encoder.transform(locations)
        df['location'] = locations
        dataset.loc_encoder = loc_encoder

        #encode title
        title_encoder = LabelEncoder()
        titles = df['title'].values
        unique_titles = np.unique(titles)
        title_encoder.fit(unique_titles)
        titles = title_encoder.transform(titles)
        df['title'] = titles
        dataset.title_encoder = title_encoder

        # fix formatted_experience_level column and ordinal encode
        # ['Associate' 'Mid-Senior level' 'Entry level' 'Executive' 'Director' 'Internship'] 
        # We want to combine Internship with entry, and Executive, Director into management
        experience_lvls = ['Entry level', 'Associate', 'Mid-Senior level', 'Management']
        fel_column = df['formatted_experience_level'].values
        for i in range(len(fel_column)):
            level = fel_column[i].strip()
            if level in experience_lvls:
                fel_column[i] = level
                continue
            
            if level == 'Internship':
                fel_column[i] = 'Entry level'
            else:
                fel_column[i] = 'Management'

        # set updated experience column
        df['formatted_experience_level'] = fel_column

        #ordinal encode
        for i in range(len(fel_column)):
            level = fel_column[i]
            fel_column[i] = experience_lvls.index(level)
        df['formatted_experience_level'] = fel_column


        # fix formatted_work_type column and ordinal encode
        # ['Full-time' 'Contract' 'Part-time' 'Temporary' 'Other' 'Internship']
        # we'll group temporary with part-time and other with contract
        work_type = ['Internship', 'Part-time', 'Contract', 'Full-time'] 
        work_type_col = df['formatted_work_type'].values

        for i in range(len(work_type_col)):
            type_ = work_type_col[i].strip()
            if type_ in work_type:
                work_type_col[i] = type_
                continue
            
            if type_ == 'Temporary':
                work_type_col[i] = 'Part-time'
            else:
                work_type_col[i] = 'Contract'

        # set updated experience column
        df['formatted_work_type'] = work_type_col

        #ordinal encode
        for i in range(len(work_type_col)):
            type_ = work_type_col[i]
            work_type_col[i] = work_type.index(type_)
        df['formatted_work_type'] = work_type_col

        # encode employee count column based off these groupings
        def interval(value):
            k = 1000
            if value < 10*k:
                return 0
            elif value >= 10*k and value < 20*k:
                return 1
            elif value >= 20*k and value < 30*k:
                return 2
            elif value >= 30*k and value < 40*k:
                return 3
            elif value >= 40*k and value < 50*k:
                return 4
            elif value >= 50*k and value < 60*k:
                return 5
            elif value >= 60*k and value < 70*k:
                return 6
            elif value >= 70*k and value < 80*k:
                return 7
            elif value >= 80*k and value < 90*k:
                return 8
            elif value >= 90*k and value < 100*k:
                return 9
            elif value >= 100*k:
                return 10

        df["employee_count"] = df["employee_count"].apply(interval)

        # create boolean column that specifies whether the respective salaries are above average
        above_avg_min = []
        above_avg_med = []
        above_avg_max = []

        min_ = df['min_salary'].values
        med_ = df['med_salary'].values
        max_ = df['max_salary'].values

        min_avg = np.mean(min_)
        max_avg = np.mean(max_)
        med_avg = np.mean(med_)


        for sal in min_:
            if sal >= min_avg:
                above_avg_min.append(1)
            else:
                above_avg_min.append(0)
                
        for sal in med_:
            if sal >= med_avg:
                above_avg_med.append(1)
            else:
                above_avg_med.append(0)
                
        for sal in max_:
            if sal >= max_avg:
                above_avg_max.append(1)
            else:
                above_avg_max.append(0)
                
        df['above_avg_min'] = above_avg_min
        df['above_avg_med'] = above_avg_med
        df['above_avg_max'] = above_avg_max

        unique_industries = list(df.groupby('industry')['industry'].count().index)
        industries = df['industry'].values
        # encode industries 
        industry_encoder = LabelEncoder()
        industry_encoder.fit(unique_industries)
        industries = industry_encoder.transform(industries)
        df['industry'] = industries
        dataset.industry_encoder = industry_encoder

        dataset.preprocessed_df = df