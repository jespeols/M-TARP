import os
import numpy as np
import pandas as pd

from pathlib import Path

# user-defined functions
from utils import country_code_to_name

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

def preprocess_TESSy(path,
                     pathogens: list,
                     save_path = None,
                     exclude_antibiotics: list = None,
                     impute_gender: bool = False,
                     impute_age: bool = False,
                     ):
    
    print(f"Reading in TESSy data from '{path}'...")
    TESSy_data = pd.read_csv(path, low_memory=False)
    if pathogens:
        print(f"Pathogens: {pathogens}")
        TESSy_data = TESSy_data[TESSy_data['Pathogen'].isin(pathogens)]
    else:
        print("No pathogens specified. Using all pathogens")
    print(f"Number of tests before parsing: {TESSy_data.shape[0]:,}")
    TESSy_data['year'] = pd.to_datetime(TESSy_data['DateUsedForStatisticsISO']).dt.year
    TESSy_data['date'] = pd.to_datetime(TESSy_data['DateUsedForStatisticsISO'], format='%Y-%m-%d')
    TESSy_data.drop(columns=['DateUsedForStatisticsISO'], inplace=True)
    TESSy_data = TESSy_data[TESSy_data['SIR'] != 'I']
    if (pathogens and len(pathogens) > 1) or not pathogens:
        cols = ['ReportingCountry', 'date', 'year', 'LaboratoryCode', 'PatientCounter',
                'Gender', 'Age','IsolateId', 'Pathogen', 'Antibiotic', 'SIR']
        df = TESSy_data[cols]
        df = df.rename(columns={'ReportingCountry': 'country',
                'Gender': 'gender',
                'Age': 'age',
                'Pathogen': 'pathogen',
                'Antibiotic': 'antibiotic',
                'SIR': 'phenotype'})
    else:
            cols = ['ReportingCountry', 'date', 'year', 'LaboratoryCode', 'PatientCounter',
                    'Gender', 'Age','IsolateId', 'Antibiotic', 'SIR']
            df = TESSy_data[cols]
            df = df.rename(columns={'ReportingCountry': 'country',
                    'Gender': 'gender',
                    'Age': 'age',
                    'Antibiotic': 'antibiotic',
                    'SIR': 'phenotype'})
    
    # drop tests
    alternative_nan = ['unknown', 'UNKNOWN']
    df['IsolateId'] = df['IsolateId'].replace(alternative_nan, np.nan)
    print(f"Dropping {df['IsolateId'].isnull().sum():,} tests with missing IsolateId") 
    df = df[df['IsolateId'].notnull()] 
    
    # filter out antibiotics
    if exclude_antibiotics:
        print(f"Filtering out antibiotics: {exclude_antibiotics}")
        df = df[~df['antibiotic'].isin(exclude_antibiotics)]
        print(f"Number of antibiotics: {df['antibiotic'].nunique():,}")
    
    print("Creating new ID of the form: country_year_labID_patientID_IsolateID")
    id_cols = ['country', 'year', 'LaboratoryCode', 'PatientCounter', 'IsolateId']
    df['ID'] = df[id_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    print(f"Number of unique IDs: {df['ID'].nunique():,}")
    
    print(f"Are there any ID-antibiotic combinations with more than one date? {'Yes' if any(df.groupby(['ID', 'antibiotic'])['date'].nunique() > 1) else 'No'}")
    duplicates = df.duplicated(subset=['ID', 'antibiotic', 'phenotype'])
    print(f"Are there duplicates of ID-antibiotic-phenotype combination? {'Yes' if duplicates.any() else 'No'}")
    if duplicates.any():
        print(f"Dropping {duplicates.sum():,} duplicates")
        df.drop_duplicates(subset=['ID', 'antibiotic', 'phenotype'], inplace=True, keep='first')
    
    ## Code to look more deeply at duplicates, seeing if there is an antibiotic with different phenotypes. 
    num_unique_phenotypes = df.groupby(['ID', 'antibiotic'])['phenotype'].nunique().sort_values(ascending=False)
    print(f"Are there IDs with more than one phenotype per antibiotic? {'Yes' if any(num_unique_phenotypes > 1) else 'No'}")
    if any(num_unique_phenotypes > 1):
        df = df.groupby(['ID', 'antibiotic']).first().reset_index()
    
    print(f"Number of tests after parsing: {df.shape[0]:,}")
    print(f"Aggregating tests for each ID...")
    df_agg = df.groupby('ID')[['antibiotic', 'phenotype']].agg(list).reset_index()
    df_agg['phenotypes'] = df_agg.apply(
        lambda x: [x['antibiotic'][i] + "_" + x['phenotype'][i] for i in range(len(x['antibiotic']))], axis=1)
    df_agg.drop(columns=['antibiotic', 'phenotype'], inplace=True)
    
    df_others = df.drop(columns=['antibiotic', 'phenotype']).groupby('ID').first().reset_index() 
    df = df_agg.merge(df_others, on='ID')
    
    cols_in_order = ['year', 'country', 'gender', 'age', 'phenotypes'] # can change to date or year-month here
    if (pathogens and len(pathogens) > 1) or not pathogens:
        df = df[['pathogen'] + cols_in_order]
    else:
        df = df[cols_in_order]
    df['country'] = df['country'].replace('United Kingdom', 'UK')
    df['country'] = df['country'].map(country_code_to_name)
    df['num_ab'] = df['phenotypes'].apply(lambda x: len(x))
    df['num_R'] = df['phenotypes'].apply(lambda x: len([p for p in x if p.endswith('R')]))
    df['num_S'] = df['phenotypes'].apply(lambda x: len([p for p in x if p.endswith('S')]))
    # make sure there are no samples without phenotypes
    df = df[df['num_ab'] > 0]
    
    if impute_age:
        df = impute_col(df, 'age', random_state=42)
    else:
        print(f"Dropping {df['age'].isnull().sum():,} samples with missing value in the 'age' column")
        df.dropna(subset=['age'], inplace=True)
        
    alternative_nan = ["UNK", "O"]
    df['gender'].replace(alternative_nan, np.nan, inplace=True)
    if impute_gender:
        df = impute_col(df, 'gender', random_state=42)
    else:
        print(f"Dropping {df['gender'].isnull().sum():,} samples with missing value in the 'gender' column")
        df.dropna(subset=['gender'], inplace=True)

    if not any([impute_age, impute_gender]):
        print(f"Number of samples after dropping samples with missing values: {df.shape[0]:,}")
    else:
        print(f"Final number of samples: {df.shape[0]:,}")
    
    df.reset_index(drop=True, inplace=True)
    if save_path:
        print(f"Saving to {save_path}")
        df.to_pickle(save_path)

    return df