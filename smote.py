from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


#Copied load_data because it wasn't working without it :(
def load_data(fname):
    """Load CSV file with any number of consecutive features, starting in column 0, where last column is the class"""
    df = pd.read_csv(fname)
    df = df.drop(columns=['Description']) 
    df = df.drop(columns=['Name']) 
    df = df.drop(columns=['PetID'])
    df = df.drop(columns=['RescuerID'])
    df = df.drop(columns=['Breed2'])
    df = df.drop(columns=['Color3'])
    df = df.drop(columns=['VideoAmt'])
    df = df[df['Age'] <= 20]
    nc = df.shape[1] 
    matrix = df.values 
    table_X = matrix [:, 0:nc-1] 
    table_y = matrix [:, nc-1]            
    features = df.columns.values[0:nc-1] 
    target_name = df.columns.values[nc-1] 
    return table_X, table_y, features, target_name, df


table_X, table_y, features, target_name, df = load_data('PetFinder_dataset.csv')

#Smote for adopted or not
def smoteadopted():
    df['Adopted'] = df['AdoptionSpeed'].apply(lambda x: 1 if x == 4 else 0) 

    X_train, X_test, y_train, y_test = train_test_split(table_X, df['Adopted'], test_size=0.2, random_state=42)

    smote = SMOTE(sampling_strategy='auto',random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    df_resampled = pd.DataFrame(X_smote, columns=features)
    df_resampled['Adopted'] = y_smote

    print(pd.Series(y_train).value_counts())
    print(pd.Series(y_smote).value_counts())

#Smote for adoptionspeed
def smoteadoptionspeed():
    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, test_size=0.2, random_state=42)
    smote = SMOTE(sampling_strategy='auto',random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    df_resampled = pd.DataFrame(X_smote, columns=features)
    df_resampled['AdoptionSpeed'] = y_smote

    print(pd.Series(y_train).value_counts())
    print(pd.Series(y_smote).value_counts())


#smoteadopted()
smoteadoptionspeed()
