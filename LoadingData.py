import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_data(fname):
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
    table_y_AdoptionSpeed = matrix [:, nc-1]         
    features = df.columns.values[0:nc-1] 
    target_name = df.columns.values[nc-1] 
    
    return table_X, table_y_AdoptionSpeed, features, target_name, df

def loadDataAdopted(df):
    Adopted = df['AdoptionSpeed'].apply(lambda x: 0 if x == 4 else 1)
    df = df.drop(columns=['AdoptionSpeed'])
    df['Adopted'] = Adopted
    nc = df.shape[1]
    matrix = df.values
    table_X = matrix[:, 0:nc-1] 
    table_y_Adoption = matrix[:, nc-1]    
    features = df.columns.values[0:nc-1] 
    target_name = df.columns.values[nc-1] 
    
    return table_X, table_y_Adoption, features, target_name, df

def loadDataAnimalType(df,number):

    df = df[df['Type'] == number]
    nc = df.shape[1]
    matrix = df.values
    table_X = matrix[:, 0:nc-1] 
    table_y = matrix[:, nc-1]   
    features = df.columns.values[0:nc-1]  
    target_name = df.columns.values[nc-1] 
    
    return table_X, table_y, features, target_name, df


def loadScaledData(df):
    nc = df.shape[1]
    matrix = df.values
    table_X = matrix[:, 0:nc-1]
    table_y_AdoptionSpeed = matrix[:, nc-1]
    X_scaled = preprocessing.scale(table_X)
    features = df.columns.values[0:nc-1]
    df_scaled = pd.DataFrame(X_scaled, columns=features)
    target_name = df.columns.values[nc-1]
    
    return X_scaled, table_y_AdoptionSpeed, features, target_name, df_scaled
