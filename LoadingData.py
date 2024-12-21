import numpy as np
import pandas as pd

def load_data(fname):
    """Load CSV file with any number of consecutive features, starting in column 0, where last column is the class"""
    df = pd.read_csv(fname)
    df = df.drop(columns=['Description']) #Podemso mudar pra 0 ou 1 se têm não sei se seria util
    df = df.drop(columns=['Name']) #Podemso mudar pra 0 ou 1 se têm não sei se seria util
    df = df.drop(columns=['PetID'])
    df = df.drop(columns=['RescuerID'])
    df = df.drop(columns=['Breed2'])
    df = df.drop(columns=['Color3'])
    df = df.drop(columns=['VideoAmt'])
    df = df[df['Age'] <= 20]
    df['Adoption'] = df['AdoptionSpeed'].apply(lambda x: 0 if x == 4 else 1)
    nc = df.shape[1] # number of columns
    matrix = df.values # Convert dataframe to darray
    table_X = matrix [:, 0:nc-2] 
    table_y_AdoptionSpeed = matrix [:, nc-1] # get class (last columns)           
    features = df.columns.values[0:nc-1] 
    target_name = df.columns.values[nc-1] #get target name
    return table_X, table_y_AdoptionSpeed, features, target_name, df

def loadDataAdopted(df):
    """
    Load CSV file with any number of consecutive features, starting in column 0, 
    where the last column is the class.
    """

    df['AdoptionSpeed'] = df['AdoptionSpeed'].apply(lambda x: 0 if x == 4 else 1)
    

    nc = df.shape[1]
    

    matrix = df.values
    

    table_X = matrix[:, 0:nc-1] 
    table_y_Adoption = matrix[:, nc-1]    
    

    features = df.columns.values[0:nc-1]  # Nomes das colunas (features)
    target_name = df.columns.values[nc-1]  # Nome da classe alvo
    
    return table_X, table_y_Adoption, features, target_name, df

def loadDataAnimalType(df,number):
    """
    Load CSV file with any number of consecutive features, starting in column 0, 
    where the last column is the class.
    Number 1 == dog, number 2 == cat
    """


    df = df[df['Type'] == number]
    

    nc = df.shape[1]
    

    matrix = df.values
    

    table_X = matrix[:, 0:nc-1] 
    table_y = matrix[:, nc-1]    
    

    features = df.columns.values[0:nc-1]  # Nomes das colunas (features)
    target_name = df.columns.values[nc-1]  # Nome da classe alvo
    
    return table_X, table_y, features, target_name, df
