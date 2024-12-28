from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def smoteadopted(table_X,table_y,features):
    
    X_train, X_test, y_train, y_test = train_test_split(table_X,table_y, test_size=0.2, random_state=0)

    smote = SMOTE(sampling_strategy='auto',random_state=0)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    df_resampled = pd.DataFrame(X_smote, columns=features)
    df_resampled['Adopted'] = y_smote
    return X_smote, y_smote, df_resampled

#Smote for adoptionspeed
def smoteadoptionspeed(table_X,table_y,features):
    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, test_size=0.2, random_state=0)
    smote = SMOTE(sampling_strategy='auto',random_state=0)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    df_resampled = pd.DataFrame(X_smote, columns=features)
    df_resampled['AdoptionSpeed'] = y_smote
    return X_smote, y_smote, df_resampled

