from Models import *
import itertools

table_X, table_y, features, target_name, df = load_data('PetFinder_dataset.csv')

#Predict Adoption Speed
table_X_Dogs, table_y_Dogs_Speed, features_Dogs, target_Name_Dogs, df_Dogs = loadDataAnimalType(df,2)
table_X_Cats, table_y_Cats_Speed, features_Cats, target_Name_Cats, df_Cats = loadDataAnimalType(df,1)

#Dogs Results
OurTree(table_X_Dogs, table_y_Dogs_Speed,60,features_Dogs)
# knn(table_X_Dogs, table_y_Dogs_Speed,3)
# naive(table_X_Dogs, table_y_Dogs_Speed)
#Cats Results
OurTree(table_X_Cats, table_y_Cats_Speed,60,features_Cats)
# knn(table_X_Cats, table_y_Cats_Speed,3)
# naive(table_X_Cats, table_y_Cats_Speed)

#Predict Adopted
table_X_Cats_Adopted, table_y_Cats_Adopted, features_Cats_Adopted, target_Name_Cats_Adopted, df_Cats_Adopted = loadDataAdopted(df_Cats)
table_X_Dogs_Adopted, table_y_Dogs_Adopted, features_Dogs_Adopted, target_Name_Dogs_Adopted, df_Dogs_Adopted = loadDataAdopted(df_Dogs)

#Dogs Results
OurTree(table_X_Dogs, table_y_Dogs_Adopted,15,features_Dogs)
# knn(table_X_Dogs, table_y_Dogs_Adopted,3)
# naive(table_X_Dogs, table_y_Dogs_Adopted)
#Cats Results
OurTree(table_X_Cats, table_y_Cats_Adopted,15,features_Cats)
# knn(table_X_Cats, table_y_Cats_Adopted,3)
# naive(table_X_Cats, table_y_Cats_Adopted)

# RandomF(table_X_Dogs, table_y_Dogs_Speed)
# RandomF(table_X_Cats, table_y_Cats_Speed)
# RandomF(table_X_Cats_Adopted, table_y_Cats_Adopted)
# RandomF(table_X_Dogs_Adopted, table_y_Dogs_Adopted)