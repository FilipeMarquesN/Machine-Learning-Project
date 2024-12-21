from LoadingData import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import neighbors
import numpy as n
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
# compute confusion matrix train
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import tree
import pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_tree(tree_to_display, feature_names, class_names, fname, figsize=(10, 10)):
    """
    Display a decision tree using sklearn's export_graphviz and pydot.
    """

    class_names = [str(name) for name in class_names.unique()]
    dot_data = tree.export_graphviz(
        tree_to_display,
        out_file=None,  # None para gerar uma string ao invés de um arquivo
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    (graph,) = pydot.graph_from_dot_data(dot_data)
    graph.write_png(fname + ".png")  # Salvar como imagem PNG

    # Exibir a imagem usando matplotlib
    img = mpimg.imread(fname + ".png")
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")  # Remover eixos
    plt.show()

def OurTree(table_X, table_y,mx_leaf_nodes):
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=mx_leaf_nodes)
    clf = clf.fit(table_X,  table_y)

    display_tree(clf, features, pd.Series(table_y), fname="decision_tree")

def naive(table_X, table_y):
    # Modelo Probabilistico Naive
    gnb = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, random_state=0)

    print(X_train.shape)
    print(X_test.shape)
    gnb = gnb.fit(X_train, y_train)
    gnb.predict_proba(X_train)
    print("Accuracy on training set:",  gnb.score(X_train, y_train))
    print("Accuracy on test set:",  gnb.score(X_test, y_test))
    y_train_pred = gnb.predict(X_train)
    y_train_pred
    #%matplotlib inline

    cm_train = confusion_matrix(y_train, y_train_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                                display_labels=gnb.classes_)
    disp.plot()
    plt.show()

## KNN
def knn(table_X, table_y,numberNeighbours):
    knn_3 = neighbors.KNeighborsClassifier(n_neighbors=numberNeighbours)
    knn_3

    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, random_state=0)
    knn_3 = knn_3.fit(X_train, y_train)
    print("Accuracy on training set:",  knn_3.score(X_train, y_train))
    print("Accuracy on test set:",  knn_3.score(X_test, y_test))

    y_train_pred = knn_3.predict(X_train)
    y_train_pred

    cm_train = confusion_matrix(y_train, y_train_pred, labels=knn_3.classes_)
    print (cm_train)
    #%matplotlib inline
    cm_train = confusion_matrix(y_train, y_train_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                                display_labels=knn_3.classes_)
    disp.plot()
    plt.show()

table_X, table_y, features, target_name, df = load_data('PetFinder_dataset.csv')

#Predict Adoption Speed
table_X_Dogs, table_y_Dogs_Speed, features_Dogs, target_Name_Dogs, df_Dogs = loadDataAnimalType(df,2)
table_X_Cats, table_y_Cats_Speed, features_Cats, target_Name_Cats, df_Cats = loadDataAnimalType(df,2)

#Dogs Results
OurTree(table_X_Dogs, table_y_Dogs_Speed,15)
knn(table_X_Dogs, table_y_Dogs_Speed,3)
naive(table_X_Dogs, table_y_Dogs_Speed)
#Cats Results
OurTree(table_X_Cats, table_y_Cats_Speed,15)
knn(table_X_Cats, table_y_Cats_Speed,3)
naive(table_X_Cats, table_y_Cats_Speed)

#Predict Adopted
table_X_Cats_Adopted, table_y_Cats_Adopted, features_Cats_Adopted, target_Name_Cats_Adopted, df_Cats_Adopted = loadDataAdopted(df_Cats)
table_X_Dogs_Adopted, table_y_Dogs_Adopted, features_Dogs_Adopted, target_Name_Dogs_Adopted, df_Dogs_Adopted = loadDataAdopted(df_Dogs)

#Dogs Results
OurTree(table_X_Dogs, table_y_Dogs_Adopted,15)
knn(table_X_Dogs, table_y_Dogs_Adopted,3)
naive(table_X_Dogs, table_y_Dogs_Adopted)
#Cats Results
OurTree(table_X_Cats, table_y_Cats_Adopted,15)
knn(table_X_Cats, table_y_Cats_Adopted,3)
naive(table_X_Cats, table_y_Cats_Adopted)