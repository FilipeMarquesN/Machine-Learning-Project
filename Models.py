from LoadingData import * 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pydot

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

def RandomF(table_X, table_y):
    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, random_state=0)

    clf = RandomForestClassifier(max_depth=10, random_state=0)

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print("Accuracy on training set:", accuracy_score(y_train, y_train_pred))
    print("Accuracy on test set:", accuracy_score(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()  
    plt.title("Confusion Matrix")
    plt.show()