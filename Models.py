from LoadingData import * 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import pydot
from sklearn import metrics

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

def OurTree(table_X, table_y,mx_leaf_nodes,features):

    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, random_state=0)
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=mx_leaf_nodes)
    clf = clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print("Accuracy on training set:", accuracy_score(y_train, y_train_pred))
    print("Accuracy on test set:", accuracy_score(y_test, y_test_pred))
    crossValidation(table_X,table_y,10,clf)
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
    crossValidation(table_X,table_y,10,gnb)
    y_train_pred = gnb.predict(X_train)
    y_train_pred
    #%matplotlib inline

    cm_train = confusion_matrix(y_train, y_train_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                                display_labels=gnb.classes_)
    disp.plot()
    plt.show()

## KNN
def Ourknn(table_X, table_y,numberNeighbours):
    knn = neighbors.KNeighborsClassifier(n_neighbors=numberNeighbours)
    knn

    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, random_state=0)
    knn = knn.fit(X_train, y_train)
    print("Accuracy on training set:",  knn.score(X_train, y_train))
    print("Accuracy on test set:",  knn.score(X_test, y_test))
    crossValidation(table_X,table_y,10,knn)
    y_train_pred = knn.predict(X_train)

    cm_train = confusion_matrix(y_train, y_train_pred, labels=knn.classes_)
    #%matplotlib inline
    cm_train = confusion_matrix(y_train, y_train_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                                display_labels=knn.classes_)
    disp.plot()
    plt.show()

def RandomF(table_X, table_y):
    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, random_state=0, stratify=table_y)


    clf = RandomForestClassifier(random_state=0, class_weight='balanced')

    param_grid = {
        'n_estimators': [25, 50, 100],         
        'max_depth': [5, 8, 10, None],    
        'min_samples_leaf': [1, 2, 10]
    }

    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_

    y_train_pred = best_clf.predict(X_train)
    y_test_pred = best_clf.predict(X_test)

    print("Accuracy on training set:", accuracy_score(y_train, y_train_pred))
    print("Accuracy on test set:", accuracy_score(y_test, y_test_pred))
    

    cm = confusion_matrix(y_test, y_test_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_clf.classes_) 
    plt.title("Confusion Matrix")
    plt.show()
    crossValidation(table_X,table_y,10,best_clf)

def OurKmeans(table_X,table_y, nclusters=8):
    kms = KMeans(nclusters, random_state=0, n_init='auto')
    kms = kms.fit(table_X)
    predicted_labels = kms.labels_
    hca = AgglomerativeClustering(linkage="ward", n_clusters=nclusters).fit(table_X)
    print("Kmeans silhouette_score", metrics.silhouette_score(table_X, kms.labels_))
    print("HCA silhouette_score", metrics.silhouette_score(table_X, hca.labels_))
    ari_score = adjusted_rand_score(table_y, predicted_labels)
    print(f"Adjusted Rand Index (ARI): {ari_score}")
    score = silhouette_score(table_X, predicted_labels)
    print(f"Silhouette Score: {score}")

def svm(table_X,table_y):
    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, random_state=0)
    lsvm = LinearSVC(dual = 'auto').fit(X_train, y_train)
    crossValidation(table_X, table_y, 10, lsvm)
    print("Training set score (Accuracy) =", lsvm.score(X_train, y_train))
    print("Test set score (Accuracy) =", lsvm.score(X_test, y_test))
    print('---'*30)
    #Não sei se é necessário
    print ("LinearSVC coefficients and intercept:")
    print ("Coeficients (w) =\n", lsvm.coef_)
    print ("Intercept (b) =", lsvm.intercept_)


def crossValidation(table_X, table_y, splits, classifier):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    kfold = StratifiedKFold(n_splits=splits).split(table_X, table_y)
    scores = []
    for k, (train, test) in enumerate(kfold):
        classifier.fit(table_X[train], table_y[train])
        score = classifier.score(table_X[test], table_y[test])
        scores.append(score)
        
        print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (
            k+1, np.bincount(table_y[train].astype(int)), score))
        
    print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))