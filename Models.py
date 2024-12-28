from LoadingData import * 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, adjusted_rand_score, silhouette_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralCoclustering
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import pydot
from sklearn import metrics

def display_tree(tree_to_display, feature_names, class_names, fname, figsize=(10, 10)):

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

    img = mpimg.imread(fname + ".png")
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off") 
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
    gnb = gnb.fit(X_train, y_train)
    gnb.predict_proba(X_train)
    print("Accuracy on training set:",  gnb.score(X_train, y_train))
    print("Accuracy on test set:",  gnb.score(X_test, y_test))
    crossValidation(table_X,table_y,10,gnb)
    y_train_pred = gnb.predict(X_train)
    y_train_pred
    cm_train = confusion_matrix(y_train, y_train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                                display_labels=gnb.classes_)
    disp.plot(cmap='Oranges') 
    plt.title("Confusion Matrix - Train Set")
    plt.show()

    # predictions for test set
    y_test_pred = gnb.predict(X_test)
    y_test_pred
    cm_test = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                display_labels=gnb.classes_)
    disp.plot(cmap='Oranges')
    plt.title("Confusion Matrix - Test Set")
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

    cm_train = confusion_matrix(y_train, y_train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                              display_labels=knn.classes_)
    disp.plot(cmap='Oranges')
    plt.title("Confusion Matrix - Train Set") 
    plt.show()

    y_test_pred = knn.predict(X_test)
    y_test_pred
    cm_test = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                display_labels=knn.classes_)
    disp.plot(cmap='Oranges')
    plt.title("Confusion Matrix - Test Set")
    plt.show()

def RandomF(table_X, table_y):
    X_train, X_test, y_train, y_test = train_test_split(
        table_X, table_y, random_state=0, stratify=table_y
    )

    clf = RandomForestClassifier(random_state=0, class_weight="balanced")

    param_grid = {
        "n_estimators": [25, 50, 100],
        "max_depth": [3, 5,20],
        "min_samples_leaf": [10, 15, 20]
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
    disp.plot(cmap='Oranges') 
    plt.title("Confusion Matrix")
    plt.show()

    crossValidation(table_X, table_y, 10, best_clf)

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def OurKmeans(table_X, table_y, nclusters=8):
    # KMeans Clustering
    kms = KMeans(n_clusters=nclusters, random_state=0, n_init='auto')
    kms = kms.fit(table_X)
    kmeans_labels = kms.labels_
    print("Kmeans silhouette_score:", metrics.silhouette_score(table_X, kmeans_labels))
    ari_score_kmeans = adjusted_rand_score(table_y, kmeans_labels)
    print(f"Kmeans Adjusted Rand Index (ARI): {ari_score_kmeans}")
    
    hca = AgglomerativeClustering(linkage="ward", n_clusters=nclusters).fit(table_X)
    hca_labels = hca.labels_
    print("HCA silhouette_score:", metrics.silhouette_score(table_X, hca_labels))
    ari_score_hca = adjusted_rand_score(table_y, hca_labels)
    print(f"HCA Adjusted Rand Index (ARI): {ari_score_hca}")

    contingency_kmeans = pd.crosstab(kmeans_labels, table_y)
    contingency_hca = pd.crosstab(hca_labels, table_y)
   
    sns.heatmap(contingency_kmeans, annot=True, fmt="d", cmap="Oranges")
    plt.title(f"K-Means Clustering (n={nclusters}) vs Target")
    plt.xlabel("Target")
    plt.ylabel("Cluster")
    plt.show()
    
    sns.heatmap(contingency_hca, annot=True, fmt="d", cmap="Purples")
    plt.title(f"HCA Clustering (n={nclusters}) vs Target")
    plt.xlabel("Target")
    plt.ylabel("Cluster")
    plt.show()

    


def OurBiCluster(table_X, table_y, nclusters=8):
    spectral_clustering = SpectralCoclustering(n_clusters=nclusters, random_state=0)
    spectral_clustering.fit(table_X)
    spectral_row_labels = spectral_clustering.row_labels_
    spectral_column_labels = spectral_clustering.column_labels_ 
    hca = AgglomerativeClustering(linkage="ward", n_clusters=nclusters)
    hca_row_labels = hca.fit_predict(table_X) 
    silhouette_spectral_rows = silhouette_score(table_X, spectral_row_labels)
    silhouette_spectral_columns = silhouette_score(table_X.T, spectral_column_labels)
    print("Spectral Co-clustering Silhouette Score for rows:", silhouette_spectral_rows)
    print("Spectral Co-clustering Silhouette Score for columns:", silhouette_spectral_columns)  
    print("HCA Silhouette Score for rows:", silhouette_score(table_X, hca_row_labels))
    ari_row_score = adjusted_rand_score(table_y, spectral_row_labels)
    print(f"Adjusted Rand Index (ARI) for row clustering from Spectral Co-clustering: {ari_row_score}")

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
    

# Logistic Regression

def logreg(table_X,table_y,df,features):
    df.head()
    print(df.shape)
    print(df['Adopted'].value_counts())
    np.bincount(df['Adopted'])
    X_train, X_test, y_train, y_test = train_test_split(table_X, table_y, test_size=0.3, random_state=0)
    crossValidation(table_X, table_y, 10, LogisticRegression(solver='liblinear'))
    petfinder_logreg = LogisticRegression(solver='liblinear').fit(X_train, y_train)
    petfinder_logreg
    print("Train set score (Accuracy)=", petfinder_logreg.score(X_train, y_train))
    print("Test set score (Accuracy)=", petfinder_logreg.score(X_test, y_test))
    petfinder_logreg100 = LogisticRegression(C=100,solver='liblinear').fit(X_train, y_train)
    print("Train set score (Accuracy)=", petfinder_logreg100.score(X_train, y_train))
    print("Test set score (Accuracy)=", petfinder_logreg100.score(X_test, y_test))
    petfinder_logreg001 = LogisticRegression(C=0.001,solver='liblinear').fit(X_train, y_train)
    print("Train set score (Accuracy)=", petfinder_logreg001.score(X_train, y_train))
    print("Test set score (Accuracy)=", petfinder_logreg001.score(X_test, y_test))
    nc = df.shape[1] 

    plt.plot(petfinder_logreg.coef_.T, 'o', label="C=1")
    plt.plot(petfinder_logreg100.coef_.T, '^', label="C=100")
    plt.plot(petfinder_logreg001.coef_.T, 'v', label="C=0.001")
    plt.xticks(range(len(features)), features , rotation=90)
    xlims = plt.xlim()
    plt.hlines(0, xlims[0], xlims[1])
    plt.xlim(xlims)
    plt.ylim(-5, 5)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.legend()

    plt.figure(figsize=(8,8))
    sns.barplot(x=features, y=petfinder_logreg.coef_.flatten())
    plt.xticks(rotation=90)
    plt.show()
    for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
        petfinder_lr_l1 = LogisticRegression(C=C, penalty="l1",max_iter=200,solver='liblinear').fit(X_train, y_train)
        print("Train accuracy of L1 logreg with C={:.3f} = {:.2f}".format(
            C, petfinder_lr_l1.score(X_train, y_train)))
        print("Test accuracy of L1 logreg with C={:.3f} = {:.2f}".format(
            C, petfinder_lr_l1.score(X_test, y_test)))
        plt.plot(petfinder_lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

    plt.xticks(range(len(features)), features, rotation=90)
    xlims = plt.xlim()
    plt.hlines(0, xlims[0], xlims[1])
    plt.xlim(xlims)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-5, 5)
    plt.legend(loc=3)
    c_values = np.linspace(0.01, 100, 1000)
    train_acc = np.zeros(1000)
    for i, c in enumerate(c_values):
        petfinder_logreg = LogisticRegression(C=c,solver='liblinear').fit(X_train, y_train)
        train_acc[i] = petfinder_logreg.score(X_train, y_train)

    plt.plot(c_values, train_acc)
    plt.xlabel("C")
    plt.ylabel("Train Accuracy")

    c_values[np.argmax(train_acc)]
    df['PhotoAmt'] = df['PhotoAmt']  
    df['Adopted'] = df['Adopted']  

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PhotoAmt', y='Adopted', hue='Adopted', palette='viridis')
    plt.title('PhotoAmt vs Adoptio status of Pets')
    plt.xlabel('PhotoAmt')
    plt.ylabel('Adopted')
    plt.legend(title='Adopted', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


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

