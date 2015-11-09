
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.externals import joblib
import pickle

#------------------------------------------#
# Exploring testing and training and
# investigating hyper parameters
#------------------------------------------#
def KNN_test_train(X, y, wesave = False):

    # Split into test and
    X_dev, X_eval, y_dev, y_eval = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=1928374)


    # Let's see how well we can do for set number of neighbors
    nbors   = [10, 15, 20, 30, 50, 70, 90, 110, 150, 200, 300, 500, 1000]
    logloss = []

    for nbor in nbors:

        # Make the classifier
        clf = KNeighborsClassifier(n_neighbors = nbor)
                
        # Fit data
        clf.fit(X_dev, y_dev)
    
        # Save log loss
        ll = log_loss(y_eval, clf.predict_proba(X_eval))
        logloss.append(ll)
        print ll, nbor
        
    print logloss

    # Plot the information
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(ncols=1,figsize=(7,6))
    plt.plot(nbors, logloss, lw=1.5, color='b')
    plt.xlabel('# Neighbors')
    plt.ylabel('log loss')

    if wesave:
        plt.savefig('plots/knn_logloss_test.png')

    plt.show()

#------------------------------------------#
# Save the model for use later
#------------------------------------------#
def KNN_save(X, y):

    # Make the classifier
    clf = KNeighborsClassifier(n_neighbors = 1000)
                
    # Fit data
    clf.fit(X, y)
    
    # Dump the model
    joblib.dump(clf, 'models/knn1000.pkl')
    


#------------------------------------------#
# Evaluate model
#------------------------------------------#
def KNN_evaluate(X):

    # Make this an option later, but load the model
    modelpath = 'mydata/knn1000.pkl'
    clf = joblib.load(modelpath)

    # Evaluate the probabilities
    probs = clf.predict_proba(X)

    return probs
