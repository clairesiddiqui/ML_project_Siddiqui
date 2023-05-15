#==================================================================#
# Description:
# This file is used to apply ML to the data collection based on
# satellite and shipboard measurements from across the study region
#==================================================================#



#======================== Loading packages ======================#

import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import neighbors
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree

from sklearn.svm import SVR
import shap
import matplotlib.pyplot as plt
from pylab import plot,show
#import pdb; pdb.set_trace()




#======================== Data Import ========================#

masterfile = "/Users/csi/private/Data_Scientist/Digethic/Python_coding/DIGETHIC_import_datafile.csv"

with open(masterfile, 'r') as nf:
    carbon_data = np.genfromtxt(masterfile, dtype=None, delimiter='\t', skip_header=0)
    nn = len(carbon_data)
    carbon_data = carbon_data.astype(float)

print(carbon_data)








#census = pd.DataFrame(census, columns = ['age', 'workclass', 'education', 'marital-status','occupation','relationship','race', 'sex', 'capital-gain',
#                'capital-loss', 'hours-per-week', 'native-country', 'target'])


# SPLIT DATA INTO TEST- AND TRAINING-DATASETS
#feature_cols = ['age', 'workclass', 'education', 'marital-status','occupation','relationship','race', 'sex', 'capital-gain',
#                'capital-loss', 'hours-per-week', 'native-country']
#X = census[feature_cols]   # Features
#y = census['target']       # Target variable

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#print('X Train: {}'.format(X_train.shape)) #X Train: (22792, 13)
#print('Y Train: {}'.format(y_train.shape)) #Y Train: (22792,)
#print('X Test:  {}'.format(X_test.shape))  #X Test:  (9769, 13)
#print('Y Test:  {}'.format(y_test.shape))  #Y Test:  (9769,)