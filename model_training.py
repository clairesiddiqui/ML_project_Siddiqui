#==================================================================#
# Description:
# This file is used to apply ML to the data collection based on
# satellite and shipboard measurements from across the study region
#==================================================================#



#======================== Loading packages ======================#

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
#import shap
import matplotlib.pyplot as plt
from pylab import plot,show
#import pdb; pdb.set_trace()




#======================== MODEL TRAINING ========================#

# import data:
masterfile = "/Users/csi/private/Data_Scientist/Digethic/Python_coding/DIGETHIC_import_datafile.csv"
with open(masterfile, 'r') as nf:
    carbon_data = np.genfromtxt(masterfile, dtype=None, delimiter='\t', skip_header=0)
    nn = len(carbon_data)
    carbon_data = carbon_data.astype(float)
print(np.shape(carbon_data))  # shape: 34734, 22


# create dataframe:
carbon_df = pd.DataFrame(carbon_data, columns = ['year', 'month', 'day', 'hour','minute','latitude','longitude', 'depth', 
                'pCO2', 'temperature', 'salinity', 'air_pressure', 'wind_speed', 'alkalinity', 'xx', 'yy', 'modis_latitude',
                'modis_longitude', 'CHL', 'SST', 'KD490', 'PAR'])

carbon_df = carbon_df.dropna()
print("length data file:", len(carbon_df))    # before: 34734, after: 12799


# split data into test- and training-datasets:
feature_cols = ['CHL', 'SST', 'KD490', 'PAR']
X = carbon_df[feature_cols]   # Features
y = carbon_df['pCO2']         # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('X Train: {}'.format(X_train.shape)) #X Train: (8959, 4)
print('Y Train: {}'.format(y_train.shape)) #Y Train: (8959,)
print('X Test:  {}'.format(X_test.shape))  #X Test:  (3840, 4)
print('Y Test:  {}'.format(y_test.shape))  #Y Test:  (3840,)






'========================='
# Linear Regression
poly = PolynomialFeatures(2)
x_poly = poly.fit_transform(X_train)
lin_reg = LinearRegression()
predictions_lin_reg = lin_reg.fit(x_poly, y_train).predict(poly.transform(X_test))
importances_lin_reg = lin_reg.coef_
print("Linear Regression:", predictions_lin_reg)
print("importances:", importances_lin_reg)
print(" ")
#accuracy_lin_reg = accuracy_score(y_test, predictions_lin_reg)
#accuracy_lin_reg = round(accuracy_lin_reg, 2)
#print("accuracy:", accuracy_lin_reg)



# Random Forrest Regressor



# Gradient Boosting Regressor



# CatBoost Regressor



'========================='
# Support Vector Machine
svm_lin = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
predictions_svm_lin = svm_lin.fit(X_train, y_train).predict(X_test)
#accuracy_svm_lin = accuracy_score(y_test, predictions_svm_lin)
#accuracy_svm_lin = round(accuracy_svm_lin,2)
print("Support Vector Machine:", predictions_svm_lin)
#print("accuracy:", accuracy_svm_lin)
print(" ")


# accuracy score only applicable with classifications



# PLOTTING DATASET
fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

# temperature and SST
#ax.scatter(y=predictions_svm_lin, x=y_test)
ax.scatter(y=predictions_lin_reg, x=y_test)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
ax.set_xlim(300,500)
ax.set_ylim(300,500)
plt.show()

