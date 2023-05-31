#==================================================================#
# Description:
# This file is used to apply ML to predict sea surface pCO2
# based on satellite and shipboard measurements
#==================================================================#



#======================== Loading packages ======================#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor


from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.inspection import permutation_importance

#import shap
import matplotlib.pyplot as plt
from pylab import plot,show



#import pdb; pdb.set_trace()




#======================== PREPROCESSING ========================#


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
print("length data file:", len(carbon_df))           
# length before: 65563, after: 19635


# preprocess satellite data (remove offsets etc. according to the netCDF file's metadata description)
carbon_df['SST'] = carbon_df['SST'] / 0.005          
carbon_df['KD490'] = carbon_df['KD490'] / 0.0002
carbon_df['PAR'] = (carbon_df['PAR'] / 0.002) - 65.5


# split data into test- and training-datasets:
#feature_cols = ['CHL', 'SST', 'KD490', 'PAR']
feature_cols = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude']
X = carbon_df[feature_cols]   # Features
y = carbon_df['pCO2']         # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('X Train: {}'.format(X_train.shape)) #X Train: (8959, 4)
print('Y Train: {}'.format(y_train.shape)) #Y Train: (8959,)
print('X Test:  {}'.format(X_test.shape))  #X Test:  (3840, 4)
print('Y Test:  {}'.format(y_test.shape))  #Y Test:  (3840,)


# apply standard scaler on input variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)





#======================== MODEL TRAINING ========================#


'=================='
# Linear Regression
poly = PolynomialFeatures(3)
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


'========================'
# Decision Tree Regressor
dtr = DecisionTreeRegressor()
predictions_dtr = dtr.fit(X_train, y_train).predict(X_test)
print("Decision Tree Regressor:", predictions_dtr)
importances_dtr = dtr.feature_importances_


'========================'
# Random Forest Regressor
random_f_r = RandomForestRegressor(random_state=1)
predictions_rfr = random_f_r.fit(X_train, y_train).predict(X_test)
print("Random Forest Regressor:", predictions_rfr)
importances_rfr = random_f_r.feature_importances_


'======================='
# Support Vector Machine
svm_lin = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
#svm_lin = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1)
#svm_lin = SVR(kernel="linear", C=100, gamma="auto")
predictions_svm_lin = svm_lin.fit(X_train, y_train).predict(X_test)
#accuracy_svm_lin = accuracy_score(y_test, predictions_svm_lin)
#accuracy_svm_lin = round(accuracy_svm_lin,2)
print("Support Vector Machine:", predictions_svm_lin)
#print("accuracy:", accuracy_svm_lin)
print(" ")
perm_importance = permutation_importance(svm_lin, X_test, y_test)
print("Shape:", np.shape(perm_importance), perm_importance)

feature_names = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude']
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

'============================'
# Gradient Boosting Regressor
gradient_b_r = GradientBoostingRegressor(random_state=1)
predictions_gbr = gradient_b_r.fit(X_train, y_train).predict(X_test)
print("Gradient Boosting Regressor:", predictions_gbr)


'============================='
# K-Nearest Neighbor Regressor
n_neighbor = 5
knn = neighbors.KNeighborsRegressor(n_neighbor, weights= "distance")
predictions_knn = knn.fit(X_train, y_train).predict(X_test)
print("K-Nearest Neighbor Regressor:", predictions_knn)


# Voting Regressor
#voting_r = VotingRegressor(estimators= [('rf', random_f_r), ('gbr', gradient_b_r), ('lr', lin_reg)])
#predictions_vr = voting_r.fit(X_train, y_train).predict(X_test)


# CatBoost Regressor







#======================== MODEL EVALUATION ========================#

#ex = shap.TreeExplainer(dtr)
#shap_values_dtr = ex.shap_values(X_test)
#shap.summary_plot(shap_values_dtr, X_test)

# error estimation, Mean Absolute Error:
#mean_absolute_error(y_true=,y_pred=)

left = [1,2,3,4,5,6]
height_dtr = importances_dtr
height_rfr = importances_rfr
#height_svm = perm_importance #importances_svm
tick_label = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude']
df = pd.DataFrame({'decision tree': height_dtr, 'random forest': height_rfr}, index = tick_label)
ax = df.plot.barh(color = {"decision tree": "navy", "random forest": "lightblue"})
plt.xlabel("Feature")
plt.title("Model interpretation")
plt.show()


# PLOTTING DATASET
fig1, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 4))

# temperature and SST
#ax.scatter(y=predictions_svm_lin, x=y_test)
ax[0,0].scatter(y=predictions_lin_reg, x=y_test)
ax[0,0].set_xlabel('measured')
ax[0,0].set_ylabel('predicted - lr')
ax[0,0].set_xlim(300,500)
ax[0,0].set_ylim(300,500)

ax[0,1].scatter(y=predictions_rfr, x=y_test)
ax[0,1].set_xlabel('measured')
ax[0,1].set_ylabel('predicted - rfr')
ax[0,1].set_xlim(300,500)
ax[0,1].set_ylim(300,500)

ax[1,0].scatter(y=predictions_gbr, x=y_test)
ax[1,0].set_xlabel('measured')
ax[1,0].set_ylabel('predicted - gbr')
ax[1,0].set_xlim(300,500)
ax[1,0].set_ylim(300,500)

ax[1,1].scatter(y=predictions_svm_lin, x=y_test)
ax[1,1].set_xlabel('measured')
ax[1,1].set_ylabel('predicted - svm')
ax[1,1].set_xlim(300,500)
ax[1,1].set_ylim(300,500)

ax[2,0].scatter(y=predictions_knn, x=y_test)
ax[2,0].set_xlabel('measured')
ax[2,0].set_ylabel('predicted - knn')
ax[2,0].set_xlim(300,500)
ax[2,0].set_ylim(300,500)

ax[2,1].scatter(y=predictions_dtr, x=y_test)
ax[2,1].set_xlabel('measured')
ax[2,1].set_ylabel('predicted - dtr')
ax[2,1].set_xlim(300,800)
ax[2,1].set_ylim(300,800)

plt.show()

