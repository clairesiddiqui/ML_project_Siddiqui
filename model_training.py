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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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
carbon_df['PAR'] = (carbon_df['PAR']  - 65.5) / 0.002


# satellite data evaluation:
sst_r2   = r2_score(carbon_df['temperature'], carbon_df['SST'])
sst_mae  = mean_absolute_error(y_true= carbon_df['temperature'], y_pred= carbon_df['SST'])
sst_rmse = mean_squared_error(carbon_df['temperature'], carbon_df['SST'], squared=False)

x_sst = np.linspace(10,28,3)
y_sst = np.linspace(10,28,3)

sst_eval_1 = 'R$^2$= '+ str('{:.3f}'.format(sst_r2))
sst_eval_2 = 'MAE= '+ str('{:.3f}'.format(sst_mae))
sst_eval_3 = 'RMSE= '+ str('{:.3f}'.format(sst_rmse))
'========================================================'
fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

sst_1_t = ax.scatter(y=carbon_df['SST'], x=carbon_df['temperature'], s=10, label = sst_eval_1, color = "white")
sst_2_t = ax.scatter(y=carbon_df['SST'], x=carbon_df['temperature'], s=10, label = sst_eval_2, color = "white")
sst_3_t = ax.scatter(y=carbon_df['SST'], x=carbon_df['temperature'], s=10, label = sst_eval_3,  color = "darkblue")
ax.plot(x_sst, y_sst, 'k--', color = "grey",  linewidth = 0.5)
#ax[0].text(-0.15, 1.1, "a)", transform=ax[0,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax.set_xlabel('Measured SST [°C]', fontsize = 15)
ax.set_ylabel('MODIS/Aqua SST [°C]', fontsize = 15)
ax.tick_params(axis="x", labelsize = 14)
ax.tick_params(axis="y", labelsize = 14)
ax.legend(loc = 2, frameon = False, fontsize = 15)
ax.set_xlim(10,28)
ax.set_ylim(10,28)






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

nbus_key_xtrain = np.where(X_train.loc[:,"latitude"] >= (-26)) [0]
sbus_key_xtrain = np.where(X_train.loc[:,"latitude"] <= (-26)) [0]

nbus_key_xtest = np.where(X_test.loc[:,"latitude"] >= (-26)) [0]
sbus_key_xtest = np.where(X_test.loc[:,"latitude"] <= (-26)) [0]

print("Length training set north, south:", len(nbus_key_xtrain), len(sbus_key_xtrain))
print("Length testing set north, south:", len(nbus_key_xtest), len(sbus_key_xtest))




# Create histograms of the input and target variables

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10,6))

"========================================================================"
axs[0,0].hist(y_train.iloc[nbus_key_xtrain], bins = 20, histtype = 'step', color = "blue", label = "NBUS")
axs[0,0].hist(y_train.iloc[sbus_key_xtrain], bins = 20, histtype = 'step', color = "grey", label = "SBUS")
axs[0,0].text(-0.15, 1.1, "a)", transform=axs[0,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[0,0].legend(loc = 1, frameon = False, fontsize = 11)
axs[0,0].set_xlabel("pCO$_2$ [µatm]", color = "black", fontsize = 11)
axs[0,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[0,1].hist(y_test.iloc[nbus_key_xtest], bins = 20, histtype = 'step', color = "blue")
axs[0,1].hist(y_test.iloc[sbus_key_xtest], bins = 20, histtype = 'step', color = "grey")
axs[0,1].text(-0.15, 1.1, "b)", transform=axs[0,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[0,1].set_xlabel("pCO$_2$ [µatm]", color = "black", fontsize = 11)
axs[0,1].set_ylabel("Frequency", color = "black", fontsize = 11)

"========================================================================"
axs[1,0].hist(X_train.iloc[nbus_key_xtrain,0], bins = 20, histtype = 'step', color = "blue")
axs[1,0].hist(X_train.iloc[sbus_key_xtrain,0], bins = 20, histtype = 'step', color = "grey")
axs[1,0].text(-0.15, 1.1, "c)", transform=axs[1,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[1,0].set_xlabel("Chl [mg m$^{-3}$]", color = "black", fontsize = 11)
axs[1,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[1,1].hist(X_test.iloc[nbus_key_xtest,0], bins = 20, histtype = 'step', color = "blue")
axs[1,1].hist(X_test.iloc[sbus_key_xtest,0], bins = 20, histtype = 'step', color = "grey")
axs[1,1].text(-0.15, 1.1, "d)", transform=axs[1,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[1,1].set_xlabel("Chl [mg m$^{-3}$]", color = "black", fontsize = 11)
axs[1,1].set_ylabel("Frequency", color = "black", fontsize = 11)

"========================================================================"
axs[2,0].hist(X_train.iloc[nbus_key_xtrain,1], bins = 20, histtype = 'step', color = "blue")
axs[2,0].hist(X_train.iloc[sbus_key_xtrain,1], bins = 20, histtype = 'step', color = "grey")
axs[2,0].text(-0.15, 1.1, "e)", transform=axs[2,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[2,0].set_xlabel("SST [°C]", color = "black", fontsize = 11)
axs[2,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[2,1].hist(X_test.iloc[nbus_key_xtest,1], bins = 20, histtype = 'step', color = "blue")
axs[2,1].hist(X_test.iloc[sbus_key_xtest,1], bins = 20, histtype = 'step', color = "grey")
axs[2,1].text(-0.15, 1.1, "f)", transform=axs[2,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[2,1].set_xlabel("SST [°C]", color = "black", fontsize = 11)
axs[2,1].set_ylabel("Frequency", color = "black", fontsize = 11)

"========================================================================"
axs[3,0].hist(X_train.iloc[nbus_key_xtrain,2], bins = 20, histtype = 'step', color = "blue")
axs[3,0].hist(X_train.iloc[sbus_key_xtrain,2], bins = 20, histtype = 'step', color = "grey")
axs[3,0].text(-0.15, 1.1, "g)", transform=axs[3,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[3,0].set_xlabel("KD-490 [m$^{-1}$]", color = "black", fontsize = 11)
axs[3,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[3,1].hist(X_test.iloc[nbus_key_xtest,2], bins = 20, histtype = 'step', color = "blue")
axs[3,1].hist(X_test.iloc[sbus_key_xtest,2], bins = 20, histtype = 'step', color = "grey")
axs[3,1].text(-0.15, 1.1, "h)", transform=axs[3,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[3,1].set_xlabel("KD-490 [m$^{-1}$]", color = "black", fontsize = 11)
axs[3,1].set_ylabel("Frequency", color = "black", fontsize = 11)

"========================================================================"
axs[4,0].hist(X_train.iloc[nbus_key_xtrain,3], bins = 20, histtype = 'step', color = "blue")
axs[4,0].hist(X_train.iloc[sbus_key_xtrain,3], bins = 20, histtype = 'step', color = "grey")
axs[4,0].text(-0.15, 1.1, "i)", transform=axs[4,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[4,0].set_xlabel("PAR [einstein m$^{-2}$ day$^{-1}$]", color = "black", fontsize = 11)
axs[4,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[4,1].hist(X_test.iloc[nbus_key_xtest,3], bins = 20, histtype = 'step', color = "blue")
axs[4,1].hist(X_test.iloc[sbus_key_xtest,3], bins = 20, histtype = 'step', color = "grey")
axs[4,1].text(-0.15, 1.1, "j)", transform=axs[4,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[4,1].set_xlabel("PAR [einstein m$^{-2}$ day$^{-1}$]", color = "black", fontsize = 11)
axs[4,1].set_ylabel("Frequency", color = "black", fontsize = 11)

plt.show()


# Export training- and testing dataset
#training_data = np.column_stack((X_train, y_train))
#training_file = "DIGETHIC_training_dataset.csv" 
#np.savetxt(training_file, training_data, delimiter = '\t')

#testing_data = np.column_stack((X_test, y_test))
#testing_file = "DIGETHIC_testing_dataset.csv" 
#np.savetxt(testing_file, testing_data, delimiter = '\t')


# Load Testing- and Trainingdataset
training_file = "/Users/csi/private/Data_Scientist/Digethic/ML_project_Siddiqui/DIGETHIC_training_dataset.csv"
with open(training_file, 'r') as nf:
    training_data = np.genfromtxt(training_file, dtype=None, delimiter='\t', skip_header=0)
    training_data = training_data.astype(float)
print(np.shape(training_data))  

testing_file = "/Users/csi/private/Data_Scientist/Digethic/ML_project_Siddiqui/DIGETHIC_testing_dataset.csv"
with open(testing_file, 'r') as nf:
    testing_data = np.genfromtxt(testing_file, dtype=None, delimiter='\t', skip_header=0)
    testing_data = testing_data.astype(float)
print(np.shape(testing_data))  


training_df = pd.DataFrame(training_data, columns = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude', 'pCO2'])
testing_df  = pd.DataFrame(testing_data,  columns = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude', 'pCO2'])

# Choose the appropriate model input variables
feature_cols = ['CHL', 'SST', 'KD490', 'PAR']
#feature_cols = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude']

X_train = training_df[feature_cols]
X_test  = testing_df[feature_cols]

y_train = training_df['pCO2']
y_test  = testing_df['pCO2']


# show correlation of model input variables
corr = X_train.corr()
corr.style.background_gradient(cmap='coolwarm')
#plt.show()


# apply standard scaler on model input variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)



#======================== MODEL TRAINING ========================#


'=================='
# Linear Regression
#poly = PolynomialFeatures(3)
#x_poly = poly.fit_transform(X_train)
lin_reg = LinearRegression()
#predictions_lin_reg = lin_reg.fit(x_poly, y_train).predict(poly.transform(X_test))
predictions_lin_reg = lin_reg.fit(X_train, y_train).predict(X_test)
predictions_lin_reg_train = lin_reg.fit(X_train, y_train).predict(X_train)

importances_lin_reg = lin_reg.coef_
print("Linear Regression:", predictions_lin_reg)
print("importances:", importances_lin_reg)
print(" ")



'========================'
# Decision Tree Regressor
dtr = DecisionTreeRegressor()
predictions_dtr = dtr.fit(X_train, y_train).predict(X_test)
predictions_dtr_train = dtr.fit(X_train, y_train).predict(X_train)

print("Decision Tree Regressor:", predictions_dtr)
importances_dtr = dtr.feature_importances_
#print("Parameters dtr:", dtr.get_params())

# hyperparameters:
# https://www.nbshare.io/notebook/312837011/Decision-Tree-Regression-With-Hyper-Parameter-Tuning-In-Python/
dtr_parameters = {
    "splitter":["best","random"],
    "max_depth" : [1,10,20,30],
    "max_leaf_nodes":[None,2,6,10,20,30,40,50,60,70,80,90] 
}

# "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
# "min_weight_fraction_leaf":[0.0,0.1,0.2,0.3,0.4,0.5],
# "max_features":["log2","sqrt",None],

'========================'
# Random Forest Regressor
random_f_r = RandomForestRegressor(random_state=1)
predictions_rfr = random_f_r.fit(X_train, y_train).predict(X_test)
predictions_rfr_train = random_f_r.fit(X_train, y_train).predict(X_train)

print("Random Forest Regressor:", predictions_rfr)
importances_rfr = random_f_r.feature_importances_
#print("Parameters rfr:", random_f_r.get_params())

# hyperparameters:
# https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/
rfr_parameters = {
    'n_estimators': [25,50,100,150,200],
    'max_depth': [1,10,20,30]
}
# 'max_leaf_nodes': [None,2,6,10,20,30,40,50,60,70,80,90]
# 'max_features': ['sqrt', 'log2', None],

'======================='
# Support Vector Machine
svm = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
#svm_lin = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1)
#svm_lin = SVR(kernel="linear", C=100, gamma="auto")
predictions_svm = svm.fit(X_train, y_train).predict(X_test)
predictions_svm_train = svm.fit(X_train, y_train).predict(X_train)

print("Support Vector Machine:", predictions_svm)
print(" ")
#print("Parameters svm:", svm.get_params())

# hyperparameters:
# https://towardsdatascience.com/hyperparameter-tuning-for-support-vector-machines-c-and-gamma-parameters-6a5097416167
svm_parameters = {
    'gamma': [1,10,20,30],
    'C': [100,125,150,200]
}


'============================'
# Gradient Boosting Regressor
gradient_b_r = GradientBoostingRegressor(random_state=1)
predictions_gbr = gradient_b_r.fit(X_train, y_train).predict(X_test)
predictions_gbr_train = gradient_b_r.fit(X_train, y_train).predict(X_train)

print("Gradient Boosting Regressor:", predictions_gbr)
#print("Parameters gbr:", gradient_b_r.get_params())

# hyperparameters:
# https://educationalresearchtechniques.com/2019/01/14/gradient-boosting-regression-in-python/
gbr_parameters = {
    'n_estimators':[25,50,100,150,200],
    'learning_rate':[0.1,0.25,0.5],
    'max_depth':[1,10,20,30]
}

# 'subsample':[.5,.75,1,1.5,2],
# 'random_state':[1]


'============================='
# K-Nearest Neighbor Regressor
n_neighbor = 5
knn = neighbors.KNeighborsRegressor(n_neighbor, weights= "distance")
predictions_knn = knn.fit(X_train, y_train).predict(X_test)
predictions_knn_train = knn.fit(X_train, y_train).predict(X_train)

print("K-Nearest Neighbor Regressor:", predictions_knn)
#print("Parameters knn:", knn.get_params())

# hyperparameters:
knn_parameters = {
    'n_neighbors': [5,10,25,50,100,150]
}







#======================== MODEL EVALUATION 1 =======================#



#========================     TRAINING      ========================#

'============================='
# Coefficient of Determination
lin_reg_r2_train = r2_score(y_train, predictions_lin_reg_train)
dtr_r2_train     = r2_score(y_train, predictions_dtr_train)
rfr_r2_train     = r2_score(y_train, predictions_rfr_train)
svm_r2_train     = r2_score(y_train, predictions_svm_train)
gbr_r2_train     = r2_score(y_train, predictions_gbr_train)
knn_r2_train     = r2_score(y_train, predictions_knn_train)


'===================='
# Mean Absolute Error
lin_reg_mae_train = mean_absolute_error(y_true= y_train,y_pred= predictions_lin_reg_train)
dtr_mae_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_dtr_train)
rfr_mae_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_rfr_train)
svm_mae_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_svm_train)
gbr_mae_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_gbr_train)
knn_mae_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_knn_train)


'========================'
# Root Mean Squared Error
lin_reg_rmse_train = mean_squared_error(y_train, predictions_lin_reg_train, squared=False)
dtr_rmse_train     = mean_squared_error(y_train, predictions_dtr_train, squared=False)
rfr_rmse_train     = mean_squared_error(y_train, predictions_rfr_train, squared=False)
svm_rmse_train     = mean_squared_error(y_train, predictions_svm_train, squared=False)
gbr_rmse_train     = mean_squared_error(y_train, predictions_gbr_train, squared=False)
knn_rmse_train     = mean_squared_error(y_train, predictions_knn_train, squared=False)


print("================================================") 
print("Model Evaluation Phase 1 Training: R2, MAE, RMSE")
print("================================================")
print("lin reg:", lin_reg_r2_train, lin_reg_mae_train, lin_reg_rmse_train)
print("dtr:", dtr_r2_train, dtr_mae_train, dtr_rmse_train)
print("rfr:", rfr_r2_train, rfr_mae_train, rfr_rmse_train)
print("svm:", svm_r2_train, svm_mae_train, svm_rmse_train)
print("gbr:", gbr_r2_train, gbr_mae_train, gbr_rmse_train)
print("knn:", knn_r2_train, knn_mae_train, knn_rmse_train)
print(" ")



#========================     TESTING      ========================#

'============================='
# Coefficient of Determination
lin_reg_r2 = r2_score(y_test, predictions_lin_reg)
dtr_r2     = r2_score(y_test, predictions_dtr)
rfr_r2     = r2_score(y_test, predictions_rfr)
svm_r2     = r2_score(y_test, predictions_svm)
gbr_r2     = r2_score(y_test, predictions_gbr)
knn_r2     = r2_score(y_test, predictions_knn)


'===================='
# Mean Absolute Error
lin_reg_mae = mean_absolute_error(y_true= y_test,y_pred= predictions_lin_reg)
dtr_mae     = mean_absolute_error(y_true= y_test,y_pred= predictions_dtr)
rfr_mae     = mean_absolute_error(y_true= y_test,y_pred= predictions_rfr)
svm_mae     = mean_absolute_error(y_true= y_test,y_pred= predictions_svm)
gbr_mae     = mean_absolute_error(y_true= y_test,y_pred= predictions_gbr)
knn_mae     = mean_absolute_error(y_true= y_test,y_pred= predictions_knn)


'========================'
# Root Mean Squared Error
lin_reg_rmse = mean_squared_error(y_test, predictions_lin_reg, squared=False)
dtr_rmse     = mean_squared_error(y_test, predictions_dtr, squared=False)
rfr_rmse     = mean_squared_error(y_test, predictions_rfr, squared=False)
svm_rmse     = mean_squared_error(y_test, predictions_svm, squared=False)
gbr_rmse     = mean_squared_error(y_test, predictions_gbr, squared=False)
knn_rmse     = mean_squared_error(y_test, predictions_knn, squared=False)


print("===============================================") 
print("Model Evaluation Phase 1 Testing: R2, MAE, RMSE")
print("===============================================")
print("lin reg:", lin_reg_r2, lin_reg_mae, lin_reg_rmse)
print("dtr:", dtr_r2, dtr_mae, dtr_rmse)
print("rfr:", rfr_r2, rfr_mae, rfr_rmse)
print("svm:", svm_r2, svm_mae, svm_rmse)
print("gbr:", gbr_r2, gbr_mae, gbr_rmse)
print("knn:", knn_r2, knn_mae, knn_rmse)
print(" ")



#================= HYPERPARAMETER OPTIMIZATION ==================#

'=============='
# Decision Tree
#dtr_search = GridSearchCV(estimator= dtr, param_grid= dtr_parameters, cv=3, n_jobs=1, verbose=0, return_train_score=True)
#dtr_search.fit(X_train, y_train)
#print("Best params dtr:", dtr_search.best_params_)
print("Best params dtr:", "max_depth= 20, max_leaf_nodes= None, splitter= best")
dtr = DecisionTreeRegressor(max_depth= 20, max_leaf_nodes= None, splitter= "best")
predictions_dtr_opt = dtr.fit(X_train, y_train).predict(X_test)
predictions_dtr_opt_train = dtr.fit(X_train, y_train).predict(X_train)

'=============='
# Random Forest
#rfr_search = GridSearchCV(estimator= random_f_r, param_grid= rfr_parameters, cv=3, n_jobs=1, verbose=0, return_train_score=True)
#rfr_search.fit(X_train, y_train)
#print("Best params rfr:", rfr_search.best_params_)
print("Best params rfr:", "max_depth= 20, n_estimators= 150")
random_f_r = RandomForestRegressor(random_state=1, max_depth= 20, n_estimators= 150)
predictions_rfr_opt = random_f_r.fit(X_train, y_train).predict(X_test)
predictions_rfr_opt_train = random_f_r.fit(X_train, y_train).predict(X_train)

'=============='
# Support V. M.
#svm_search = GridSearchCV(estimator= svm, param_grid= svm_parameters, cv=3, n_jobs=1, verbose=0, return_train_score=True)
#svm_search.fit(X_train, y_train)
#print("Best params svm:", svm_search.best_params_)
print("Best params svm:", "C= 200, gamma= 30")
svm = SVR(kernel="rbf", C=200, gamma=30, epsilon=0.1)
predictions_svm_opt = svm.fit(X_train, y_train).predict(X_test)
predictions_svm_opt_train = svm.fit(X_train, y_train).predict(X_train)

'=================='
# Gradient Boosting
#gbr_search = GridSearchCV(estimator= gradient_b_r, param_grid= gbr_parameters, cv=3, n_jobs=1, verbose=0, return_train_score=True)
#gbr_search.fit(X_train, y_train)
#print("Best params gbr:", gbr_search.best_params_)
print("Best params gbr:", "learning_rate= 0.1, max_depth= 20, n_estimators= 50")
gradient_b_r = GradientBoostingRegressor(random_state=1, learning_rate= 0.1, max_depth= 20, n_estimators= 50)
predictions_gbr_opt = gradient_b_r.fit(X_train, y_train).predict(X_test)
predictions_gbr_opt_train = gradient_b_r.fit(X_train, y_train).predict(X_train)

'==================='
# K-Nearest Neighbor
#knn_search = GridSearchCV(estimator= knn, param_grid= knn_parameters, cv=3, n_jobs=1, verbose=0, return_train_score=True)
#knn_search.fit(X_train, y_train)
#print("Best params knn:", knn_search.best_params_)
print("Best params knn:", "n_neighbors= 150")
knn = neighbors.KNeighborsRegressor(n_neighbors=150, weights= "distance")
predictions_knn_opt = knn.fit(X_train, y_train).predict(X_test)
predictions_knn_opt_train = knn.fit(X_train, y_train).predict(X_train)



#======================== MODEL EVALUATION 2 =======================#

#========================     TRAINING      ========================#

'============================='
# Coefficient of Determination
lin_reg_r2_train = r2_score(y_train, predictions_lin_reg_train)
dtr_r2_opt_train    = r2_score(y_train, predictions_dtr_opt_train)
rfr_r2_opt_train    = r2_score(y_train, predictions_rfr_opt_train)
svm_r2_opt_train    = r2_score(y_train, predictions_svm_opt_train)
gbr_r2_opt_train    = r2_score(y_train, predictions_gbr_opt_train)
knn_r2_opt_train    = r2_score(y_train, predictions_knn_opt_train)


'===================='
# Mean Absolute Error
lin_reg_mae_train = mean_absolute_error(y_true= y_train,y_pred= predictions_lin_reg_train)
dtr_mae_opt_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_dtr_opt_train)
rfr_mae_opt_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_rfr_opt_train)
svm_mae_opt_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_svm_opt_train)
gbr_mae_opt_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_gbr_opt_train)
knn_mae_opt_train     = mean_absolute_error(y_true= y_train,y_pred= predictions_knn_opt_train)


'========================'
# Root Mean Squared Error
lin_reg_rmse_train = mean_squared_error(y_train, predictions_lin_reg_train, squared=False)
dtr_rmse_opt_train     = mean_squared_error(y_train, predictions_dtr_opt_train, squared=False)
rfr_rmse_opt_train     = mean_squared_error(y_train, predictions_rfr_opt_train, squared=False)
svm_rmse_opt_train     = mean_squared_error(y_train, predictions_svm_opt_train, squared=False)
gbr_rmse_opt_train     = mean_squared_error(y_train, predictions_gbr_opt_train, squared=False)
knn_rmse_opt_train     = mean_squared_error(y_train, predictions_knn_opt_train, squared=False)


print("================================================") 
print("Model Evaluation Phase 2 Training: R2, MAE, RMSE")
print("================================================")
print("lin reg:", lin_reg_r2_train, lin_reg_mae_train, lin_reg_rmse_train)
print("dtr:", dtr_r2_opt_train, dtr_mae_opt_train, dtr_rmse_opt_train)
print("rfr:", rfr_r2_opt_train, rfr_mae_opt_train, rfr_rmse_opt_train)
print("svm:", svm_r2_opt_train, svm_mae_opt_train, svm_rmse_opt_train)
print("gbr:", gbr_r2_opt_train, gbr_mae_opt_train, gbr_rmse_opt_train)
print("knn:", knn_r2_opt_train, knn_mae_opt_train, knn_rmse_opt_train)



#========================     TESTING      ========================#

'============================='
# Coefficient of Determination
lin_reg_r2 = r2_score(y_test, predictions_lin_reg)
dtr_r2_opt     = r2_score(y_test, predictions_dtr_opt)
rfr_r2_opt     = r2_score(y_test, predictions_rfr_opt)
svm_r2_opt     = r2_score(y_test, predictions_svm_opt)
gbr_r2_opt     = r2_score(y_test, predictions_gbr_opt)
knn_r2_opt     = r2_score(y_test, predictions_knn_opt)


'===================='
# Mean Absolute Error
lin_reg_mae = mean_absolute_error(y_true= y_test,y_pred= predictions_lin_reg)
dtr_mae_opt     = mean_absolute_error(y_true= y_test,y_pred= predictions_dtr_opt)
rfr_mae_opt     = mean_absolute_error(y_true= y_test,y_pred= predictions_rfr_opt)
svm_mae_opt     = mean_absolute_error(y_true= y_test,y_pred= predictions_svm_opt)
gbr_mae_opt     = mean_absolute_error(y_true= y_test,y_pred= predictions_gbr_opt)
knn_mae_opt     = mean_absolute_error(y_true= y_test,y_pred= predictions_knn_opt)


'========================'
# Root Mean Squared Error
lin_reg_rmse = mean_squared_error(y_test, predictions_lin_reg, squared=False)
dtr_rmse_opt     = mean_squared_error(y_test, predictions_dtr_opt, squared=False)
rfr_rmse_opt     = mean_squared_error(y_test, predictions_rfr_opt, squared=False)
svm_rmse_opt     = mean_squared_error(y_test, predictions_svm_opt, squared=False)
gbr_rmse_opt     = mean_squared_error(y_test, predictions_gbr_opt, squared=False)
knn_rmse_opt     = mean_squared_error(y_test, predictions_knn_opt, squared=False)


print("===============================================") 
print("Model Evaluation Phase 2 Testing: R2, MAE, RMSE")
print("===============================================")
print("lin reg:", lin_reg_r2, lin_reg_mae, lin_reg_rmse)
print("dtr:", dtr_r2_opt, dtr_mae_opt, dtr_rmse_opt)
print("rfr:", rfr_r2_opt, rfr_mae_opt, rfr_rmse_opt)
print("svm:", svm_r2_opt, svm_mae_opt, svm_rmse_opt)
print("gbr:", gbr_r2_opt, gbr_mae_opt, gbr_rmse_opt)
print("knn:", knn_r2_opt, knn_mae_opt, knn_rmse_opt)







#======================== RESULT GRAPHICS ========================#
#============   Permutation and Feature Importances   ============#

perm_importance_svm = permutation_importance(svm, X_test, y_test, scoring='neg_root_mean_squared_error')
perm_importance_knn = permutation_importance(knn, X_test, y_test, scoring='neg_root_mean_squared_error')

feature_names = ['CHL', 'SST', 'KD490', 'PAR']
#feature_names = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude']
features = np.array(feature_names)

df = pd.DataFrame({'svm': perm_importance_svm.importances_mean, 'knn': perm_importance_knn.importances_mean}, index=features)
ax1 = df.plot.barh(color = {"svm": "navy", "knn": "lightblue"})
plt.xlabel("Permutation Importance")
plt.show()


perm_importance_dtr = permutation_importance(dtr, X_test, y_test, scoring='neg_root_mean_squared_error')
perm_importance_rfr = permutation_importance(random_f_r, X_test, y_test, scoring='neg_root_mean_squared_error')

feature_names = ['CHL', 'SST', 'KD490', 'PAR']
#feature_names = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude']
features = np.array(feature_names)

df = pd.DataFrame({'decision tree': perm_importance_dtr.importances_mean, 'random forest': perm_importance_rfr.importances_mean}, index=features)
ax1 = df.plot.barh(color = {"decision tree": "navy", "random forest": "lightblue"})
plt.xlabel("Permutation Importance")
plt.show()

'============================================'
importances_dtr = dtr.feature_importances_
importances_rfr = random_f_r.feature_importances_

left = [1,2,3,4]
height_dtr = importances_dtr
height_rfr = importances_rfr
tick_label = ['CHL', 'SST', 'KD490', 'PAR']
#tick_label = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude']
df = pd.DataFrame({'decision tree': height_dtr, 'random forest': height_rfr}, index = tick_label)
ax = df.plot.barh(color = {"decision tree": "navy", "random forest": "lightblue"})
plt.xlabel("Feature Importance")
plt.show()





'======================================================================'
#===================   PLOTTING PCO2 PREDICTIONS   ====================#

# Linear regression:
lin_reg_eval_1_train = 'R$^2$= '+ str('{:.3f}'.format(lin_reg_r2_train))
lin_reg_eval_2_train = 'MAE= '+ str('{:.3f}'.format(lin_reg_mae_train))
lin_reg_eval_3_train = 'RMSE= '+ str('{:.3f}'.format(lin_reg_rmse_train))

lin_reg_eval_1 = 'R$^2$= '+ str('{:.3f}'.format(lin_reg_r2))
lin_reg_eval_2 = 'MAE= '+ str('{:.3f}'.format(lin_reg_mae))
lin_reg_eval_3 = 'RMSE= '+ str('{:.3f}'.format(lin_reg_rmse))

# kNN:
knn_eval_1_train = 'R$^2$= '+ str('{:.3f}'.format(knn_r2_opt_train))
knn_eval_2_train = 'MAE= '+ str('{:.3f}'.format(knn_mae_opt_train))
knn_eval_3_train = 'RMSE= '+ str('{:.3f}'.format(knn_rmse_opt_train))

knn_eval_1 = 'R$^2$= '+ str('{:.3f}'.format(knn_r2_opt))
knn_eval_2 = 'MAE= '+ str('{:.3f}'.format(knn_mae_opt))
knn_eval_3 = 'RMSE= '+ str('{:.3f}'.format(knn_rmse_opt))

# SVM:
svm_eval_1_train = 'R$^2$= '+ str('{:.3f}'.format(svm_r2_opt_train))
svm_eval_2_train = 'MAE= '+ str('{:.3f}'.format(svm_mae_opt_train))
svm_eval_3_train = 'RMSE= '+ str('{:.3f}'.format(svm_rmse_opt_train))

svm_eval_1 = 'R$^2$= '+ str('{:.3f}'.format(svm_r2_opt))
svm_eval_2 = 'MAE= '+ str('{:.3f}'.format(svm_mae_opt))
svm_eval_3 = 'RMSE= '+ str('{:.3f}'.format(svm_rmse_opt))

# Decision tree:
dtr_eval_1_train = 'R$^2$= '+ str('{:.3f}'.format(dtr_r2_opt_train))
dtr_eval_2_train = 'MAE= '+ str('{:.3f}'.format(dtr_mae_opt_train))
dtr_eval_3_train = 'RMSE= '+ str('{:.3f}'.format(dtr_rmse_opt_train))

dtr_eval_1 = 'R$^2$= '+ str('{:.3f}'.format(dtr_r2_opt))
dtr_eval_2 = 'MAE= '+ str('{:.3f}'.format(dtr_mae_opt))
dtr_eval_3 = 'RMSE= '+ str('{:.3f}'.format(dtr_rmse_opt))

# Random forest:
rfr_eval_1_train = 'R$^2$= '+ str('{:.3f}'.format(rfr_r2_opt_train))
rfr_eval_2_train = 'MAE= '+ str('{:.3f}'.format(rfr_mae_opt_train))
rfr_eval_3_train = 'RMSE= '+ str('{:.3f}'.format(rfr_rmse_opt_train))

rfr_eval_1 = 'R$^2$= '+ str('{:.3f}'.format(rfr_r2_opt))
rfr_eval_2 = 'MAE= '+ str('{:.3f}'.format(rfr_mae_opt))
rfr_eval_3 = 'RMSE= '+ str('{:.3f}'.format(rfr_rmse_opt))

# Gradient boosting:
gbr_eval_1_train = 'R$^2$= '+ str('{:.3f}'.format(gbr_r2_opt_train))
gbr_eval_2_train = 'MAE= '+ str('{:.3f}'.format(gbr_mae_opt_train))
gbr_eval_3_train = 'RMSE= '+ str('{:.3f}'.format(gbr_rmse_opt_train))

gbr_eval_1 = 'R$^2$= '+ str('{:.3f}'.format(gbr_r2_opt))
gbr_eval_2 = 'MAE= '+ str('{:.3f}'.format(gbr_mae_opt))
gbr_eval_3 = 'RMSE= '+ str('{:.3f}'.format(gbr_rmse_opt))

x_eval = np.linspace(300,700,3)
y_eval = np.linspace(300,700,3)



'========================================================'
fig1, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 4))

lin_reg_1_t = ax[0,0].scatter(y=predictions_lin_reg_train, x=y_train, s=10, label = lin_reg_eval_1_train, color = "white")
lin_reg_2_t = ax[0,0].scatter(y=predictions_lin_reg_train, x=y_train, s=10, label = lin_reg_eval_2_train, color = "white")
lin_reg_3_t = ax[0,0].scatter(y=predictions_lin_reg_train, x=y_train, s=10, label = lin_reg_eval_3_train,  color = "darkblue")
ax[0,0].plot(x_eval, y_eval, 'k--', color = "grey",  linewidth = 0.5)
ax[0,0].text(-0.15, 1.1, "a)", transform=ax[0,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[0,0].set_xlabel('Measured pCO$_2$ [µatm]')
ax[0,0].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[0,0].set_title("Model Training")
ax[0,0].legend(loc = 2, frameon = False, fontsize = 7)
ax[0,0].set_xlim(300,700)
ax[0,0].set_ylim(300,700)

lin_reg_1 = ax[0,1].scatter(y=predictions_lin_reg, x=y_test, s=10, label = lin_reg_eval_1,  color = "white")
lin_reg_2 = ax[0,1].scatter(y=predictions_lin_reg, x=y_test, s=10, label = lin_reg_eval_2,  color = "white")
lin_reg_3 = ax[0,1].scatter(y=predictions_lin_reg, x=y_test, s=10, label = lin_reg_eval_3,  color = "darkblue")
ax[0,1].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[0,1].text(-0.15, 1.1, "b)", transform=ax[0,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[0,1].set_xlabel('Measured pCO$_2$ [µatm]')
ax[0,1].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[0,1].set_title("Model Validation")
ax[0,1].legend(loc = 2, frameon = False, fontsize = 7)
ax[0,1].set_xlim(300,700)
ax[0,1].set_ylim(300,700)

delta_lin_reg = predictions_lin_reg - y_test
ax[0,2].hist(delta_lin_reg, bins = 100, histtype = 'step', color = "darkblue")
ax[0,2].text(-0.15, 1.1, "c)", transform=ax[0,2].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[0,2].set_xlabel('Difference predicted/measured')
ax[0,2].set_ylabel('Number of points')
ax[0,2].set_title("Histogram of residuals")
ax[0,2].set_xlim((-200), 200)

'================================='
knn_1_t = ax[1,0].scatter(y=predictions_knn_opt_train, x=y_train, s=10, label = knn_eval_1_train,  color = "white")
knn_2_t = ax[1,0].scatter(y=predictions_knn_opt_train, x=y_train, s=10, label = knn_eval_2_train,  color = "white")
knn_3_t = ax[1,0].scatter(y=predictions_knn_opt_train, x=y_train, s=10, label = knn_eval_3_train,  color = "darkblue")
ax[1,0].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[1,0].text(-0.15, 1.1, "d)", transform=ax[1,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[1,0].set_xlabel('Measured pCO$_2$ [µatm]')
ax[1,0].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[1,0].legend(loc = 2, frameon = False, fontsize = 7)
ax[1,0].set_xlim(300,700)
ax[1,0].set_ylim(300,700)

knn_1 = ax[1,1].scatter(y=predictions_knn_opt, x=y_test, s=10, label = knn_eval_1, color = "white")
knn_2 = ax[1,1].scatter(y=predictions_knn_opt, x=y_test, s=10, label = knn_eval_2, color = "white")
knn_3 = ax[1,1].scatter(y=predictions_knn_opt, x=y_test, s=10, label = knn_eval_3, color = "darkblue")
ax[1,1].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[1,1].text(-0.15, 1.1, "e)", transform=ax[1,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[1,1].set_xlabel('Measured pCO$_2$ [µatm]')
ax[1,1].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[1,1].legend(loc = 2, frameon = False, fontsize = 7)
ax[1,1].set_xlim(300,700)
ax[1,1].set_ylim(300,700)

delta_knn = predictions_knn_opt - y_test
ax[1,2].hist(delta_knn, bins = 100, histtype = 'step', color = "darkblue")
ax[1,2].text(-0.15, 1.1, "f)", transform=ax[1,2].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[1,2].set_xlabel('Difference predicted/measured')
ax[1,2].set_ylabel('Number of points')
ax[1,2].set_xlim((-50), 50)

'================================='
svm_1_t = ax[2,0].scatter(y=predictions_svm_opt_train, x=y_train, s=10, label = svm_eval_1_train,  color = "white")
svm_2_t = ax[2,0].scatter(y=predictions_svm_opt_train, x=y_train, s=10, label = svm_eval_2_train,  color = "white")
svm_3_t = ax[2,0].scatter(y=predictions_svm_opt_train, x=y_train, s=10, label = svm_eval_3_train,  color = "darkblue")
ax[2,0].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[2,0].text(-0.15, 1.1, "g)", transform=ax[2,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[2,0].set_xlabel('Measured pCO$_2$ [µatm]')
ax[2,0].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[2,0].legend(loc = 2, frameon = False, fontsize = 7)
ax[2,0].set_xlim(300,700)
ax[2,0].set_ylim(300,700)

svm_1 = ax[2,1].scatter(y=predictions_svm_opt, x=y_test, s=10, label = svm_eval_1,  color = "white")
svm_2 = ax[2,1].scatter(y=predictions_svm_opt, x=y_test, s=10, label = svm_eval_2,  color = "white")
svm_3 = ax[2,1].scatter(y=predictions_svm_opt, x=y_test, s=10, label = svm_eval_3,  color = "darkblue")
ax[2,1].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[2,1].text(-0.15, 1.1, "h)", transform=ax[2,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[2,1].set_xlabel('Measured pCO$_2$ [µatm]')
ax[2,1].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[2,1].legend(loc = 2, frameon = False, fontsize = 7)
ax[2,1].set_xlim(300,700)
ax[2,1].set_ylim(300,700)

delta_svm = predictions_svm_opt - y_test
ax[2,2].hist(delta_svm, bins = 100, histtype = 'step', color = "darkblue")
ax[2,2].text(-0.15, 1.1, "i)", transform=ax[2,2].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[2,2].set_xlabel('Difference predicted/measured')
ax[2,2].set_xlim((-50), 50)
ax[2,2].set_ylabel('Number of points')

plt.show()




fig2, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 4))

'================================='
dtr_1_t = ax[0,0].scatter(y=predictions_dtr_opt_train, x=y_train, s=10, label = dtr_eval_1_train,  color = "white")
dtr_2_t = ax[0,0].scatter(y=predictions_dtr_opt_train, x=y_train, s=10, label = dtr_eval_2_train,  color = "white")
dtr_3_t = ax[0,0].scatter(y=predictions_dtr_opt_train, x=y_train, s=10, label = dtr_eval_3_train,  color = "darkblue")
ax[0,0].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[0,0].text(-0.15, 1.1, "a)", transform=ax[0,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[0,0].set_xlabel('Measured pCO$_2$ [µatm]')
ax[0,0].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[0,0].set_title("Model Training")
ax[0,0].legend(loc = 2, frameon = False, fontsize = 7)
ax[0,0].set_xlim(300,700)
ax[0,0].set_ylim(300,700)

dtr_1 = ax[0,1].scatter(y=predictions_dtr_opt, x=y_test, s=10, label = dtr_eval_1,  color = "white")
dtr_2 = ax[0,1].scatter(y=predictions_dtr_opt, x=y_test, s=10, label = dtr_eval_2,  color = "white")
dtr_3 = ax[0,1].scatter(y=predictions_dtr_opt, x=y_test, s=10, label = dtr_eval_3,  color = "darkblue")
ax[0,1].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[0,1].text(-0.15, 1.1, "b)", transform=ax[0,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[0,1].set_xlabel('Measured pCO$_2$ [µatm]')
ax[0,1].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[0,1].set_title("Model Validation")
ax[0,1].legend(loc = 2, frameon = False, fontsize = 7)
ax[0,1].set_xlim(300,700)
ax[0,1].set_ylim(300,700)

delta_dtr = predictions_dtr_opt - y_test
ax[0,2].hist(delta_dtr, bins = 100, histtype = 'step', color = "darkblue")
ax[0,2].text(-0.15, 1.1, "c)", transform=ax[0,2].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[0,2].set_title("Histogram of residuals")
ax[0,2].set_xlabel('Difference predicted/measured')
ax[0,2].set_ylabel('Number of points')
ax[0,2].set_xlim((-50), 50)

'================================='
rfr_1_t = ax[1,0].scatter(y=predictions_rfr_opt_train, x=y_train, s=10, label = rfr_eval_1_train,  color = "white")
rfr_2_t = ax[1,0].scatter(y=predictions_rfr_opt_train, x=y_train, s=10, label = rfr_eval_2_train,  color = "white")
rfr_3_t = ax[1,0].scatter(y=predictions_rfr_opt_train, x=y_train, s=10, label = rfr_eval_3_train,  color = "darkblue")
ax[1,0].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[1,0].text(-0.15, 1.1, "d)", transform=ax[1,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[1,0].set_xlabel('Measured pCO$_2$ [µatm]')
ax[1,0].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[1,0].legend(loc = 2, frameon = False, fontsize = 7)
ax[1,0].set_xlim(300,700)
ax[1,0].set_ylim(300,700)

rfr_1 = ax[1,1].scatter(y=predictions_rfr_opt, x=y_test, s=10, label = rfr_eval_1, color = "white")
rfr_2 = ax[1,1].scatter(y=predictions_rfr_opt, x=y_test, s=10, label = rfr_eval_2, color = "white")
rfr_3 = ax[1,1].scatter(y=predictions_rfr_opt, x=y_test, s=10, label = rfr_eval_3, color = "darkblue")
ax[1,1].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[1,1].text(-0.15, 1.1, "e)", transform=ax[1,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[1,1].set_xlabel('Measured pCO$_2$ [µatm]')
ax[1,1].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[1,1].legend(loc = 2, frameon = False, fontsize = 7)
ax[1,1].set_xlim(300,700)
ax[1,1].set_ylim(300,700)

delta_knn = predictions_knn_opt - y_test
ax[1,2].hist(delta_knn, bins = 100, histtype = 'step', color = "darkblue")
ax[1,2].text(-0.15, 1.1, "f)", transform=ax[1,2].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[1,2].set_xlabel('Difference predicted/measured')
ax[1,2].set_ylabel('Number of points')
ax[1,2].set_xlim((-50), 50)

'================================='
gbr_1_t = ax[2,0].scatter(y=predictions_gbr_opt_train, x=y_train, s=10, label = gbr_eval_1_train, color = "white")
gbr_2_t = ax[2,0].scatter(y=predictions_gbr_opt_train, x=y_train, s=10, label = gbr_eval_2_train, color = "white")
gbr_3_t = ax[2,0].scatter(y=predictions_gbr_opt_train, x=y_train, s=10, label = gbr_eval_3_train, color = "darkblue")
ax[2,0].plot(x_eval, y_eval, 'k--', color = "grey", linewidth = 0.5)
ax[2,0].text(-0.15, 1.1, "g)", transform=ax[2,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[2,0].set_xlabel('Measured pCO$_2$ [µatm]')
ax[2,0].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[2,0].legend(loc = 2, frameon = False, fontsize = 7)
ax[2,0].set_xlim(300,700)
ax[2,0].set_ylim(300,700)

gbr_1 = ax[2,1].scatter(y=predictions_gbr_opt, x=y_test, s=10, label = gbr_eval_1,  color = "white")
gbr_2 = ax[2,1].scatter(y=predictions_gbr_opt, x=y_test, s=10, label = gbr_eval_2,  color = "white")
gbr_3 = ax[2,1].scatter(y=predictions_gbr_opt, x=y_test, s=10, label = gbr_eval_3,  color = "darkblue")
ax[2,1].plot(x_eval, y_eval, 'k--', color = "grey",  linewidth = 0.5)
ax[2,1].text(-0.15, 1.1, "h)", transform=ax[2,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[2,1].set_xlabel('Measured pCO$_2$ [µatm]')
ax[2,1].set_ylabel('Predicted pCO$_2$ [µatm]')
ax[2,1].legend(loc = 2, frameon = False, fontsize = 7)
ax[2,1].set_xlim(300,700)
ax[2,1].set_ylim(300,700)

delta_gbr = predictions_gbr_opt - y_test
ax[2,2].hist(delta_gbr, bins = 100, histtype = 'step', color = "darkblue")
ax[2,2].text(-0.15, 1.1, "i)", transform=ax[2,2].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
ax[2,2].set_xlabel('Difference predicted/measured')
ax[2,2].set_ylabel('Number of points')
ax[2,2].set_xlim((-50), 50)

plt.show()
