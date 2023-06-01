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

nbus_key_xtrain = np.where(X_train.loc[:,"latitude"] >= (-26)) [0]
sbus_key_xtrain = np.where(X_train.loc[:,"latitude"] <= (-26)) [0]

#nbus_key_ytrain = np.where(y_train.loc[:,"latitude"] >= (-26)) [0]
#sbus_key_ytrain = np.where(y_train.loc[:,"latitude"] <= (-26)) [0]

nbus_key_xtest = np.where(X_test.loc[:,"latitude"] >= (-26)) [0]
sbus_key_xtest = np.where(X_test.loc[:,"latitude"] <= (-26)) [0]

#nbus_key_ytest = np.where(y_test.loc[:,"latitude"] >= (-26)) [0]
#sbus_key_ytest = np.where(y_test.loc[:,"latitude"] <= (-26)) [0]

print(len(nbus_key_xtest), len(sbus_key_xtest), len(nbus_key_xtrain), len(sbus_key_xtrain))
print(nbus_key_xtest)



# Create histograms of the input and target variables

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10,6))

"========================================================================"
axs[0,0].hist(y_train.iloc[nbus_key_xtrain], bins = 20, histtype = 'step', color = "blue")
axs[0,0].hist(y_train.iloc[sbus_key_xtrain], bins = 20, histtype = 'step', color = "grey")
axs[0,0].text(-0.15, 1.1, "a)", transform=axs[0,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[0,0].set_xlabel("pCO$_2$", color = "black", fontsize = 11)
axs[0,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[0,1].hist(y_test.iloc[nbus_key_xtest], bins = 20, histtype = 'step', color = "blue")
axs[0,1].hist(y_test.iloc[sbus_key_xtest], bins = 20, histtype = 'step', color = "grey")
axs[0,1].text(-0.15, 1.1, "b)", transform=axs[0,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[0,1].set_xlabel("pCO$_2$", color = "black", fontsize = 11)
axs[0,1].set_ylabel("Frequency", color = "black", fontsize = 11)

"========================================================================"
axs[1,0].hist(X_train.iloc[nbus_key_xtrain,0], bins = 20, histtype = 'step', color = "blue")
axs[1,0].hist(X_train.iloc[sbus_key_xtrain,0], bins = 20, histtype = 'step', color = "grey")
axs[1,0].text(-0.15, 1.1, "c)", transform=axs[1,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[1,0].set_xlabel("Chl", color = "black", fontsize = 11)
axs[1,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[1,1].hist(X_test.iloc[nbus_key_xtest,0], bins = 20, histtype = 'step', color = "blue")
axs[1,1].hist(X_test.iloc[sbus_key_xtest,0], bins = 20, histtype = 'step', color = "grey")
axs[1,1].text(-0.15, 1.1, "d)", transform=axs[1,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[1,1].set_xlabel("Chl", color = "black", fontsize = 11)
axs[1,1].set_ylabel("Frequency", color = "black", fontsize = 11)

"========================================================================"
axs[2,0].hist(X_train.iloc[nbus_key_xtrain,1], bins = 20, histtype = 'step', color = "blue")
axs[2,0].hist(X_train.iloc[sbus_key_xtrain,1], bins = 20, histtype = 'step', color = "grey")
axs[2,0].text(-0.15, 1.1, "e)", transform=axs[2,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[2,0].set_xlabel("SST", color = "black", fontsize = 11)
axs[2,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[2,1].hist(X_test.iloc[nbus_key_xtest,1], bins = 20, histtype = 'step', color = "blue")
axs[2,1].hist(X_test.iloc[sbus_key_xtest,1], bins = 20, histtype = 'step', color = "grey")
axs[2,1].text(-0.15, 1.1, "f)", transform=axs[2,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[2,1].set_xlabel("SST", color = "black", fontsize = 11)
axs[2,1].set_ylabel("Frequency", color = "black", fontsize = 11)

"========================================================================"
axs[3,0].hist(X_train.iloc[nbus_key_xtrain,2], bins = 20, histtype = 'step', color = "blue")
axs[3,0].hist(X_train.iloc[sbus_key_xtrain,2], bins = 20, histtype = 'step', color = "grey")
axs[3,0].text(-0.15, 1.1, "g)", transform=axs[3,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[3,0].set_xlabel("KD-490", color = "black", fontsize = 11)
axs[3,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[3,1].hist(X_test.iloc[nbus_key_xtest,2], bins = 20, histtype = 'step', color = "blue")
axs[3,1].hist(X_test.iloc[sbus_key_xtest,2], bins = 20, histtype = 'step', color = "grey")
axs[3,1].text(-0.15, 1.1, "h)", transform=axs[3,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[3,1].set_xlabel("KD-490", color = "black", fontsize = 11)
axs[3,1].set_ylabel("Frequency", color = "black", fontsize = 11)

"========================================================================"
axs[4,0].hist(X_train.iloc[nbus_key_xtrain,3], bins = 20, histtype = 'step', color = "blue")
axs[4,0].hist(X_train.iloc[sbus_key_xtrain,3], bins = 20, histtype = 'step', color = "grey")
axs[4,0].text(-0.15, 1.1, "i)", transform=axs[4,0].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[4,0].set_xlabel("PAR", color = "black", fontsize = 11)
axs[4,0].set_ylabel("Frequency", color = "black", fontsize = 11)

axs[4,1].hist(X_test.iloc[nbus_key_xtest,3], bins = 20, histtype = 'step', color = "blue")
axs[4,1].hist(X_test.iloc[sbus_key_xtest,3], bins = 20, histtype = 'step', color = "grey")
axs[4,1].text(-0.15, 1.1, "j)", transform=axs[4,1].transAxes, fontsize = 12, fontweight="bold", va="top", ha="center")
axs[4,1].set_xlabel("PAR", color = "black", fontsize = 11)
axs[4,1].set_ylabel("Frequency", color = "black", fontsize = 11)

plt.show()

# Export training- and testing dataset
training_data = np.column_stack((X_train, y_train))
training_file = "DIGETHIC_training_dataset.csv" 
np.savetxt(training_file, training_data, delimiter = '\t')

testing_data = np.column_stack((X_test, y_test))
testing_file = "DIGETHIC_testing_dataset.csv" 
np.savetxt(testing_file, testing_data, delimiter = '\t')




# Load Testing- and Trainingdataset
# --> remember to first make sure all feature columns were used when generating the ultimate files!
#     (to make sure same inputs were used when taking latitude & longitude into account)

feature_cols = ['CHL', 'SST', 'KD490', 'PAR']
X_train = X_train[feature_cols]
X_test = X_test[feature_cols]

#y_train = y_train["pCO2"]
#y_test = y_test["pCO2"]


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
#predictions_lin_reg = lin_reg.fit(x_poly, y_train).predict(poly.transform(X_test))
predictions_lin_reg = lin_reg.fit(X_train, y_train).predict(X_test)
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
svm = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
#svm_lin = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1)
#svm_lin = SVR(kernel="linear", C=100, gamma="auto")
predictions_svm = svm.fit(X_train, y_train).predict(X_test)
#accuracy_svm_lin = accuracy_score(y_test, predictions_svm_lin)
#accuracy_svm_lin = round(accuracy_svm_lin,2)
print("Support Vector Machine:", predictions_svm)
#print("accuracy:", accuracy_svm_lin)
print(" ")

perm_importance = permutation_importance(svm, X_test, y_test)
print("Shape:", np.shape(perm_importance), perm_importance)
feature_names = ['CHL', 'SST', 'KD490', 'PAR']
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


print("===============================") 
print("Model Evaluation: R2, MAE, RMSE")
print("===============================")
print("lin reg:", lin_reg_r2, lin_reg_mae, lin_reg_rmse)
print("dtr:", dtr_r2, dtr_mae, dtr_rmse)
print("rfr:", rfr_r2, rfr_mae, rfr_rmse)
print("svm:", svm_r2, svm_mae, svm_rmse)
print("gbr:", gbr_r2, gbr_mae, gbr_rmse)
print("knn:", knn_r2, knn_mae, knn_rmse)




#======================== RESULT GRAPHICS ========================#


#left = [1,2,3,4,5,6]
#left = [1,2,3,4]
left = [1,2,3,4,5]
height_dtr = importances_dtr
height_rfr = importances_rfr
#height_svm = perm_importance #importances_svm
#tick_label = ['CHL', 'SST', 'KD490', 'PAR', 'latitude', 'longitude']
tick_label = ['CHL', 'SST', 'KD490', 'PAR']
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

ax[1,1].scatter(y=predictions_svm, x=y_test)
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

