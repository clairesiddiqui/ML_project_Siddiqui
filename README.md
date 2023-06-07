# ML_project_Siddiqui

This collection comprises two python scripts that were used to perform sea surface pCO2 reconstructions with the help of 
Machine Learning and remote sensing data, and underlying .csv files used for training and testing the models.


data_processing.py
- this file merely includes the code to match remote sensing data products from MODIS/Aqua satellite with the field dataset


model_training.py
- this file includes all code that has been applied to generate the results and plots of this study and is structured into two parts
- The first part is used for data cleaning (removing NANs), evaluating satellite SST data, splitting the input file into train- and
  test-datasets, creating histograms of the target and model input variables, and for saving the test- and train- datasets as .csv files
  to be loaded for model application. 
- The second part is used for loading the test- and train-datasets (DIGETHIC_training_dataset.csv, DIGETHIC_testing_dataset.csv),
  setting up the relevant input parameters to be used for model training (e.g., including/excluding latitude and longitude),
  giving out the correlation matrix of input variables, scaling the input data using StandardScaler, 
  running the individual models (without optimized hyperparameters), 
  evaluating the first model runs by calculating R2, MAE and RMSE, individually for training and testing,
  optimizing hyperparameters using GridSearchCV and running models with the tuned parameters,
  evaluating the second model runs by calculating R2, MAE and RMSE, individually for training and testing,
  and finally for generating the graphical results (Permutation importances and pCO2 predictions)
  
  
  DIGETHIC_training_dataset.csv
  - this file contains all the parameters used for training the model, including the target variable pCO2
  - file structure: 7 columns, tab-separated, order: Chlorophyll, SST, KD-490, PAR, latitude, longitude, pCO2 
  
  
  DIGETHIC_testing_dataset.csv
  - this file contains all the parameters used for testing the model, including the target variable pCO2
  - file structure: 7 columns, tab-separated, order: Chlorophyll, SST, KD-490, PAR, latitude, longitude, pCO2 


.venv (folder)
- this folder contains the virtual environment with packages that need to be installed for running the model_training.py script











  
