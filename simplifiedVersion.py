#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 10:41:09 2023

@author: Jose Muniz, Matheus Teixeira, Ruben Yury, Shiaw-Rong Lin
"""
# Imports
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

#   1. Data exploration: a complete review and analysis of the dataset including:
#   1.1. Load and describe data elements (columns), provide descriptions & types,
# ranges and values of elements as appropriate. – use pandas, numpy and any
# other python packages.

# Load Data
df_KSI = pd.read_csv(Path("D:/pCloudFolder/Repositories/SuperLearningProject/KSI.csv"))

# provide descriptions & types
print(df_KSI.info())
print(df_KSI.columns.values)
print(df_KSI.head(3))
print(df_KSI.shape)
print(df_KSI.dtypes)
#-------------------
 
#   1.2. Statistical assessments including means, averages, correlations
print(df_KSI.describe())
#-------------------

#   1.3. Missing data evaluations – use pandas, numpy and any other python
# packages
print(df_KSI.isnull().sum())
print(df_KSI.isna().sum())
df_KSI_nan = df_KSI.replace(' ', np.nan,  regex=False)
print(df_KSI_nan.isna().sum()/len(df_KSI_nan)*100)
#-------------------

#   1.4. Graphs and visualizations – use pandas, matplotlib, seaborn, numpy and
# any other python packages, you also can use power BI desktop.
##############################################################################

#   2. Data modelling:
#   2.1. Data transformations – includes handling missing data, categorical
# data management, data normalization and standardizations as needed.

# label
df_KSI['ACCLASS'] = np.where(df_KSI['ACCLASS'] ==
                                     'Property Damage Only', 'No-Fatal',
                                     df_KSI['ACCLASS'])

df_KSI['ACCLASS'] = np.where(df_KSI['ACCLASS'] ==
                                     'Non-Fatal Injury', 'No-Fatal',
                                     df_KSI['ACCLASS'])

df_KSI['ACCLASS_TARG'] = df_KSI['ACCLASS'].replace({'Fatal': 1, 'No-Fatal': 0})

df_KSI = df_KSI.drop(["ACCLASS"], axis=1)

# replace missing values in the label based on the Injury field,
# If the injury field is missing, drop the line
for i, value in enumerate(df_KSI['ACCLASS_TARG'].isna()):
  if value == True:
    if df_KSI['INJURY'][i] != ' ':
      if df_KSI['INJURY'][i] == 'Fatal':
        df_KSI['ACCLASS_TARG'][i] = 1
      else:
        df_KSI['ACCLASS_TARG'][i] = 0
    else:
      df_KSI = df_KSI['ACCLASS_TARG'].drop(index=[i])
#-------------------

# Data cleaning
df_KSI_Dropped = df_KSI.drop(["INDEX_", "ACCNUM", "YEAR", "OFFSET", "WARDNUM",
                              "INJURY", "FATAL_NO", "DIVISION", "ObjectId",
                              "OFFSET", "X", "Y"], axis=1)
#-------------------

# Categorical
categorical_columns_with_Yes = ["CYCLIST","PEDESTRIAN","AUTOMOBILE",
                                "MOTORCYCLE", "TRUCK","TRSN_CITY_VEH",
                                "EMERG_VEH", "PASSENGER","SPEEDING","AG_DRIV",
                                "REDLIGHT","ALCOHOL","DISABILITY"]

for column in categorical_columns_with_Yes:
    df_KSI_Dropped[column] = df_KSI_Dropped[column].replace({'Yes': 1,
                                                             np.nan: 0})

objdtype_cols = df_KSI_Dropped.select_dtypes(["object"]).columns
df_KSI_Dropped[objdtype_cols] = df_KSI_Dropped[objdtype_cols].astype('category')
#-------------------

df_KSI_Dropped['DATE'] = pd.to_datetime(df_KSI_Dropped['DATE'])
df_KSI_Dropped['MONTH'] = df_KSI_Dropped['DATE'].dt.month
df_KSI_Dropped['WEEKDAY'] = df_KSI_Dropped['DATE'].dt.weekday
df_KSI_Dropped = df_KSI_Dropped.drop(["DATE"], axis=1)


# print(df_KSI_Dropped.isna().sum()/len(df_KSI_Dropped)*100)

#   2.2. Feature selection – use pandas and sci-kit learn. (The group needs to
# justify each feature used and any data columns discarded)
#   2.3. Train, Test data splitting – use numpy, sci-kit learn.
#   2.4. Managing imbalanced classes if needed. Check here for info:
# https://elitedatascience.com/imbalanced-classes
#   2.5. Use pipelines class to streamline all the pre-processing
#   transformations.
print(df_KSI_Dropped.dtypes)

float_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
                 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING',
                 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']

int_columns = ['TIME', 'MONTH', 'WEEKDAY']

object_columns = ['ROAD_CLASS', 'DISTRICT', 'LOCCOORD', 'TRAFFCTL',
                  'LIGHT', 'RDSFCOND', 'INVTYPE',
                  'INVAGE', 'INITDIR', 'VEHTYPE', 'MANOEUVER', 'DRIVACT',
                  'DRIVCOND', 'HOOD_158',
                  
                  'STREET1', 'STREET2', 'ACCLOC', 'PEDTYPE',
                  'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND']

ramito = ['VISIBILITY']

# Concatenate all feature names
features = float_columns + int_columns + object_columns + ramito

# Split the data into X and y
X = df_KSI_Dropped[features]
y = df_KSI_Dropped['ACCLASS_TARG']

# Create transformers
numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

special = Pipeline(steps=[
    ('imputer1', SimpleImputer(strategy='most_frequent')),
    # ('imputer1', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot1', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, float_columns + int_columns),
        ('cat', categorical_transformer, object_columns),
        ('special2', special, ramito)
    ])

# Combine preprocessor and model in one pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestRegressor(random_state=42))])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
model.fit(X_train, y_train)

# Get feature importances
importances = model.named_steps['classifier'].feature_importances_

# Get one hot encoder feature names
ohe_feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(object_columns)

ohe_feature_names2 = model.named_steps['preprocessor'].transformers_[2][1].named_steps['onehot1'].get_feature_names_out(ramito)

# Concat numerical columns names and one-hot encoder column names
feature_names = np.concatenate([float_columns, int_columns, ohe_feature_names, ohe_feature_names2])

# Create a DataFrame to store the features and their respective importance
important_features_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the DataFrame in descending order of importance
important_features_df = important_features_df.sort_values(by='Importance', ascending=False)

# Plot the top 10 important features
important_features_df.head(15).plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()
