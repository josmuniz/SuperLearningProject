#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 03:49:21 2023

@author: josemuniz
"""


import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas_profiling as profile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib


#Load the data and carry out initial exploration (10 marks)
#Download the  "KSI.csv" from here.
#load the file into a data frame named df_firstname
#carry out initial analysis per column and show the types, number of missing values, range per feature(min,max), mean, median.
#Summarize your findings in the below box, make sure to use your own words and give a full picture of the dataset. 
#Generate a pairplot showing the relationship between the all the columns, take a screenshot.


#pandas setting
pd.set_option('display.max_columns', None)

#A - Load Data

df_KSI = pd.read_csv(Path("/Users/josemuniz/Desktop/COMP247/Project/KSI.csv"))

#Carry out initial analysis per column and show the types, number of missing values, 
#range per feature(min,max), mean and median  information.
print(df_KSI.info())
print(df_KSI.columns.values)
print(df_KSI.head(3))
print(df_KSI.describe())
print(df_KSI.shape)
print(df_KSI.dtypes)

print(df_KSI.isnull().sum())
print(df_KSI.isna().sum())

# Missing values for other columns replace ' ' with nan.

df_KSI = df_KSI.replace(' ', np.nan,  regex=False) #for No categorical number column

# printing percentage of missing values for each feature
print(df_KSI.isna().sum()/len(df_KSI)*100)

fig, ax = plt.subplots(figsize=(15,7))
# heatmap to visualize features with most missing values
sns.heatmap(df_KSI.isnull(), yticklabels=False,cmap='Blues')


# we obtain the unique value to valide to drop or not.
Pedtype_data = df_KSI.loc[df_KSI['PEDTYPE'].notnull(), 'PEDTYPE'].unique()
# Data example:
#  "Vehicle turns left while ped crosses with ROW at inter."
#  "Pedestrian hit at mid-block"
#  "Pedestrian involved in a collision with transit vehicle anywhere along roadway"

Pedact_data = df_KSI.loc[df_KSI['PEDACT'].notnull(), 'PEDACT'].unique()
# Data example:
#  "Crossing with right of way
#  "Crossing, no Traffic Control
#  "Running onto Roadway
#  "Crossing without right of way

Pedcon_data = df_KSI.loc[df_KSI['PEDCOND'].notnull(), 'PEDCOND'].unique()
# Data example:
#  "Had Been Drinking
#  "Medical or Physical Disability
#  "Ability Impaired, Alcohol
#  "Ability Impaired, Alcohol Over .80
   
CyclisTipe_data = df_KSI.loc[df_KSI['CYCLISTYPE'].notnull(), 'CYCLISTYPE'].unique()
# Data example:
#  "Crossing with right of way
#  "Crossing, no Traffic Control
#  "Running onto Roadway
#  "Crossing without right of way

CycACT_data = df_KSI.loc[df_KSI['CYCACT'].notnull(), 'CYCACT'].unique()
# Data example:
#  "Driving Properly
#  "Other
#  "Improper Passing

Cyccond_data = df_KSI.loc[df_KSI['CYCCOND'].notnull(), 'CYCCOND'].unique()
# Data example
#  "Had Been Drinking
#  "Ability Impaired, Drugs
#  "Ability Impaired, Alcohol Over .80

Offset_data = df_KSI.loc[df_KSI['OFFSET'].notnull(), 'OFFSET'].unique()
# Data example:
#  "60 NORTH OF
#  "6 m West of
#  "4 m North of
#  "10 m East of

x_data = df_KSI.loc[df_KSI['X'].notnull(), 'X'].unique()
# Data example:
#  "-8.84461e+06
#  "-8.81648e+06
#  "-8.82973e+06

y_data = df_KSI.loc[df_KSI['Y'].notnull(), 'Y'].unique()
# Data example:
#   "5412413.88830144
#   "5412413.88830144
#	"5434843.38873295

index_data = df_KSI.loc[df_KSI['INDEX_'].notnull(), 'INDEX_'].unique()
# Data example
#  "3387730
#  "3387731
#  "3388101

# Data cleaning by dropping columns with large amount of missing value from heat map

df_KSI_Dropped = df_KSI.drop(["PEDTYPE", "PEDACT", "PEDCOND", "CYCLISTYPE", "CYCACT", "CYCCOND", "OFFSET", "X", "Y", "INDEX_"], axis=1)


# Missing values in Categorial columns

# All this column have are categorical and Have "YES" or ' '.
# Replace YES by 1 and ' ' by 0
categorical_columns_with_Yes = ["CYCLIST","PEDESTRIAN","AUTOMOBILE","MOTORCYCLE","TRUCK","TRSN_CITY_VEH","EMERG_VEH","PASSENGER","SPEEDING","AG_DRIV","REDLIGHT","ALCOHOL","DISABILITY"]

for column in categorical_columns_with_Yes:
    df_KSI_Dropped[column] = df_KSI_Dropped[column].replace({'Yes': 1, np.nan: 0})

print(df_KSI_Dropped.isna().sum()) # 

# Missing value in FATAL_NO



# Analysis other columns
ACCLASS_data = df_KSI_Dropped.loc[df_KSI_Dropped['ACCLASS'].notnull(), 'ACCLASS'].unique()
# Fatal
# Non-Fatal Injury  >>  No Fatal
# Property Damage Only >> No Fatal
# Note the are several record that in class appear like Fatal but in FATAL_NO is not counted like Fatal.The same happend in INJURY.
# We are going to Change the Property Damage and non-fatal columns to Non-Fatal¶

df_KSI_Dropped['ACCLASS'] = np.where(df_KSI_Dropped['ACCLASS'] == 'Property Damage Only', 'No-Fatal', df_KSI_Dropped['ACCLASS'])
df_KSI_Dropped['ACCLASS'] = np.where(df_KSI_Dropped['ACCLASS'] == 'Non-Fatal Injury', 'No-Fatal', df_KSI_Dropped['ACCLASS'])
df_KSI_Dropped.ACCLASS.unique()


ROAD_CLASS_data = df_KSI_Dropped.loc[df_KSI_Dropped['ROAD_CLASS'].notnull(), 'ROAD_CLASS'].unique()
# Major Arterial,Local, Minor Arterial, Collector, Other, Pending
# Laneway, Expressway, Expressway Ramp, Major Arterial Ramp

ACCLOC_data = df_KSI_Dropped.loc[df_KSI_Dropped['ACCLOC'].notnull(), 'ACCLOC'].unique()

# At Intersection,Intersection Related, Non Intersection, At/Near Private Drive
# Underpass or Tunnel, Private Driveway, Overpass or Bridge,Trail,Laneway

IMPACTYPE_data = df_KSI_Dropped.loc[df_KSI_Dropped['IMPACTYPE'].notnull(), 'IMPACTYPE'].unique()
# Pedestrian Collisions, Turning Movement, Approaching, Other
# Cyclist Collisions, Angle, SMV Other, Rear End, Sideswipe
# SMV Unattended Vehicle

INJURY_data = df_KSI_Dropped.loc[df_KSI_Dropped['INJURY'].notnull(), 'INJURY'].unique()
# None, Fatal, Major, Minor, Minimal

FATAL_NO_data = df_KSI_Dropped.loc[df_KSI_Dropped['FATAL_NO'].notnull(), 'FATAL_NO'].unique()
# Number and nan
# Replace Nan ( 17367 reg) by 0


df_KSI_Dropped["FATAL_NO"] = df_KSI_Dropped["FATAL_NO"].fillna(0)


DRIVCOND_data = df_KSI_Dropped.loc[df_KSI_Dropped['DRIVCOND'].notnull(), 'DRIVCOND'].unique()
# Unknown, Ability Impaired, Alcohol, Normal, Ability Impaired, Alcohol Over .08
# Inattentive, Had Been Drinking, Medical or Physical Disability
# Fatigue, Other, Ability Impaired, Drugs

# Data cleaning by changing data type from objetive to categorical columns.

print(df_KSI_Dropped.select_dtypes(["object"]).columns)

# Change objetive to categorical.
# The benefit of changing columns from object type to category in this case is 
# to optimize memory usage and potentially improve performance in certain operations.
objdtype_cols = df_KSI_Dropped.select_dtypes(["object"]).columns
df_KSI_Dropped[objdtype_cols] = df_KSI_Dropped[objdtype_cols].astype('category')

print(df_KSI_Dropped.info())

# Data relationship 
#
############################################################################
# Number of Unique accidents by Year

YEAR_data = df_KSI_Dropped.loc[df_KSI_Dropped['YEAR'].notnull(), 'YEAR'].unique()

# In this graph we need to fill in the null data from ACCNUM, and we need to apply 
# a logic to avoid duplicating the number because 1 accident can have more than 1 record 
# since a record is created for each person involved in an accident.

# And we need to complete the NaN values of column ACCNUM following the same logic. 
# We checked that ACCNUM share the same values by accident: year, date, time, street 1, street 2, latitude and longitude and share the same X value.    
# So, we are going to use column 'LATITUD' and 'LONGITUD' to filled value left.

df_KSI_Dropped['SUM_copy'] = (df_KSI_Dropped['LATITUDE'].astype(float) + df_KSI_Dropped['LONGITUDE'].astype(float)).astype(str)

# Assign the copied values to 'ACCNUM' where 'ACCNUM' is null
df_KSI_Dropped['ACCNUM'] = np.where(df_KSI_Dropped['ACCNUM'].isnull(), df_KSI_Dropped['SUM_copy'], df_KSI_Dropped['ACCNUM'])

# Remove the temporary 'SUM_copy' column
df_KSI_Dropped.drop(columns=['SUM_copy'], inplace=True)

print(df_KSI_Dropped.isnull().sum())


Num_accident = df_KSI_Dropped.groupby('YEAR')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("Total Accidents by years")
plt.ylabel('Number of Accidents (ACCNUM)')

datx = plt.gca()
datx.tick_params(axis='x', colors='blue')
datx.tick_params(axis='y', colors='red')
my_colors = list('rgbkymc')   #red, green, blue, black, etc.
Num_accident.plot(
    kind='bar', 
    color='blue',
    edgecolor='black'
)
plt.show()

# answer : The number of accident have been decrease in the las 5 years. 


############################################################################
# Number of Unique accidents by Month

df_KSI_Dropped['DATE'] = pd.to_datetime(df_KSI_Dropped['DATE'])

# Extract year and month
df_KSI_Dropped['YEAR'] = df_KSI_Dropped['DATE'].dt.year
df_KSI_Dropped['MONTH'] = df_KSI_Dropped['DATE'].dt.month

# Group byymonth and count unique 'ACCNUM'
num_accidents_per_month = df_KSI_Dropped.groupby(['MONTH'])['ACCNUM'].nunique()

plt.figure(figsize=(12,6))
plt.title("Accidents  by  Months from 2006 to 2022")
plt.ylabel('Number of Accidents (ACCNUM)')

datx = plt.gca()
datx.tick_params(axis='x', colors='blue')
datx.tick_params(axis='y', colors='red')
my_colors = list('rgbkymc')   #red, green, blue, black, etc.
num_accidents_per_month.plot(
    kind='bar', 
    color='blue',
    edgecolor='black'
)
plt.show()

# answer : Accidents happened more from June to October

############################################################################
# Fatality happended
#
# Creating a Heatmap where Fatality happened


import folium
from folium.plugins import HeatMap


df_Fatal = df_KSI_Dropped[df_KSI_Dropped['INJURY'] == 'Fatal']
df_Fatal = df_Fatal[['LATITUDE', 'LONGITUDE', 'FATAL_NO']]
latitud_Toronto = df_Fatal.describe().at['mean','LATITUDE']
longitud_Toronto = df_Fatal.describe().at['mean','LONGITUDE']
Toronto_location = [latitud_Toronto, longitud_Toronto]


# Create the map
Fatal_map = folium.Map(location=[latitud_Toronto, longitud_Toronto], zoom_start=10.255)

# Description
description = "<h3>This is a HeatMap of Fatal Accidents in Toronto</h3>"
description_popup = folium.Popup(description, max_width=300)
description_marker = folium.Marker(location=[latitud_Toronto, longitud_Toronto], popup=description_popup)
Fatal_map.add_child(description_marker)

title_html = '''
             <h2 align="center" style="font-size:20px"><b>Fatal Accidents HeatMap</b></h2>
             '''
Fatal_map.get_root().html.add_child(folium.Element(title_html))

# Layer
HeatMap(df_Fatal.values, min_opacity=0.3).add_to(Fatal_map)

# Display the map
Fatal_map

top_5_districts = df_KSI_Dropped['DISTRICT'].value_counts().nlargest(5)

############################################################################
# Categorizing Fatal vs. No Fatal Incident 

plt.figure(figsize=(12, 6)) 

# Plot the catplot
ax = sns.catplot(x='YEAR', kind='count', data=df_KSI_Dropped, hue='ACCLASS')
ax.set_xticklabels(rotation=45, ha='right')  
plt.title('Count of Fatal vs No-Fatal Accidents')
plt.show()

###########################################################################
#Fatality over years (# of people died)

Fatality = df_KSI_Dropped[df_KSI_Dropped['ACCLASS'] =='Fatal']
Fatality = Fatality.groupby(df_KSI_Dropped['YEAR']).count()
plt.figure(figsize=(12,6))


plt.ylabel('Number of Fatality')
plt.title('Number of Fatality by year of ACCLASS')
Fatality['ACCLASS'].plot(kind='bar',color="blue" , edgecolor='black')

plt.show()
                 

############################################################################
# Categorizing Fatal  Incident 

INJURY_counts = df_KSI_Dropped['INJURY'].value_counts()
print(INJURY_counts)

ACCLASS_counts = df_KSI_Dropped['ACCLASS'].value_counts()
print(ACCLASS_counts)

FATAL_NO__fatal_counts = len(df_KSI_Dropped[df_KSI_Dropped['FATAL_NO'] > 0])
FATAL_NO__non_fatal_counts = len(df_KSI_Dropped[df_KSI_Dropped['FATAL_NO'] == 0])

print("Fatal Counts:", FATAL_NO__fatal_counts)
print("Non-Fatal Counts:", FATAL_NO__non_fatal_counts)

# Are not coincident the Fatal number between INJURY, ACCLASS , and FATAL_NO
# INJURY column
# None       6956
# Major      6151
# Minor      1422
# Minimal    1123
# Fatal       931

# ACCLASS column
# No-Fatal    15616
# Fatal        2573

# FATAL_NO
#  FATAL_NO__fatal_counts = 17367
#  FATAL_NO__fatal_counts =827

#############################################################################3
# Fatality over years (# of people died)

Fatality = df_KSI_Dropped[df_KSI_Dropped['INJURY'] =='Fatal']
Fatality = Fatality.groupby(df_KSI_Dropped['YEAR']).count()
plt.figure(figsize=(12,6))


plt.ylabel('Number of Injury = Fatal')
plt.title('Count of Injury = Fatal by Year')
Fatality['INJURY'].plot(kind='bar',color="blue" , edgecolor='black')

plt.show()

#############################################################################3
# DISTRICT 

#where accident happens

Region_df_KSI_Dropped = df_KSI_Dropped['DISTRICT'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
plt.title('Accident by DISTRICT')
Region_df_KSI_Dropped.plot(kind='bar',color=list('rgbkmc') )
plt.show()

#############################################################################3
# NEIGHBOURHOOD_140

NEIGHBOURHOOD = df_KSI_Dropped['NEIGHBOURHOOD_140'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
plt.title('10 Neighbourhood with more Accident')
NEIGHBOURHOOD.nlargest(10).plot(kind='bar',color=list('rgbkmc') )
plt.show()

##############################################################################
# ACCIDENT BY WEEKDAY

# Convert the DATE column to datetime format
df_KSI_Dropped['DATE'] = pd.to_datetime(df_KSI_Dropped['DATE'])

# Count the number of accidents by weekday
Weekday_ = df_KSI_Dropped['DATE'].dt.weekday.value_counts()

# Create the plot
plt.figure(figsize=(12, 6))
plt.ylabel('Number of Accidents')
Weekday_.plot(kind='bar', color=list('rgbkmc'))
plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xlabel('Weekday')
plt.title('Number of Accidents by Weekday')
plt.show()

##################################################################################
# DRIVING CONDITION VS ACCIDENT

## creating a pivot table for accidents causing by 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL'
df_KSI_pivot_cause = df_KSI_Dropped.pivot_table(index='YEAR', 
                           values = ['SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL'],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'Total Under Category')
fig, ax1 = plt.subplots(figsize=(12,6))
df_KSI_pivot_cause.iloc[16].plot(kind='pie', ax=ax1, autopct='%3.1f%%',fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('Driving condition VS Accidents in Ontario in last 15 years(%age)',fontsize=10)
plt.title('Driving condition VS Accidents')


###################################################################################
# Accidents causing by 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL'  in the las 15 years

from matplotlib.ticker import MaxNLocator

df_KSI_pivot_cause = df_KSI_Dropped.pivot_table(index='YEAR',
                                                values=['SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL'],
                                                aggfunc=np.sum,
                                                margins = True)
# Drop the 'All' row from the pivot table
df_KSI_pivot_cause.drop('All', axis=0, inplace=True)

# Plot the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
df_KSI_pivot_cause.plot(kind='bar', ax=ax)
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Accidents by Cause in Ontario 2006 to 2022')
plt.legend(title='Cause')
plt.xticks(rotation=0)


plt.show()

##################################################################################

## Kind of Vechile VS Accident #
## creating a pivot table for accidents causing by 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH'   in 15 years
df_KSI_pivot_Types = df_KSI_Dropped.pivot_table(index='YEAR', 
                           values = [ 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH' ],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'Total Under Category')

fig, ax1 = plt.subplots(figsize=(8,8))
df_KSI_pivot_Types.iloc[16].plot(kind='pie', ax=ax1, autopct='%3.1f%%',fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('% Accidents by Vehicle Type',fontsize=12)
plt.title('Accidents by Vehicle type in Ontario 2006 to 2022')

##################################################################################
#Victims by 'CYCLIST','PEDESTRIAN','PASSENGER' 

df_KSI_pivot_Victims = df_KSI_Dropped.pivot_table(index='YEAR', 
                           values = [ 'CYCLIST','PEDESTRIAN','PASSENGER' ],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'Total Under Category')
fig, ax1 = plt.subplots(figsize=(8,8))
df_KSI_pivot_Victims.iloc[16].plot(kind='pie', ax=ax1, autopct='%3.1f%%',fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('% Victims in Accidents by Type',fontsize=12)
plt.title('Victims Type VS Accidents in Ontario 2006 to 2022')

#################################################################################
# Fatal and Disability VS accident #

## creating a pivot table for 'FATAL','DISABILITY' against accidents #
df_KSI_pivot_DF = df_KSI_Dropped.pivot_table(index='YEAR',
                                             values=['FATAL_NO', 'DISABILITY'],
                                             aggfunc=lambda x: np.sum(x > 0),
                                             margins=True,
                                             margins_name='Total Under Category')
fig, ax1 = plt.subplots(figsize=(8, 8))
df_KSI_pivot_DF.rename(columns={'FATAL_NO': 'FATAL'}, inplace=True)
df_KSI_pivot_DF.iloc[16].plot(kind='pie', ax=ax1, autopct='%3.1f%%', fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('% Total Accidents with FATALITY OR DISABILITY', fontsize=12)
plt.title('Accidents with Fatalities and Disability in Ontario 2006 to 2022')
plt.show()



#######################################3
## creating a pivot table for 'FATAL','DISABILITY' against accidents years
df_KSI_pivot_DF = df_KSI_Dropped.pivot_table(index='YEAR',
                                             values=['FATAL_NO', 'DISABILITY'],
                                             aggfunc=lambda x: np.sum(x > 0),
                                             margins=True,
                                             margins_name='Total Under Category')

# Remove the 'Total Under Category' row
df_KSI_pivot_DF.drop('Total Under Category', inplace=True)

# Create a bar graph for each year
fig, ax = plt.subplots(figsize=(10, 6))
df_KSI_pivot_DF.rename(columns={'FATAL_NO': 'FATAL'}, inplace=True)
df_KSI_pivot_DF.plot(kind='bar', ax=ax)
ax.set_ylabel('Count')
ax.set_xlabel('Year')
ax.set_title('Total Accidents with FATAL and/or DISABILITY Resulted in Ontario 2006 to 2022')

plt.tight_layout()
plt.show()

##################################################################
# Fatal Accident  by MONTH

# Convert the DATE column to datetime format
df_KSI_Dropped['DATE'] = pd.to_datetime(df_KSI_Dropped['DATE'])

# Count the number of accidents by month
months = df_KSI_Dropped['DATE'].dt.month

df_KSI_pivot_FN = df_KSI_Dropped.pivot_table(index=months,
                                             values=['FATAL_NO',],
                                             aggfunc=lambda x: np.sum(x > 0),
                                             margins=True,
                                             margins_name='Total Under Category')

# Remove the 'Total Under Category' row
df_KSI_pivot_FN.drop('Total Under Category', inplace=True)

# Create a bar graph for each year
fig, ax = plt.subplots(figsize=(10, 6))
df_KSI_pivot_FN.rename(columns={'FATAL_NO': 'FATAL'}, inplace=True)
df_KSI_pivot_FN.plot(kind='bar', ax=ax)
ax.set_ylabel('Count')
ax.set_xlabel('month')
ax.set_title('Total FATAL Accidents in Ontario 2006 to 2022')

plt.tight_layout()
plt.show()

##################################################################


# Extract month and year from the DATE column
df_KSI_Dropped['DATE'] = pd.to_datetime(df_KSI_Dropped['DATE'])
df_KSI_Dropped['MONTH'] = df_KSI_Dropped['DATE'].dt.month
df_KSI_Dropped['YEAR'] = df_KSI_Dropped['DATE'].dt.year

# Create a new column 'FATAL_COUNT' where FATAL_NO > 0 is counted as 1
df_KSI_Dropped['FATAL_COUNT'] = df_KSI_Dropped['FATAL_NO'].apply(lambda x: 1 if x > 0 else 0)

# Create a pivot table to sum the 'FATAL_COUNT' by month and year
df_pivot = df_KSI_Dropped.pivot_table(
    index='MONTH',
    columns='YEAR',
    values='FATAL_COUNT',
    aggfunc='sum',
    fill_value=0
)

# Create the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_pivot, center=df_pivot.loc[1, 2007], annot=True, fmt="d", cmap="YlGnBu")
plt.show()

############################################################################################

#Data Modeling
#Data preparation for modeling by checking null value, get dummies

print(df_KSI_Dropped.shape)
print(df_KSI_Dropped.columns)
print(df_KSI_Dropped.dtypes)

# WE ARE GOING TO USER FEATURE SELECCTION OF REGRESION TREE TO DECIDE IMPORTANT COLUMNS 
# TO PREDICT FATAL ACCIDENT.

df_KSI_Dropped.isnull().sum()


from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel

# Select the columns for feature selection
num_columns = ['YEAR', 'TIME', 'WARDNUM', 'LATITUDE', 'LONGITUDE', 
               'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
               'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 
               'MONTH', 'FATAL_COUNT']

cat_columns = ['ACCNUM', 'STREET1', 'STREET2', 'ROAD_CLASS', 'DISTRICT', 'LOCCOORD', 'ACCLOC', 'TRAFFCTL',
                  'VISIBILITY', 'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY', 'INITDIR',
                  'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140',
                  'NEIGHBOURHOOD_140', 'DIVISION']

# Copy the selected columns from the original dataframe
X = df_KSI_Dropped[num_columns + cat_columns].copy()

# Assuming df_KSI_Dropped is your dataframe
y = df_KSI_Dropped['FATAL_COUNT']

# Fill missing values in numerical columns with mean
num_imputer = SimpleImputer(strategy='mean')
X[num_columns] = num_imputer.fit_transform(X[num_columns])

# Fill missing values in categorical columns with 'NA'
cat_imputer = SimpleImputer(strategy='constant', fill_value='NA')
X[cat_columns] = cat_imputer.fit_transform(X[cat_columns])

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=cat_columns)

# Perform feature selection using regression tree
regression_tree = DecisionTreeRegressor(random_state=42)
selector = SelectFromModel(regression_tree)
selector.fit(X, y)

# Get the selected feature names
selected_feature_names = X.columns[selector.get_support()]

# Print the selected feature names
print("Selected features:")
for feature in selected_feature_names:
    print(feature)
#####
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Define the columns for feature selection
float_columns = ['WARDNUM', 'LATITUDE', 'LONGITUDE', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
                 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
int_columns = ['YEAR', 'TIME', 'MONTH']
object_columns = ['ACCNUM', 'STREET1', 'STREET2', 'ROAD_CLASS', 'DISTRICT', 'LOCCOORD', 'ACCLOC', 'TRAFFCTL',
                  'VISIBILITY', 'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY', 'INITDIR',
                  'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140',
                  'NEIGHBOURHOOD_140', 'DIVISION']

X = df_KSI_Dropped[float_columns + int_columns + object_columns].copy()
y = df_KSI_Dropped['FATAL_COUNT']

# Define the preprocessing for float and int columns (as before)
float_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

int_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])


def convert_to_string(df):
    return df.astype(str)

string_converter = FunctionTransformer(convert_to_string)

# Update categorical transformer pipeline
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('string_converter', string_converter),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num_float', float_transformer, float_columns),
        ('num_int', int_transformer, int_columns),
        ('cat', cat_transformer, object_columns)
    ])



# Define the model
model = DecisionTreeRegressor(random_state=42)

# Make a pipeline that combines the preprocessor and the model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('feature_selection', SelectFromModel(model)),
                      ('classifier', model)])

# Apply cross-validation
scores = cross_val_score(clf, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean()}")

from sklearn.ensemble import RandomForestRegressor

# Define your original pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeRegressor(random_state=42))])

# Fit your original pipeline and perform cross-validation
clf.fit(X, y)
scores = cross_val_score(clf, X, y, cv=5)

# Print cross-validation scores
print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean()}")

# Define a new pipeline for feature importance calculation
clf_rf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestRegressor(random_state=42))])

# Fit the new pipeline
clf_rf.fit(X, y)

# Get the feature importances
importances = clf_rf.named_steps['classifier'].feature_importances_

# Get feature names after transformation, if any
try:
    feature_names = clf_rf.named_steps['preprocessor'].transformers_[2][1]\
                       .named_steps['onehot'].get_feature_names(input_features=object_columns)
except AttributeError:
    feature_names = object_columns

feature_names = np.concatenate([float_columns, int_columns, feature_names])

# Print feature importances
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance}")



##################

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor



# Define the columns for feature selection
float_columns = ['WARDNUM', 'LATITUDE', 'LONGITUDE', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
                 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
int_columns = ['YEAR', 'TIME', 'MONTH']
object_columns = ['LOCCOORD', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'ACCLASS', 
                  'IMPACTYPE','INVTYPE']

X = df_KSI_Dropped[float_columns + int_columns + object_columns].copy()
y = df_KSI_Dropped['FATAL_COUNT']

# Define the preprocessing for float and int columns
float_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

int_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

def convert_to_string(df):
    return df.astype(str)

string_converter = FunctionTransformer(convert_to_string)

# Update categorical transformer pipeline
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('string_converter', string_converter),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num_float', float_transformer, float_columns),
        ('num_int', int_transformer, int_columns),
        ('cat', cat_transformer, object_columns)
    ])

# Define a new pipeline for feature importance calculation
clf_rf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestRegressor(random_state=42))])

# Fit the new pipeline
clf_rf.fit(X, y)

# Get the feature importances
importances = clf_rf.named_steps['classifier'].feature_importances_

# Get feature names after transformation, if any
try:
    feature_names = clf_rf.named_steps['preprocessor'].transformers_[2][1]\
                       .named_steps['onehot'].get_feature_names(input_features=object_columns)
except AttributeError:
    feature_names = object_columns

feature_names = np.concatenate([float_columns, int_columns, feature_names])

# Print feature importances
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance}")


# COLUMN ELIMINATED BY SCORE 0 THAT suggests that a feature doesn't contribute anything to the model's ability to PREDICT

# ACCNUM, Ward_Name, Hood_Name,District, street1, street2, offset,ROAD_CLASS, ACCLOC,  MANOEUVER, PEDTYPE, PEDACT, PEDCOND, CYCLISTYPE, CYCACT, DIVISION AND OBJECTID
# HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140'
# VEHTYPE', 'DRIVACT', 'DRIVCOND'
#'INVAGE', 'INJURY', 'INITDIR'
# 'INVTYPE'















