#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 03:49:21 2023

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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


##############################################################################
# 1. Data exploration: a complete review and analysis of the dataset including:

# 1.1. Load and describe data elements (columns), provide descriptions & types,
# ranges and values of elements as appropriate. – use pandas, numpy and any
# other python packages.

# Pandas setting
pd.set_option('display.max_columns', None)

# Download the  "KSI.csv" from here.


# Load Data
df_KSI = pd.read_csv(Path("D:/pCloudFolder/Repositories/SuperLearningProject/KSI.csv"))

# Carry out initial analysis per column and show the types, number of missing
# values
print(df_KSI.info())
print(df_KSI.columns.values)
print(df_KSI.head(3))
print(df_KSI.shape)
print(df_KSI.dtypes)

# 1.2. Statistical assessments including means, averages, correlations
print(df_KSI.describe())

# 1.3. Missing data evaluations – use pandas, numpy and any other python
# packages
print(df_KSI.isnull().sum())
print(df_KSI.isna().sum())

# Missing values for other columns replace ' ' with nan.
#for No categorical number column
df_KSI_nan = df_KSI.replace(' ', np.nan,  regex=False)

# printing percentage of missing values for each feature
print(df_KSI_nan.isna().sum()/len(df_KSI_nan)*100)

fig, ax = plt.subplots(figsize=(15,7))
# heatmap to visualize features with most missing values
sns.heatmap(df_KSI_nan.isnull(), yticklabels=False,cmap='Blues')

##############################################################################
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
   
CyclisTipe_data = df_KSI.loc[df_KSI['CYCLISTYPE'].notnull(),
                             'CYCLISTYPE'].unique()
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
# Data example:(500 UNIQUE)
#  "60 NORTH OF
#  "6 m West of
#  "4 m North of
#  "10 m East of

x_data = df_KSI.loc[df_KSI['X'].notnull(), 'X'].unique()
# Data example:(18 MIL UNIQUE)
#  "-8.84461e+06
#  "-8.81648e+06
#  "-8.82973e+06

y_data = df_KSI.loc[df_KSI['Y'].notnull(), 'Y'].unique()
# Data example:(18 MIL UNIQUE)
#   "5412413.88830144
#   "5412413.88830144
#	"5434843.38873295

index_data = df_KSI.loc[df_KSI['INDEX_'].notnull(), 'INDEX_'].unique()
# Data example (18 MIL UNIQUE)
#  "3387730
#  "3387731
#  "3388101

ObjectId_data = df_KSI.loc[df_KSI['ObjectId'].notnull(), 'ObjectId'].unique()
# Data example:
#  incremental ID : 1,2,3......n
#  
STREET1_data = df_KSI.loc[df_KSI['STREET1'].notnull(), 'STREET1'].unique()
# HAVE 1856 UNIQUE DATA

STREET2_data = df_KSI.loc[df_KSI['STREET2'].notnull(), 'STREET2'].unique()
# HAVE 2704 UNIQUE DATA

WARDNUM_data = df_KSI.loc[df_KSI['WARDNUM'].notnull(), 'WARDNUM'].unique()
# HAVE 2704 UNIQUE DATA

ACCLOC_data = df_KSI.loc[df_KSI['ACCLOC'].notnull(), 'ACCLOC'].unique()
# EXAMPLES 
# At Intersection
# Intersection Related
# Non Intersection
# At/Near Private Drive
# Underpass or Tunnel

# Data cleaning by dropping columns with large amount of missing value from
# heat map
  # df_KSI_Dropped = df_KSI.drop(["PEDTYPE", "PEDACT", "PEDCOND", "CYCLISTYPE",
  #                               "CYCACT", "CYCCOND", "OFFSET", "X", "Y",
  #                               "INDEX_", "ObjectId", "STREET1","STREET2",
  #                               "WARDNUM"], axis=1)

df_KSI_Dropped = df_KSI.drop(["PEDTYPE", "PEDACT", "PEDCOND", "CYCLISTYPE",
                              "CYCACT", "CYCCOND", "OFFSET", "X", "Y",
                              "INDEX_", "ObjectId", "STREET1","STREET2",
                              "WARDNUM", 'FATAL_NO'], axis=1)
#-------------------------------------
# Missing values in Categorial columns
# All this column have are categorical and Have "YES" or ' '.
# Replace YES by 1 and ' ' by 0
categorical_columns_with_Yes = ["CYCLIST","PEDESTRIAN","AUTOMOBILE",
                                "MOTORCYCLE", "TRUCK","TRSN_CITY_VEH",
                                "EMERG_VEH", "PASSENGER","SPEEDING","AG_DRIV",
                                "REDLIGHT","ALCOHOL","DISABILITY"]

for column in categorical_columns_with_Yes:
    df_KSI_Dropped[column] = df_KSI_Dropped[column].replace({'Yes': 1,
                                                             np.nan: 0})

print(df_KSI_Dropped['ACCLASS'].isnull().sum())

# Analysis other columns
ACCLASS_data = df_KSI_Dropped.loc[df_KSI_Dropped['ACCLASS'].notnull(),
                                  'ACCLASS'].unique()
# Fatal
# Non-Fatal Injury  >>  No Fatal
# Property Damage Only >> No Fatal
# Note the are several record that in class appear like Fatal but in FATAL_NO
# is not counted like Fatal.The same happend in INJURY.
# We are going to Change the Property Damage and non-fatal columns to Non-Fatal

# 5 null of 18000 record --> drop is strategy.
print(df_KSI_Dropped['ACCLASS'].isnull()) 


df_KSI_Dropped['ACCLASS'] = np.where(df_KSI_Dropped['ACCLASS'] ==
                                     'Property Damage Only', 'No-Fatal',
                                     df_KSI_Dropped['ACCLASS'])

df_KSI_Dropped['ACCLASS'] = np.where(df_KSI_Dropped['ACCLASS'] ==
                                     'Non-Fatal Injury', 'No-Fatal',
                                     df_KSI_Dropped['ACCLASS'])

df_KSI_Dropped.ACCLASS.unique()

df_KSI_Dropped['ACCLASS_TARG'] = df_KSI_Dropped['ACCLASS'].replace({'Fatal': 1,
                                                                    'No-Fatal': 0})

ROAD_CLASS_data = df_KSI_Dropped.loc[df_KSI_Dropped['ROAD_CLASS'].notnull(),
                                     'ROAD_CLASS'].unique()
# Major Arterial,Local, Minor Arterial, Collector, Other, Pending
# Laneway, Expressway, Expressway Ramp, Major Arterial Ramp

ACCLOC_data = df_KSI_Dropped.loc[df_KSI_Dropped['ACCLOC'].notnull(),
                                 'ACCLOC'].unique()
# At Intersection,Intersection Related, Non Intersection, At/Near Private Drive
# Underpass or Tunnel, Private Driveway, Overpass or Bridge,Trail,Laneway

IMPACTYPE_data = df_KSI_Dropped.loc[df_KSI_Dropped['IMPACTYPE'].notnull(),
                                    'IMPACTYPE'].unique()
# Pedestrian Collisions, Turning Movement, Approaching, Other
# Cyclist Collisions, Angle, SMV Other, Rear End, Sideswipe
# SMV Unattended Vehicle

INJURY_data = df_KSI_Dropped.loc[df_KSI_Dropped['INJURY'].notnull(),
                                 'INJURY'].unique()
# None, Fatal, Major, Minor, Minimal

  # FATAL_NO_data = df_KSI_Dropped.loc[df_KSI_Dropped['FATAL_NO'].notnull(),
  #                                    'FATAL_NO'].unique()
  # # Number and nan
  # # Replace Nan ( 17367 reg) by 0
  # df_KSI_Dropped["FATAL_NO"] = df_KSI_Dropped["FATAL_NO"].fillna(0)


# Data cleaning by changing data type from objetive to categorical columns.
print(df_KSI_Dropped.select_dtypes(["object"]).columns)

# Change objetive to categorical.
# The benefit of changing columns from object type to category in this case is 
# to optimize memory usage and potentially improve performance in certain
# operations.
objdtype_cols = df_KSI_Dropped.select_dtypes(["object"]).columns
df_KSI_Dropped[objdtype_cols] = df_KSI_Dropped[objdtype_cols].astype('category')

print(df_KSI_Dropped.info())

############################################################################
# 1.4. Graphs and visualizations – use pandas, matplotlib, seaborn, numpy and any
# other python packages, you also can use power BI desktop.
# Number of Unique accidents by Year

YEAR_data = df_KSI_Dropped.loc[df_KSI_Dropped['YEAR'].notnull(),
                               'YEAR'].unique()

# In this graph we need to fill in the null data from ACCNUM, and we need to apply 
# a logic to avoid duplicating the number because 1 accident can have more than 1 record 
# since a record is created for each person involved in an accident.

# And we need to complete the NaN values of column ACCNUM following the same logic. 
# We checked that ACCNUM share the same values by accident: year, date, time,
#street 1, street 2, latitude and longitude and share the same X value.    
# So, we are going to use column 'LATITUD' and 'LONGITUD' to filled value left.

dateList = []
dates = pd.to_datetime(df_KSI_Dropped['DATE'])
for i, value in enumerate(dates):
  dateList.append(int(dates[i].value/1000000000))

df_KSI_Dropped['SUM_copy'] = (df_KSI_Dropped['LATITUDE'].astype(float) +
                              df_KSI_Dropped['LONGITUDE'].astype(float) +
                              dateList).astype(str)

  # df_KSI_Dropped['SUM_copy'] = (df_KSI_Dropped['LATITUDE'].astype(float) +
  #                               df_KSI_Dropped['LONGITUDE'].astype(float)).astype(str)

# c = {}
# for i, test in enumerate(df_KSI_Dropped['SUM_copy']):
#   if(test in c.keys()):
#     if(str(df_KSI_Dropped['DATE'][i]) in c[test].keys()):
#       if df_KSI_Dropped['ACCNUM'][i] not in c[test][str(df_KSI_Dropped['DATE'][i])]:
#         c[test][str(df_KSI_Dropped['DATE'][i])].append(df_KSI_Dropped['ACCNUM'][i])
#     else:
#       # print(6)
#       c[test][str(df_KSI_Dropped['DATE'][i])] =  [df_KSI_Dropped['ACCNUM'][i]]
#       # pass
#       # c[test][str(df_KSI_Dropped['DATE'][i])] = [df_KSI_Dropped['ACCNUM'][i]]
      
#   else:
#     # if(df_KSI_Dropped['DATE'][i] not in c[test].keys()):
#       pass
#       c[test] = {str(df_KSI_Dropped['DATE'][i]): [df_KSI_Dropped['ACCNUM'][i]]}


# Assign the copied values to 'ACCNUM' where 'ACCNUM' is null
df_KSI_Dropped['ACCNUM'] = np.where(df_KSI_Dropped['ACCNUM'].isnull(),
                                    df_KSI_Dropped['SUM_copy'],
                                    df_KSI_Dropped['ACCNUM'])

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

df_Fatal = df_KSI_Dropped[df_KSI_Dropped['INJURY'] == 'Fatal']
  # df_Fatal = df_Fatal[['LATITUDE', 'LONGITUDE', 'FATAL_NO']]
df_Fatal = df_Fatal[['LATITUDE', 'LONGITUDE']]
latitud_Toronto = df_Fatal.describe().at['mean','LATITUDE']
longitud_Toronto = df_Fatal.describe().at['mean','LONGITUDE']
Toronto_location = [latitud_Toronto, longitud_Toronto]

# Create the map
Fatal_map = folium.Map(location=[latitud_Toronto, longitud_Toronto],
                       zoom_start=10.255)

# Description
description = "<h3>This is a HeatMap of Fatal Accidents in Toronto</h3>"
description_popup = folium.Popup(description, max_width=300)
description_marker = folium.Marker(location=[latitud_Toronto,
                                             longitud_Toronto],
                                             popup=description_popup)
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
ax = sns.catplot(x='YEAR', kind='count', data=df_KSI_Dropped, hue='ACCLASS_TARG')
ax.set_xticklabels(rotation=45, ha='right')  
plt.title('Count of Fatal vs No-Fatal Accidents')
plt.show()

###########################################################################
#Fatality over years (# of people involved in fatal accidents)

Fatality = df_KSI_Dropped[df_KSI_Dropped['ACCLASS'] =='Fatal']
Fatality = Fatality.groupby(df_KSI_Dropped['YEAR']).count()
plt.figure(figsize=(12,6))


plt.ylabel('Number of Fatality')
plt.title('Number of Fatality by year of ACCLASS')
Fatality['ACCLASS'].plot(kind='bar',color="blue" , edgecolor='black')

plt.show()

##############################################################################
# Fatality over years (# of people died)

Fatality = df_KSI_Dropped[df_KSI_Dropped['INJURY'] =='Fatal'] 
Fatality = Fatality.groupby(df_KSI_Dropped['YEAR']).count()
plt.figure(figsize=(12,6))

plt.ylabel('Number of Injury = Fatal')
plt.title('Count of Injury = Fatal by Year')
Fatality['INJURY'].plot(kind='bar',color="blue" , edgecolor='black')

plt.show()

##############################################################################
# DISTRICT 

#where accident happens

Region_df_KSI_Dropped = df_KSI_Dropped['DISTRICT'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
plt.title('Accident by DISTRICT')
Region_df_KSI_Dropped.plot(kind='bar',color=list('rgbkmc') )
plt.show()

##############################################################################
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
plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                      'Saturday', 'Sunday'])
plt.xlabel('Weekday')
plt.title('Number of Accidents by Weekday')
plt.show()
df_KSI_Dropped['WEEKDAY'] = df_KSI_Dropped['DATE'].dt.weekday

##############################################################################
# DRIVING CONDITION VS ACCIDENT

# creating a pivot table for accidents causing by 'SPEEDING', 'AG_DRIV',
# 'REDLIGHT', 'ALCOHOL'
df_KSI_Dropped['NOCONDITION'] = (df_KSI_Dropped['SPEEDING'].astype(int) |
                                  df_KSI_Dropped['AG_DRIV'].astype(int)  |
                                  df_KSI_Dropped['REDLIGHT'].astype(int) |
                                  df_KSI_Dropped['ALCOHOL'].astype(int))
df_KSI_Dropped['NOCONDITION'] = np.logical_xor(df_KSI_Dropped['NOCONDITION'],
                                                1).astype(int)

df_KSI_pivot_cause = df_KSI_Dropped.pivot_table(index='YEAR',
                                                values=['SPEEDING', 'AG_DRIV',
                                                        'REDLIGHT', 'ALCOHOL'],
                                                aggfunc=np.sum,
                                                margins = True)

df_KSI_pivot_cause = df_KSI_Dropped.pivot_table(index='YEAR', 
                           values = ['SPEEDING', 'AG_DRIV', 'REDLIGHT',
                                     'ALCOHOL', 'NOCONDITION'],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'All')
fig, ax1 = plt.subplots(figsize=(12,6))
df_KSI_pivot_cause.iloc[17].plot(kind='pie', ax=ax1,
                                 autopct='%3.1f%%',fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('Driving condition VS Accidents in Ontario in last 15 years(%age)',
               fontsize=10)
plt.title('Driving condition VS Accidents')

##############################################################################
# Accidents causing by 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL' in the
# last 15 years

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

##############################################################################
# Kind of Vehicle VS Accident #
# creating a pivot table for accidents causing by 'AUTOMOBILE', 'MOTORCYCLE',
# 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH'   in 15 years
df_KSI_pivot_Types = df_KSI_Dropped.pivot_table(index='YEAR', 
                           values = [ 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
                                     'TRSN_CITY_VEH', 'EMERG_VEH' ],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'Total Under Category')

fig, ax1 = plt.subplots(figsize=(8,8))
df_KSI_pivot_Types.iloc[17].plot(kind='pie', ax=ax1, autopct='%3.1f%%',
                                 fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('% Accidents by Vehicle Type',fontsize=12)
plt.title('Accidents by Vehicle type in Ontario 2006 to 2022')

##############################################################################
#Victims by 'CYCLIST','PEDESTRIAN','PASSENGER' 

df_KSI_pivot_Victims = df_KSI_Dropped.pivot_table(index='YEAR', 
                           values = [ 'CYCLIST','PEDESTRIAN','PASSENGER' ],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'Total Under Category')
fig, ax1 = plt.subplots(figsize=(8,8))
df_KSI_pivot_Victims.iloc[17].plot(kind='pie', ax=ax1, autopct='%3.1f%%',
                                   fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('% Victims in Accidents by Type',fontsize=12)
plt.title('Victims Type VS Accidents in Ontario 2006 to 2022')

##############################################################################
# Fatal and Disability VS accident #

## VER1 creating a pivot table for 'FATAL','DISABILITY' against accidents #
df_KSI_pivot_DF = df_KSI_Dropped.pivot_table(index='YEAR',
                                             values=['ACCLASS_TARG',
                                                     'DISABILITY'],
                                             aggfunc=np.sum,
                                             margins=True,
                                             margins_name='Total Under Category')
fig, ax1 = plt.subplots(figsize=(8, 8))
#df_KSI_pivot_DF.rename(columns={'INJURY': 'FATAL'}, inplace=True)
df_KSI_pivot_DF.iloc[17].plot(kind='pie', ax=ax1, autopct='%3.1f%%',
                              fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('% Total Accidents FATALITY and DISABILITY', fontsize=12)
plt.title('Accidents with Fatalities and Disability in Ontario 2006 to 2022')
plt.show()

##############################################################################
## creating a pivot table for 'FATAL','DISABILITY' against accidents years

# Remove the 'Total Under Category' row
df_KSI_pivot_DF.drop('Total Under Category', inplace=True)

# Create a bar graph for each year
fig, ax = plt.subplots(figsize=(10, 6))
df_KSI_pivot_DF.rename(columns={'ACCLASS_TARG': 'FATAL'}, inplace=True)
df_KSI_pivot_DF.plot(kind='bar', ax=ax)
ax.set_ylabel('Count')
ax.set_xlabel('Year')
ax.set_title('Total Accidents with FATAL and/or DISABILITY Resulted in Ontario 2006 to 2022')

plt.tight_layout()
plt.show()

#############################################################################
# Fatal Accident  by MONTH

# Convert the DATE column to datetime format
df_KSI_Dropped['DATE'] = pd.to_datetime(df_KSI_Dropped['DATE'])

# Count the number of accidents by month
months = df_KSI_Dropped['DATE'].dt.month

df_KSI_pivot_FN = df_KSI_Dropped.pivot_table(index=months,
                                             values=['ACCLASS_TARG',],
                                             aggfunc=np.sum,
                                             margins=True,
                                             margins_name='Total Under Category')

# Remove the 'Total Under Category' row
df_KSI_pivot_FN.drop('Total Under Category', inplace=True)

# Create a bar graph for each year
fig, ax = plt.subplots(figsize=(10, 6))
df_KSI_pivot_FN.rename(columns={'ACCLASS_TARG': 'FATAL'}, inplace=True)
df_KSI_pivot_FN.plot(kind='bar', ax=ax)
ax.set_ylabel('Count')
ax.set_xlabel('month')
ax.set_title('Total FATAL Accidents in Ontario 2006 to 2022')

plt.tight_layout()
plt.show()

#############################################################################


# Extract month and year from the DATE column
df_KSI_Dropped['DATE'] = pd.to_datetime(df_KSI_Dropped['DATE'])
df_KSI_Dropped['MONTH'] = df_KSI_Dropped['DATE'].dt.month
df_KSI_Dropped['YEAR'] = df_KSI_Dropped['DATE'].dt.year

# Create a new column 'FATAL_COUNT' where FATAL_NO > 0 is counted as 1
#df_KSI_Dropped['FATAL_COUNT'] = df_KSI_Dropped['FATAL_NO'].apply(lambda x: 1 if x > 0 else 0)

# Create a pivot table to sum the 'FATAL_COUNT' by month and year
df_pivot = df_KSI_Dropped.pivot_table(
    index='MONTH',
    columns='YEAR',
    values='ACCLASS_TARG',
    aggfunc='sum',
    fill_value=0
)

# Create the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_pivot, center=df_pivot.loc[1, 2007], annot=True, fmt="d",
            cmap="YlGnBu")
plt.show()

##############################################################################
# 2. Data modelling:
#Data preparation for modeling by checking null value, get dummies
print(df_KSI_Dropped.shape)
print(df_KSI_Dropped.columns)
print(df_KSI_Dropped.dtypes)

# WE ARE GOING TO USE FEATURE SELECCTION OF REGRESION TREE TO DECIDE IMPORTANT
# COLUMNS TO PREDICT FATAL ACCIDENT.
print(df_KSI_Dropped.isnull().sum())


######################### SECOND DROP ####################
DRIVCOND_data = df_KSI_Dropped.loc[df_KSI_Dropped['DRIVCOND'].notnull(),
                                   'DRIVCOND'].unique()
# Unknown, Ability Impaired, Alcohol, Normal, Ability Impaired, Alcohol Over .08
# Inattentive, Had Been Drinking, Medical or Physical Disability
# Fatigue, Other, Ability Impaired, Drugs


MANOEUVER_data = df_KSI_Dropped.loc[df_KSI_Dropped['MANOEUVER'].notnull(),
                                    'MANOEUVER'].unique()
# example 16 unique --> CANDIDATE DROP
# Turning Left, Turning Right , Going Ahead, Stopped, Overtaking, Reversing, Other
# Slowing or Stopping


NEIGHBOURHOOD_158_data = df_KSI_Dropped.loc[df_KSI_Dropped['NEIGHBOURHOOD_158'].notnull(),
                                            'NEIGHBOURHOOD_158'].unique()
# example 159 unique  --> CANDIDATE DROP
# High Park North
# Malvern East
# Woodbine-Lumsden
# Kennedy Park
# Trinity-Bellwoods
# Princess-Rosethorn
# Wexford/Maryvale
# Kingsview Village-The Westway
# Yonge-Bay Corridor
# Keelesdale-Eglinton West
# Rockcliffe-Smythe

HOOD_158_DATA = df_KSI_Dropped.loc[df_KSI_Dropped['HOOD_158'].notnull(),
                                   'HOOD_158'].unique()
# 159 UNIQUE

NEIGHBOURHOOD_140_data = df_KSI_Dropped.loc[df_KSI_Dropped['NEIGHBOURHOOD_140'].notnull(),
                                            'NEIGHBOURHOOD_140'].unique()
# example 141 UNIQUE   --> CANDIDATE DROP
# High Park North (88)
# Malvern (132)
# Woodbine-Lumsden (60)
# Eglinton East (138)
# Trinity-Bellwoods (81)
# Princess-Rosethorn (10)
# Wexford/Maryvale (119)
# Kingsview Village-The Westway (6)
# Bay Street Corridor (76)
# Keelesdale-Eglinton West (110)

HOOD_140_DATA = df_KSI_Dropped.loc[df_KSI_Dropped['HOOD_140'].notnull(),
                                   'HOOD_140'].unique()
# 141 UNIQUE

ACCNUM_data = df_KSI_Dropped.loc[df_KSI_Dropped['ACCNUM'].notnull(),
                                 'ACCNUM'].unique()
# HAVE 4629 UNIQUE --> CANDIDATE TO DROP

DIVISION_data = df_KSI_Dropped.loc[df_KSI_Dropped['DIVISION'].notnull(),
                                   'DIVISION'].unique()
# HAVE 17 UNIQUE --> CANDIDATE TO DROP


# FATAL_NO_data = df_KSI_Dropped.loc[df_KSI_Dropped['FATAL_NO'].notnull(),
 # 'FATAL_NO'].unique()

# HAVE 4629 UNIQUE --> CANDIDATE TO DROP

INJURY_data = df_KSI_Dropped.loc[df_KSI_Dropped['INJURY'].notnull(),
                                 'INJURY'].unique()
# HAVE 5 UNIQUE --> CANDIDATE TO DROP

DATE_DATA = df_KSI_Dropped.loc[df_KSI_Dropped['DATE'].notnull(),
                               'DATE'].unique()
# 3925 UNIQUE DATA. --> CANDIDATE TO DROP

LATITUDE_DATA = df_KSI_Dropped.loc[df_KSI_Dropped['LATITUDE'].notnull(),
                                   'LATITUDE'].unique()
# 4500 UNIQUE DATA. --> CANDIDATE TO DROP

LONGITUDE_DATA = df_KSI_Dropped.loc[df_KSI_Dropped['LONGITUDE'].notnull(),
                                    'LONGITUDE'].unique()
# 4937 UNIQUE DATA. --> CANDIDATE TO DROP


# POSITITION GEOGRAFIC IS DEFINE BY THE FOLLOWING COLUMNS : LATITUDE, LOGITUDE,
# HOOD, NEIGHBOURHOOD, DIVISION, STREETS
# WE ARE MANTAIN ONLY ONE ("HOOD_158") AND OTHER WE ARE DROPPING. 

df_KSI_Dropped2 = df_KSI_Dropped.drop(["ACCLASS", "ACCNUM",
                                       "NEIGHBOURHOOD_140",
                                       "NEIGHBOURHOOD_158", "HOOD_140",
                                       "INJURY","DIVISION"], axis=1)

print(df_KSI_Dropped2.dtypes)


##############################################################################
# 2.2. Feature selection – use pandas and sci-kit learn.
# (The group needs to justify each feature used and any data columns discarded)
# Define the columns for feature selection
float_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
                 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING',
                 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']

int_columns = ['WEEKDAY', 'TIME', 'MONTH']

object_columns = ['ROAD_CLASS', 'DISTRICT', 'LOCCOORD', 'TRAFFCTL',
                  'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE',
                  'INVAGE', 'INITDIR', 'VEHTYPE', 'MANOEUVER', 'DRIVACT',
                  'DRIVCOND','HOOD_158']


# Convert all the object columns to string type
# for column in object_columns:
#     df_KSI_Dropped2[column] = df_KSI_Dropped2[column].astype(str)


# Concatenate all feature names
features = float_columns + int_columns + object_columns

# Split the data into X and y
X = df_KSI_Dropped2[features]
y = df_KSI_Dropped2['ACCLASS_TARG']

# Removing rows with NaN target values --> 5 row
nan_mask = y.isnull()
X = X[~nan_mask]
y = y[~nan_mask]

# Create transformers
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, float_columns + int_columns),
        ('cat', categorical_transformer, object_columns)
    ])

# Combine preprocessor and model in one pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestRegressor(random_state=42))])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# Fit the model
model.fit(X_train, y_train)

# Get feature importances
importances = model.named_steps['classifier'].feature_importances_

# Get one hot encoder feature names
ohe_feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(object_columns)

# Concat numerical columns names and one-hot encoder column names
feature_names = np.concatenate([float_columns, int_columns, ohe_feature_names])

# Create a DataFrame to store the features and their respective importance
important_features_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the DataFrame in descending order of importance
important_features_df = important_features_df.sort_values(by='Importance',
                                                          ascending=False)

# Plot the top 10 important features
important_features_df.head(15).plot(kind='barh', x='Feature', y='Importance',
                                    legend=False)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()


# for i, value in enumerate(df_KSI_Dropped['ACCLASS_TARG'].isna()):
#   if value == True:
#     print(i)

################################################################

