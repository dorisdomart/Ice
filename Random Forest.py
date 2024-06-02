#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:35:03 2023

@author: doris
"""

#%% Import librairy and function definition

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Random Forest 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import r2_score, confusion_matrix, classification_report


# Thresholds for freeze-up and break-up dates = 0.5
FU_thresh = 'FU50hy'
BU_thresh = 'BU50hy'
thresh = 0.5

# Colors for plots
colorHydroLAKES = 'dodgerblue'
colorGRanD = 'darkorange'
colorFU = 'olivedrab'
colorBU = 'red'

# Paths
general_path = os.path.abspath(os.path.join(__file__, r'../../../'))

df_all = pd.read_csv(os.path.join(general_path, 'Data', 'GEE', 'all_phenology.csv'))

# Performance metrics function
def metrics_calc(y_train, y_test, pred_train, pred_test):
    MAE_train = round(metrics.mean_absolute_error(y_train, pred_train),0)
    MAE_test = round(metrics.mean_absolute_error(y_test, pred_test),0)
    
    r2_train = round(r2_score(y_train.ravel(), pred_train),2)
    r2_test = round(r2_score(y_test.ravel(), pred_test),2)

    return MAE_train, MAE_test, r2_train, r2_test



#%% Random Forest arrays

"""
GRanD dataset : dfG
HydroLAKES dataset : dfH
Combined (lakes and reservoirs) dataset : df_all

Classifier :
Input variables : 'latitude', 'longitude', 'elevation', 'area', 'meandepth', 'capacity', 'max_icefraction', 'AFDDmax', 'delay_dd', 'delay_days'
Output variables : 'ice' 0 (max_ice_fraction < 0.5) or 1 (max_ice_fraction > 0.5)

Regressor : 
Input variables : 'latitude', 'longitude', 'elevation', 'area', 'meandepth', 'capacity', 'max_icefraction', 'AFDDmax', 'delay_dd', 'delay_days'
Output variables : 'FU50hy', 'BU50hy'


"""


# Input features
input_features = ['latitude',
                  'longitude',
                  'elevation',
                  'area',
                  'meandepth',
                  'res_time',
                  'maxafdd',
                  'maxafdd_datehy',
                  'first_neg_datehy',
                  'total_radiation',
                  'solid_precip',
                  'mix_layer_depth']

feature_list = list(['Latitude', 'Longitude', 'Elevation', 
                     'Area', 'Mean depth', 'Residence time', 
                     'AFDDmax', 'LastNegTemp date', 'FirstNegTemp date', 
                     'Total SW radiation', 'Solid precipitation', 'Mixed layer depth'])


# Dataframes for classifier and regressor (freeze-up + break-up)
df_ice = df_all[['i_lake', 'ice'] + input_features + ['year']].dropna()
df_FU = df_all[['i_lake', 'FU50hy'] + input_features + ['year']].dropna()
df_BU = df_all[['i_lake', 'BU50hy'] + input_features + ['year']].dropna()


# Classifier model 
df_output_ice = np.array(df_ice['ice'].astype(int))
df_input_ice = np.array(df_ice[input_features])
groups_ice = df_ice[['i_lake', 'year']]



# Regressor models

# Freeze-up
df_input_FU = np.array(df_FU[input_features])
df_output_FU = np.array(df_FU['FU50hy'].astype(int))
groups_FU = df_FU[['i_lake', 'year']]

# Break-up
df_input_BU = np.array(df_BU[input_features])
df_output_BU = np.array(df_BU['BU50hy'].astype(int))
groups_BU = df_BU[['i_lake', 'year']]



#%% RF model - classifier and regressor serie 

# Split train/test for classifier
X_train_class, X_test_class, y_train_class, y_test_class, y_train_groups_ice, y_test_groups_ice = train_test_split(df_input_ice, df_output_ice, groups_ice, test_size = 0.8, random_state = 42)
# X_train_FU, X_test_FU, y_train_FU, y_test_FU, y_train_groups_FU, y_test_groups_FU = train_test_split(df_input_FU, df_output_FU, groups_FU, test_size = 0.8, random_state = 42)
# X_train_BU, X_test_BU, y_train_BU, y_test_BU, y_train_groups_BU, y_test_groups_BU = train_test_split(df_input_BU, df_output_BU, groups_BU, test_size = 0.8, random_state = 42)

# Count number of observations for classifier calibration 
G_ice = y_train_groups_ice[y_train_groups_ice['i_lake'].str.contains('G')]
H_ice = y_train_groups_ice[y_train_groups_ice['i_lake'].str.contains('H')]


# Train the model on training set
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_class, y_train_class) 
y_pred_train_class = rf_classifier.predict(X_train_class)

# Test the model on testing set
y_pred_test_class = rf_classifier.predict(X_test_class) 


# Confusion matrix on test set
matrix = confusion_matrix(y_test_class, y_pred_test_class)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
print(classification_report(y_pred_test_class, y_test_class))

# Figure
csfont = {'fontname':'Arial'}
plt.rc('font', size=10)  
fig, ax = plt.subplots(1, figsize=(5,3))
res = sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=1, square=True, vmin=0, vmax=1, fmt='.2f', )

cbar = res.collections[0].colorbar
cbar.outline.set_edgecolor('black')  # Set the edge color
cbar.outline.set_linewidth(1) 

# Drawing the frame 
for _, spine in res.spines.items(): 
    spine.set_visible(True) 
    spine.set_linewidth(1) 

class_names = ['No ice', 'Ice']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks2,class_names)
plt.yticks(tick_marks2,class_names, rotation=0)
plt.xlabel('Predicted')
plt.ylabel('Observed (Sentinel-2)')

plt.savefig('/Users/doris/Desktop/confusion_matrix.png', bbox_inches='tight', format="png", dpi=1200)
plt.savefig('/Users/doris/Desktop/confusion_matrix.svg', bbox_inches='tight')

# Feature importance for classifier:
# Get numerical feature importances
importances = list(rf_classifier.feature_importances_)
# List of tuples with variable and importance
feature_importances_class = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances_class = sorted(feature_importances_class, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_class];
# List of x locations for plotting
x_values = list(range(len(importances)))


# Fig. 14. 

df_classifier['Diff'] = df_classifier['y_pred_class'] - df_classifier['y_test_class']

df_classifier.to_csv('/Users/doris/Desktop/df_classifier.csv')

#%% Regressor

# Transition from classifier to regressor : keep only lakes and reservoirs that are classified "ice" 

# Dataframe classifier output
data = {'i_lake': y_test_groups_ice['i_lake'], 'year': y_test_groups_ice['year'], 'y_pred_class': y_pred_test_class, 'y_test_class': y_test_class}
df_classifier = pd.DataFrame(data).reset_index(drop=True)

# Retrieve freeze-up and break-up date for i_lake and year 
df_classifier = df_classifier.merge(df_all[['i_lake', 'year', 'FU50hy', 'BU50hy'] + input_features], on=['i_lake', 'year'], how='left')


# Transition classifier - regressor 
df_classifier1 = df_classifier[df_classifier['y_pred_class'] == 1]
df_classifier2 = df_classifier1[df_classifier1['y_test_class'] == 1]

df_FU = df_classifier2[['i_lake', 'FU50hy'] + input_features + ['year']].dropna()
df_BU = df_classifier2[['i_lake', 'BU50hy'] + input_features + ['year']].dropna()


# Freeze-up
df_input_FU = np.array(df_FU[input_features])
df_output_FU = np.array(df_FU['FU50hy'].astype(int))
groups_FU = df_FU[['i_lake', 'year']]

# Break-up
df_input_BU = np.array(df_BU[input_features])
df_output_BU = np.array(df_BU['BU50hy'].astype(int))
groups_BU = df_BU[['i_lake', 'year']]


# Regressor

rf_regressor = RandomForestRegressor()

# Split train/test
X_train_FU, X_test_FU, y_train_FU, y_test_FU, y_train_groups_FU, y_test_groups_FU = train_test_split(df_input_FU, df_output_FU, groups_FU, test_size = 0.25, random_state = 42)
X_train_BU, X_test_BU, y_train_BU, y_test_BU, y_train_groups_BU, y_test_groups_BU = train_test_split(df_input_BU, df_output_BU, groups_BU, test_size = 0.25, random_state = 42)

# Count number of observations for classifier calibration 
# Freeze-up
G_FU = groups_FU[groups_FU['i_lake'].str.contains('G')]
H_FU = groups_FU[groups_FU['i_lake'].str.contains('H')]

# Break-up
G_BU = groups_BU[groups_BU['i_lake'].str.contains('G')]
H_BU = groups_BU[groups_BU['i_lake'].str.contains('H')]



# # # # # # # # # #
# Freeze-up dates #
# # # # # # # # # #


# Train the model on training set
rf_regressor.fit(X_train_FU, y_train_FU) 
pred_train_FU = rf_regressor.predict(X_train_FU)

# Test the model on testing set
pred_test_FU = rf_regressor.predict(X_test_FU)


# Feature importance for freeze-up:
# Get numerical feature importances
importancesFU = list(rf_regressor.feature_importances_)
# List of tuples with variable and importance
feature_importances_FU = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importancesFU)]
# Sort the feature importances by most important first
feature_importances_FU = sorted(feature_importances_FU, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('FU Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_FU];
# List of x locations for plotting
x_values = list(range(len(importancesFU)))



# # # # # # # # # #
#  Break-up dates #
# # # # # # # # # #

# Train the model on training set
rf_regressor.fit(X_train_BU, y_train_BU) 
pred_train_BU = rf_regressor.predict(X_train_BU)

# Test the model on testing set
pred_test_BU = rf_regressor.predict(X_test_BU)


# Feature importance
# Get numerical feature importances
importancesBU = list(rf_regressor.feature_importances_)
# List of tuples with variable and importance
feature_importances_BU = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importancesBU)]
# Sort the feature importances by most important first
feature_importances_BU = sorted(feature_importances_BU, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('BU Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_BU];
# List of x locations for plotting
x_values = list(range(len(importancesBU)))




# # # # # # # # # #
#  Stats/analysis #
# # # # # # # # # #

MAE_train_FU, MAE_test_FU, r2_train_FU, r2_test_FU = metrics_calc(y_train_FU, y_test_FU, pred_train_FU, pred_test_FU)
MAE_train_BU, MAE_test_BU, r2_train_BU, r2_test_BU = metrics_calc(y_train_BU, y_test_BU, pred_train_BU, pred_test_BU)



#%% Plot predicted FU and predicted BU (horizontal)

# Train  
csfont = {'fontname':'Arial'}
plt.rc('font', size=10)
fig,ax = plt.subplots(1,2, figsize=(11, 4), constrained_layout=True, sharey=True)
ax[0].grid(linewidth=0.2, zorder=0)
ax[0].plot([-30,450],[-30,450], color='k', linestyle='dashed', linewidth=.5, label='_nolegend_',zorder=1)
ax[0].scatter(pred_train_FU, y_train_FU, s=.5, color='mediumseagreen', zorder=3)
ax[0].scatter(pred_train_BU, y_train_BU,  s=.5, color='red', zorder=2)
ax[0].set_xlabel('Date\nPredicted')
ax[0].tick_params(axis='both', which='both', pad=5, top=False, right=False, length=3, width=1, direction='in')
ax[0].get_xaxis().set_label_coords(0.5,-0.12)
ax[0].set_xlim([-30,450])
ax[0].set_ylim([-30,450])
ax[0].set_ylabel('Date\nObserved (Sentinel-2)')
ax[0].get_yaxis().set_label_coords(-0.12,0.5)
ax[0].set_yticks([30, 90, 150, 210, 270, 330, 390], labels=['Sep', 'Nov', 'Jan', 'Mar', 'May', 'Jul', 'Sep'])
ax[0].set_xticks([30, 90, 150, 210, 270, 330, 390], labels=['Sep', 'Nov', 'Jan', 'Mar', 'May', 'Jul', 'Sep'])
ax[0].set_title('(a) Training', pad=10, fontweight='bold')
ax[0].legend(['Freeze-up (MAE={} days, R$^2$={:.2f})'.format(int(MAE_train_FU), r2_train_FU), 
            'Break-up (MAE={} days, R$^2$={:.2f})'.format(int(MAE_train_BU), r2_train_BU)], 
          markerscale=3, frameon=False, loc='upper left')

# Test  
ax[1].grid(linewidth=0.2, zorder=0)    
ax[1].plot([-30,450],[-30,450], color='k', linestyle='--', linewidth=0.5, label='_nolegend_', zorder=1)
ax[1].scatter(pred_test_FU, y_test_FU,  s=0.5, color='mediumseagreen', zorder=3)
ax[1].scatter(pred_test_BU, y_test_BU,  s=0.5, color='red', zorder=2)
ax[1].tick_params(axis='both', which='both', pad=5, top=False, right=False, length=3, width=1, direction='in')
ax[1].set_xlabel('Date\nPredicted')
ax[1].get_xaxis().set_label_coords(0.5,-0.12)
ax[1].set_xticks([30, 90, 150, 210, 270, 330, 390], labels=['Sep', 'Nov', 'Jan', 'Mar', 'May', 'Jul', 'Sep'])
ax[1].set_xlim([-30,450])
ax[1].get_yaxis().set_label_coords(-0.12,0.5)
ax[1].set_ylim([-30,450])
ax[1].set_title('(b) Testing', pad=10, fontweight='bold')
ax[1].legend(['Freeze-up (MAE={} days, R$^2$={:.2f})'.format(int(MAE_test_FU), r2_test_FU), 
            'Break-up (MAE={} days, R$^2$={:.2f})'.format(int(MAE_test_BU), r2_test_BU)], 
         markerscale=3, frameon=False, loc='upper left')

fig.savefig('/Users/doris/Desktop/performance_reg_combined.png', bbox_inches='tight', format="png", dpi=1200)
# fig.savefig('/Users/doris/Desktop/performance_reg_combined.pdf', bbox_inches='tight')
fig.savefig('/Users/doris/Desktop/performance_reg_combined.svg', bbox_inches='tight')





#%% Feature importance

feature_importances_class = pd.DataFrame(feature_importances_class)
feature_importances_FU = pd.DataFrame(feature_importances_FU)
feature_importances_BU = pd.DataFrame(feature_importances_BU)


# Classifier 
csfont = {'fontname':'Arial'}
plt.rc('font', size=22)  
fig,ax = plt.subplots(1, figsize=(7, 8), constrained_layout=True)
    #Freeze-up
ax.barh(feature_importances_class[0], feature_importances_class[1], color='blue')
ax.set_xlim([0,0.6])
ax.set_yticks(x_values, feature_importances_class[0])
ax.set_xlabel('Feature importance')
ax.get_xaxis().set_label_coords(0.5,-0.09)
ax.invert_yaxis()
ax.tick_params(axis='x', which='both', pad=10, top=False, right=False, length=5, width=1, direction='in')
ax.set_xticks(np.arange(0, 0.7, 0.1))




# Regressor
csfont = {'fontname':'Arial'}
plt.rc('font', size=10)  
fig,ax = plt.subplots(1,2, figsize=(7, 4), constrained_layout=True)

# Freeze-up    
ax[0].barh(feature_importances_FU[0], feature_importances_FU[1], color='mediumseagreen')
ax[0].set_xlim([0,0.6])
ax[0].set_yticks(x_values, feature_importances_FU[0])
ax[0].set_xlabel('Feature importance')
ax[0].get_xaxis().set_label_coords(0.5,-0.09)
ax[0].set_title('(a) Freeze-up date', pad=10, fontweight="bold")
ax[0].invert_yaxis()
ax[0].tick_params(axis='x', which='both', pad=5, top=False, right=False, length=3, width=1, direction='in')
ax[0].set_xticks(np.arange(0, 0.7, 0.1))

# Break-up    
ax[1].barh(feature_importances_BU[0], feature_importances_BU[1], color='red')
ax[1].set_xlim([0,0.6])
ax[1].set_yticks(x_values, feature_importances_BU[0])
ax[1].set_xlabel('Feature importance')
ax[1].get_xaxis().set_label_coords(0.5,-0.09)
ax[1].set_title('(b) Break-up date', pad=10, fontweight="bold")
ax[1].invert_yaxis()
ax[1].tick_params(axis='x', which='both', pad=5, top=False, right=False, length=3, width=1, direction='in')
ax[1].set_xticks(np.arange(0, 0.7, 0.1))

fig.savefig('/Users/doris/Desktop/Feature_importance_res_lakes_combined_reg.png', bbox_inches='tight', format="png", dpi=1600)
fig.savefig('/Users/doris/Desktop/Feature_importance_res_lakes_combined_reg.pdf', bbox_inches='tight')
fig.savefig('/Users/doris/Desktop/Feature_importance_res_lakes_combined_reg.svg', bbox_inches='tight')

