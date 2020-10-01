# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# load the raw data

# %%
df = pd.read_csv('./data/sample_200_0k_20170120.csv')
df['id'] = df.index.values + 1

# %%
# create default indicator using the field "MaxOverDueDays"
days_default_point = 90
df = (
    df.assign(is_default=lambda x: x.MaxOverDueDays.apply(lambda y: 1 if y > days_default_point else 0))
    .drop(columns=['MaxOverDueDays'])
)
print(df.is_default.value_counts() / df.shape[0])

# %%
# covert fields for date info to datetime foramt
l_date = [col for col in df.columns if 'date' in col]
for col in l_date:
    df[col] = pd.to_datetime(df[col]).dt.date


# %% [markdown]
# data exploration and feature engineering

# four kinds of features:
# 1. demographic information
# 2. geographic information(latitude, longtitude, zipcode, distance)
# 3. aggregation features: numeric features (avg, std, cnt) 
#     3.1 income, income_nation, income_area, 
#     3.2 sale_house_price(5000, 10000), sale_apartment_price(5000, 10000), 
#     3.3 rent_house_price(5000, 10000), rent_apartment_price(5000, 10000)
# 4. previous: need to extract and transform the data in the column "previous"


# %%
# missing data
from utils import check_missing_data
df_missing_data = check_missing_data(df)
print(df_missing_data)

# %% [markdown]
# 2 fields with considerable missing data are "MainBusinessSinceYear" and "EmploymentSinceYear", but I observed that many sample at least have 1 field out of the 2.
# I did a quick check to find out how many samples have no information in both fields

# %%
df_temp = df.loc[:, ['EmploymentSinceYear', 'MainBusinessSinceYear']]
print(pd.notnull(df_temp).all(axis=1).value_counts())
print(pd.notnull(df_temp).any(axis=1).value_counts())
del df_temp

# %% [markdown]
# from the result above, we can find that only 30 samples don't have any information and 199970 samples have useful data

# %%
# use these 2 fields to create a new features "work_since_year"
df['work_since_year'] = np.minimum(df.EmploymentSinceYear.fillna(9999), df.MainBusinessSinceYear.fillna(9999)).astype('int')

# use the median value to replace the abnormal value
medval_work_since_year = df['work_since_year'].median()
df['work_since_year'] = df['work_since_year'].replace([0, 9999], medval_work_since_year)

# drop the fields "EmploymentSinceYear" and "MainBusinessSinceYear"
df = df.drop(columns=['EmploymentSinceYear', 'MainBusinessSinceYear'])

# %%
# jobpos, I find that the missing percentage of this field is around 43%, so I try to set "Missing" as one category 
df['jobpos'] = df.jobpos.fillna('Missing')

# since the percentage of samples with missing data of these features is around 2%, I just use the most common categories/values to fill the missing entries
df['jobtypeid'] = df.jobtypeid.fillna('Others')
df['maritalstatus'] = df.maritalstatus.fillna('MARRIED')
df['spouseincome'] = df.spouseincome.fillna(0)
df['distance_residence_company'] = df.distance_residence_company.fillna(0)

# derived features with missing data, the percentage is also very low, use median values to fill
cols_derived_with_missing = [col for col in df_missing_data.index.values if (('avg' in col) or ('std' in col) or ('cnt' in col))]
df.loc[:, cols_derived_with_missing] = (
    df.loc[:, cols_derived_with_missing]
    .apply(lambda x: x.fillna(x.median()))
    )

# company zip, longitude and latitude, low percentage, use median values to fill
cols_geo_with_missing = ['companyzipcode', 'company_long', 'company_lat']
df.loc[:, cols_geo_with_missing] = (
    df.loc[:, cols_geo_with_missing]
    .apply(lambda x: x.fillna(x.median()))
    )

# %%
# check missing data again
df_missing_data = check_missing_data(df)
print(df_missing_data)

# %% [markdown]
# now only the field "previous" still has missing data.
# The information in this field seems useful, I will transform the data and explore in the later part instead of simply handle it.


# %% [markdown]
# 1. demographic information

# %%
# create 3 new features: 
# age: current age of the customer
# age_first_app: age when the customer first apply loans
# years_since_app: number of years the customer have applied loans

current_date = pd.datetime.now()

df = (
    df.assign(
        age=lambda x: (pd.Series(current_date, index=x.index).dt.date - x.birthdate).apply(lambda y: int(y.days / 365)),
        age_first_app=lambda x: (x.newapplicationdate - x.birthdate).apply(lambda y: int(y.days / 365)),
        years_since_app=lambda x: (pd.Series(current_date, index=x.index).dt.date - x.newapplicationdate).apply(lambda y: y.days / 365))
    .drop(columns=['birthdate', 'newapplicationdate'])
)

# %%
# create features about income of the customer:
# monthly_income_per_family_member: monthly family income divided number of the family
# income_over_workingyears: monthly income divided by the number of years during working
# income_over_nationlevel: monthly income divided by the averge income of the nation
# income_over_arealevel: monthly income divided by the averge income of the area
df = (
    df.assign(
        monthly_income_per_family_member=lambda x: (x.monthlyfixedincome + x.monthlyvariableincome + x.spouseincome) / (x.numofdependence + 1),
        income_over_workingyears=lambda x: (x.monthlyfixedincome + x.monthlyvariableincome) / (current_date.year - x.work_since_year),
        income_over_nationlevel=lambda x: (x.monthlyfixedincome + x.monthlyvariableincome) / x.avg_income_nation,
        income_over_arealevel=lambda x: (x.monthlyfixedincome + x.monthlyvariableincome) / x.avg_income_area
    )
)

# %%
# convert the fields "numofdependence" and "homestatus" from numeric type to categorical type
df.loc[:, ['numofdependence', 'homestatus']] = \
    df.loc[:, ['numofdependence', 'homestatus']].apply(lambda x: x.astype(object))

# drop the field birthplace as the number of categories of this field is too big...
df = df.drop(columns=['birthplace'])


# %% [markdown]
# 2. geographic features

# %%
# There are 3 types of geographic information in the dataset: residence, company, legal
# quick check the correlation of these 3 types

from utils import check_correlation

print(df.pipe(check_correlation, ['residence_lat', 'company_lat', 'legal_lat']))
print(df.pipe(check_correlation, ['residence_long', 'company_long', 'legal_long']))
print(df.pipe(check_correlation, ['residencezipcode', 'companyzipcode', 'legalzipcode']))

# %% [markdown] 
# from the correlation matrix, we can only keep one type of latitude/longitude/zipcode data, to reduce the number of features 
# say only keep the type "legal"

# %%
df = df.drop(columns=['residence_lat', 'company_lat', 'residence_long', 'company_long', 'residencezipcode', 'companyzipcode'])

# %%
# now I try to create a scatterplot to visualze the data
df.plot(kind='scatter', x='legal_long', y='legal_lat', alpha=0.4,
    figsize=(10, 7), c='is_default', cmap=plt.get_cmap('jet'), colorbar=True, sharex=False)

# %% [markdown]
# It seems that the default rate of area from (106, -6) to (118, -8) is high. Below is a quick check:

# %%
flag_high_default = (df.legal_long < 118) & (df.legal_lat < -6) & \
    (df.legal_long > 106) & (df.legal_lat > -8)

df_temp = df.loc[flag_high_default, :]

print(df_temp.is_default.value_counts() / df_temp.shape[0])
del df_temp

# %% [markdown]
# The default rate is 19.2%, 23% higher than the default rate of the population (15.6%).
# According to this, I will create a feature "high_default_area"

# %%
df['high_default_area'] = 0
df.loc[flag_high_default, 'high_default_area'] = 1

# %% [markdown]
# 3. derived features

# %%
# I noticed that the values of the derived features are quite big  
# try to scale them to smaller level

from utils import scale_numeric_cols

cols_derived = [col for col in df.columns if (('avg' in col) or ('std' in col) or ('cnt' in col))]
cols_income = [col for col in df.columns if (('income' in col) and (col not in cols_derived))]

df = (
    df.pipe(scale_numeric_cols, cols_derived)
    .pipe(scale_numeric_cols, cols_income)
)

# %% [markdown]
# 4. previous application record

# %%
from previous import transform_data_prev, aggregate_data_prev

# the pipeline below is for extracting, transforming and aggregating data in the "previous" column
# the purpose is to generate new features for the next modelling part 
df_prev_agg = (
    df.pipe(transform_data_prev)
    .pipe(aggregate_data_prev)
)

# scaling 
cols_from_prev = [col for col in df_prev_agg.columns if (('var1' in col) or ('var3' in col))]
df_prev_agg = df_prev_agg.pipe(scale_numeric_cols, cols_from_prev)

# create a new feature "is_prev_app": whether a customer has previous application record, assign 1 if yes, otherwise 0
df['is_prev_app'] = 0
df.loc[df.previous.notnull(), 'is_prev_app'] = 1

# drop the original "previous" column
df = df.drop(columns=['previous'])


# %% [markdown]
# now I prepare the data for machine learning algorithms later.

# %%
# the pipeline below:
# 1. one-hot encoding for the categorical variables
# 2. merge the derived features from previous application records to the original data
# 3. for customer id without previous application, put 0 to the columns related to previous application 
df = (
    df.pipe(pd.get_dummies)
    .merge(df_prev_agg, how='left', left_on='id', right_index=True)
    .fillna(0)
)

# %% [markdown]
# Select and Train a Model

# %%
# create training set and test set, use the stratified spliting to make the distribution of training and test set similar.
from sklearn.model_selection import train_test_split
X = df.drop(columns=['id', 'is_default'])
y = df['is_default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# %% [markdown]
# random forest

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

clf_rf = RandomForestClassifier()
param_grid_rf = {'n_estimators': [100, 300], 'max_depth': [4, 6], 'class_weight': ['balanced'], 'random_state': [1]}

grid_search_rf = GridSearchCV(clf_rf, param_grid_rf, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# %%
# print the auc of the shortlisted models
for auc, params in zip(grid_search_rf.cv_results_['mean_test_score'], grid_search_rf.cv_results_['params']):
    print(auc, params)
# select the best estimator of gridsearch as the model
model_rf = grid_search_rf.best_estimator_

# %% [markdown]
# check out the AUROC of xgboost model

# %%
from utils import plot_roc_auc

# %%
# plot the auc_roc of the training set
y_train_pred_rf = model_rf.predict_proba(X_train)[:, 1]
plot_roc_auc(y_train, y_train_pred_rf)

# %%
# plot the auc_roc of the test set
y_test_pred_rf = model_rf.predict_proba(X_test)[:, 1]
plot_roc_auc(y_test, y_test_pred_rf)

# %% [markdown]
# xgboost

# %%
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

clf_xgb = xgb.XGBClassifier()
param_grid_xgb = [
    {'n_estimators': [50, 200], 'max_depth': [3, 4], 'learning_rate': [0.1], 'random_state': [1]}
]

grid_search_xgb = GridSearchCV(clf_xgb, param_grid_xgb, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

# %%
# print the auc of the shortlisted models
for auc, params in zip(grid_search_xgb.cv_results_['mean_test_score'], grid_search_xgb.cv_results_['params']):
    print(auc, params)
# select the best estimator of gridsearch as the model
model_xgb = grid_search_xgb.best_estimator_


# %% [markdown]
# check out the AUROC of xgboost model

# %%
# plot the auc_roc of the training set
y_train_pred_xgb = model_xgb.predict_proba(X_train)[:, 1]
plot_roc_auc(y_train, y_train_pred_xgb)

# %%
# plot the auc_roc of the test set
y_test_pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
plot_roc_auc(y_test, y_test_pred_xgb)

# %% [markdown]
# the auc of test set is 3% lower than the one of training set, so still need to fine tune the model to avoid overfitting...


# %% [markdown]
# feature importance

# %%
from utils import display_top_n_features

feature_importances = model_xgb.feature_importances_
feature_names = X_train.columns.values
display_top_n_features(feature_names, feature_importances)

# %% [markdown]
# based on the above barchart, I found that the previous application information is helpful indicator to predict default risk.
# the feature importance of "is_prev_app", "prev_var2_max"(I guess "prev_var2" is the max overdue days of previous application), "prev_var3_mean", "prev_var2_mean" and "ith_prev_app_max"
# are among the top 20 features selected.
# features "jobpos_Others" and "jobpos_Missing" are also notable indicator, it implies that high risk customers probably won't have specific information of the job positions.
# feature "high_default_area" is important, which is as expected.


# %%
# save the model
# model_xgb.save_model('./model/advance-ai-test.model_xgb_v1')
# model_rf.save_model('./model/advance-ai-test.model_rf_v1')