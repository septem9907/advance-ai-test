
# %% [markdown]
# overfitting
# Overfitting happens when the model is too complicated relative to the training data. 
# When the model learns the training samples "too well", probabaly it incorporates the property of the specific samples and it descreses the generalization ability of the model.

# %% [markdown]
# regularization
# Constraining a model to make it simpler and reduce overfitting is called regularization.

# %% [markdown]
# ensemble
# A group of predictors is called ensemble. Usually aggregating the predictions of a group of predictors will get better predictions than using an individual predictor.
# bagging, boosting and stacking are the most common ensemble methods we use.

# %%
# code example
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.datasets import load_boston


# %%
data = load_boston()
y = data['target']
X = data['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# %%
reg_dt = DecisionTreeRegressor(random_state=1)
reg_dt.fit(X_train, y_train)

y_train_pred = reg_dt.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_test_pred = reg_dt.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(rmse_train)
print(rmse_test)

# %%
reg_xgb = XGBRegressor(n_estimators=10, random_state=1)
reg_xgb.fit(X_train, y_train)

y_train_pred = reg_xgb.predict(X_train)
y_test_pred = reg_xgb.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(rmse_train)
print(rmse_test)


# %%
reg_xgb_penalty = XGBRegressor(n_estimators=10, reg_lambda=2, random_state=1)
reg_xgb_penalty.fit(X_train, y_train)

y_train_pred = reg_xgb_penalty.predict(X_train)
y_test_pred = reg_xgb_penalty.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(rmse_train)
print(rmse_test)

# %% [markdown]
# the decision tree regressor is obviously overfitting as the root mean sqaure deviation of training set is 0
# while I used the ensemble method (gradient boosting decision tree) to train the model, the prediction is more accurate since RMSE of test set decreases from  5.66 to 3.21
# after applying the L2 regularization to the GBDT, we reduced a overfitting since the difference of RMSE between training set and test set decreases
