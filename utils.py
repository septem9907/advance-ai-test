# %%
import pandas as pd

# %%
# Check missing data
def check_missing_data(df):
    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )
    percent = (
        df.isnull().sum() / 
        df.isnull().count()
        .sort_values(ascending=False)
    )
    res = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])
    res = res.loc[res.Missing > 0, :]
    return res

# %%
def check_correlation(df, cols):
    return df[cols].corr()

# %%
def scale_numeric_cols(df, cols):
    
    cols_mean = df[cols].mean()
    cols_std = df[cols].std()

    df.loc[:, cols] = (df.loc[:, cols] - cols_mean) / cols_std

    return df

# %%
def display_top_n_features(arr_feature_names, arr_feature_importances, n_features=20):

    assert len(arr_feature_importances) == len(arr_feature_names) 

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # construct the dataframe of feature importance info
    df_feature_importance = pd.DataFrame(index=range(len(arr_feature_importances)), columns=['feature', 'importance'])
    df_feature_importance['feature'] = arr_feature_names
    df_feature_importance['importance'] = arr_feature_importances
    # sort by importance of features
    df_top_n_features = df_feature_importance.sort_values(by='importance', ascending=False)[:n_features]

    # plot the barchart of the feature importance dataframe
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=df_top_n_features)
    plt.title('notable features')
    plt.tight_layout()

# %%
def plot_roc_auc(label, pred):
    
    # construct roc curve and calculate roc_auc
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(label, pred)

    roc_auc = auc(fpr, tpr)
    
    # plot the roc curve and display roc_auc
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
