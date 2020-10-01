# %%
import pandas as pd

# %%
def transform_data_prev(df):
    """
    This function is used for extracting information from the original column "previous", 
    and transforming to a standard dataframe for analysis 
    """

    # split using separator ';' because 1 customer may have multiple previous application
    df_temp = (
        df.previous.dropna()
        .str.split(';', expand=True)
    )
    
    # convert the wide format data to long format data, each application record put into diffrent row
    df_temp = (
        pd.concat([df.id, df_temp], axis=1)
        .pipe(pd.melt, id_vars=['id'])
        .rename(columns={'variable': 'ith_prev_app'})
    )

    # the first previous application starts from 1 instead of 0
    df_temp['ith_prev_app'] += 1
    
    # drop invalid rows
    df_temp = df_temp.loc[pd.notnull(df_temp.value), :]

    # split the information of each previous application record
    # since no additional information about the fields, I simply set them to prev_var1, prev_var2 and prev_var3 
    df_temp_val = (
        df_temp.value.str.split(',', expand=True)
        .rename(columns={
            0: 'prev_app_time',
            1: 'prev_var1',
            2: 'prev_var2',
            3: 'prev_var3'})
    )

    df_prev = (
        pd.concat([df_temp.loc[:, ['id', 'ith_prev_app']], df_temp_val], axis=1)
        .reset_index(drop=True)
    )

    # set datatype for the fields
    df_prev.loc[:, 'ith_prev_app'] = \
        df_prev.loc[:, 'ith_prev_app'].astype('int')
    
    df_prev.loc[:, ['prev_var1', 'prev_var2', 'prev_var3']] = \
        df_prev.loc[:, ['prev_var1', 'prev_var2', 'prev_var3']].apply(lambda x: x.astype('float'))

    # create feature 'years_since_app_prev': the number of years from previous application dates onwards
    current_date = pd.datetime.now()
    
    df_prev['prev_app_time'] = pd.to_datetime(df_prev.prev_app_time).dt.date
    
    df_prev['years_since_app_prev'] = (
        (pd.Series(current_date, index=df_prev.index).dt.date - df_prev.prev_app_time).apply(lambda y: y.days / 365)
    )

    return df_prev

# %%
def aggregate_data_prev(df):
    """
    This function is for generating new features of the previous application data by aggregating the numeric features 
    """

    func_agg = {
        'ith_prev_app': ['max', 'mean'],
        'years_since_app_prev': ['max', 'mean'],
        'prev_var1': ['min', 'max', 'mean', 'sum'],
        'prev_var2': ['min', 'max', 'mean'],
        'prev_var3': ['min', 'max', 'mean', 'sum']
    }

    df_agg = (
        df.groupby('id')
        .agg(func_agg)
    )
    
    # standardize the column names of the derived features
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns])

    return df_agg

# %%