"""
Pick out a data set and prep it to be used to train a model

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def run_prep(problem_name):
    
    if problem_name == 'monks':
        # Import data
        train_pkl = 'data/interim/monk_interim_train.pkl'
        test_pkl  = 'data/interim/monk_interim_test.pkl'
        df_train, df_test = pd.read_pickle(train_pkl), pd.read_pickle(test_pkl)

        # Data prep
        y_label = 'y'
        y_train = df_train[y_label].copy()
        x_train = df_train.drop(columns=[y_label])

        y_test = df_test[y_label].copy()
        x_test = df_test.drop(columns=[y_label])

        # Encode binary target class... not sure if/how this makes any difference
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)
    
    elif problem_name == 'synthetic':
        import src.nn_synthetic_data.nn_data as nn_data
        from src.nn_synthetic_data.make_data import plot_synthetic_dataset
        x_train, y_train, x_test, y_test = nn_data.get_sample_test_train(10,10,0.0)
        # plot_synthetic_dataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, transparent_bg=False, bg_color='black')

    # Standard Scaler
    scaler_cred = StandardScaler()
    scaler_cred.fit(x_train)
    x_train = scaler_cred.transform(x_train)
    x_test = scaler_cred.transform(x_test)
    
    return x_train, y_train, x_test, y_test