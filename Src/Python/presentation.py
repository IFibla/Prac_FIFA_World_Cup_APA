from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
import numpy as np
import time
import os

def dataframe_to_dictionary(df, key, continue_list=None):
    result = dict()
    for _key, _val in zip(df[key].keys(), df[key].values):
        if _key not in continue_list:
            result[_key] = _val
    return result

def replace_string_to_int_dataframe(df, keys, input):
    for keys, inp in zip(keys, input):
        df[keys].replace(inp, list(range(len(inp))), inplace=True)
    return df

def delete_from_dataframe(df, delete_keys):
    for key in delete_keys:
        del df[key]
    return df

if 'rd_alpha' not in st.session_state: st.session_state['rd_alpha'] = 1
if 'kd_neigh' not in st.session_state: st.session_state['kd_neigh'] = 28
if 'rf_estimators' not in st.session_state: st.session_state['rf_estimators'] = 200
if 'svm_value' not in st.session_state: st.session_state['svm_value'] = 1.99
if 'mlp_value' not in st.session_state: st.session_state['mlp_value'] = 0.001
if 'to_simulate' not in st.session_state: st.session_state['to_simulate'] = True
if 'home_team' not in st.session_state: st.session_state['home_team'] = None
if 'away_team' not in st.session_state: st.session_state['away_team'] = None
if 'simulation_results' not in st.session_state: st.session_state['simulation_results'] = None

path_to_scaler = '../../Data/international_matches.csv'
path_to_dataset = '../../Data/international_matches_clean.csv'
path_to_teams = '../../Data/fifa_teams_score.csv'


st.markdown("<h1 style='text-align: center'>FIFA World Cup Predictor.</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center'>Ignasi Fibla Figuerola  &  Mark Smithson Rivas</h5>", unsafe_allow_html=True)

if os.path.isfile(path_to_dataset) and 'dataset' not in st.session_state:
    df = pd.read_csv(path_to_dataset)
    X = df[df.columns.difference(['result'])]
    y = df[['result']]
    st.session_state['dataset'] = df
    st.session_state['X_train'], st.session_state['X_test'], st.session_state['y_train'], st.session_state['y_test'] = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler_df = pd.read_csv(path_to_scaler, delimiter=';')
    scaler_df = scaler_df[~scaler_df.isnull().any(axis=1)]
    scaler_df = delete_from_dataframe(scaler_df, ['date', 'home_team', 'away_team', 'tournament', 'city', 'country', 'neutral_location', 'home_team_result'])
    replace_keys = ['home_team_continent', 'away_team_continent', 'shoot_out']
    replace_input = [['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'], ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'], ['No', 'Yes']]
    scaler_df = replace_string_to_int_dataframe(scaler_df, replace_keys, replace_input)
    scaler_df = scaler_df[scaler_df.columns.difference(['home_team_score', 'away_team_score'])]
    st.session_state['scaler'] = MinMaxScaler().fit(scaler_df)

if 'dataset' in st.session_state:
    model_selector = st.selectbox(
                    'Select the desired model:',
                    ('Linear Regression', 'Ridge', 'KNeighbors Regressor', 'Random Forest', 'SVM con kernel RBF', 'MLP')
                )

    if model_selector == 'Linear Regression':
        st.session_state['to_simulate'] = True
        if 'lr' not in st.session_state:
            t0 = time.time()
            st.session_state['lr'] = LinearRegression()
            st.session_state['lr'].fit(st.session_state['X_train'], st.session_state['y_train'].squeeze())
            st.session_state['lrtime'] = time.time() - t0
        train_time = st.session_state['lrtime']
        model_score = st.session_state['lr'].score(st.session_state['X_test'], st.session_state['y_test'].squeeze())
        model_cross = np.array(cross_val_score(st.session_state['lr'], st.session_state['X_test'], st.session_state['y_test'].squeeze()))
        title = 'Linear Regression Results'

    elif model_selector == 'Ridge':
        st.session_state['to_simulate'] = True
        rd_alpha = st.slider("Select the alpha value:", 0, 40, value=st.session_state['rd_alpha'], step=1)
        if 'rd' not in st.session_state or rd_alpha != st.session_state['rd_alpha']:
            st.session_state['rd_alpha'] = rd_alpha
            t0 = time.time()
            st.session_state['rd'] = Ridge(alpha=st.session_state['rd_alpha'])
            st.session_state['rd'].fit(st.session_state['X_train'], st.session_state['y_train'].squeeze())
            st.session_state['rdtime'] = time.time() - t0
        train_time = st.session_state['rdtime']
        model_score = st.session_state['rd'].score(st.session_state['X_test'], st.session_state['y_test'].squeeze())
        model_cross = np.array(cross_val_score(st.session_state['rd'], st.session_state['X_test'], st.session_state['y_test'].squeeze()))
        title = 'Ridge Results with alpha of {alpha}'.format(alpha=st.session_state['rd_alpha'])

    elif model_selector == 'KNeighbors Regressor':
        st.session_state['to_simulate'] = True
        kd_neigh = st.slider("Select the number of neighbors:", 1, 800, value=st.session_state['kd_neigh'], step=1)
        if 'kd' not in st.session_state or kd_neigh != st.session_state['kd_neigh']:
            st.session_state['kd_neigh'] = kd_neigh
            t0 = time.time()
            st.session_state['kd'] = KNeighborsRegressor(n_neighbors=st.session_state['kd_neigh'])
            st.session_state['kd'].fit(st.session_state['X_train'], st.session_state['y_train'].squeeze())
            st.session_state['kdtime'] = time.time() - t0
        train_time = st.session_state['kdtime']
        model_score = st.session_state['kd'].score(st.session_state['X_test'], st.session_state['y_test'].squeeze())
        model_cross = np.array(cross_val_score(st.session_state['kd'], st.session_state['X_test'], st.session_state['y_test'].squeeze()))
        title = 'KNeighbors Regressor with a number of neighbors {neighbors}'.format(neighbors=st.session_state['kd_neigh'])

    elif model_selector == 'Random Forest':
        st.session_state['to_simulate'] = True
        rf_estimators = st.slider("Select the number of estimators:", 5, 300, value=st.session_state['rf_estimators'], step=1)
        if 'rf' not in st.session_state or rf_estimators != st.session_state['rf_estimators']:
            st.session_state['rf_estimators'] = rf_estimators
            t0 = time.time()
            st.session_state['rf'] = RandomForestRegressor(random_state=0, criterion='squared_error', max_depth=5, min_samples_leaf=2, n_estimators=st.session_state['rf_estimators'])
            st.session_state['rf'].fit(st.session_state['X_train'], st.session_state['y_train'].squeeze())
            st.session_state['rftime'] = time.time() - t0
        train_time = st.session_state['rftime']
        model_score = st.session_state['rf'].score(st.session_state['X_test'], st.session_state['y_test'].squeeze())
        model_cross = np.array(cross_val_score(st.session_state['rf'], st.session_state['X_test'], st.session_state['y_test'].squeeze()))
        title = 'Random Forest with a number of estimators {estimators}'.format(estimators=st.session_state['rf_estimators'])

    elif model_selector == 'SVM con kernel RBF':
        st.session_state['to_simulate'] = True
        svm_value = st.slider("Select the C value:", 0.00, 3.00, value=st.session_state['svm_value'], step=0.01)
        if 'svm' not in st.session_state or svm_value != st.session_state['svm_value']:
            st.session_state['svm_value'] = svm_value
            t0 = time.time()
            st.session_state['svm'] = SVC(kernel='rbf', max_iter=25000, random_state=0, C=st.session_state['svm_value'], gamma='auto')
            st.session_state['svm'].fit(st.session_state['X_train'], st.session_state['y_train'].squeeze())
            st.session_state['svmtime'] = time.time() - t0
        train_time = st.session_state['svmtime']
        model_score = st.session_state['svm'].score(st.session_state['X_test'], st.session_state['y_test'].squeeze())
        model_cross = np.array(cross_val_score(st.session_state['svm'], st.session_state['X_test'], st.session_state['y_test'].squeeze()))
        title = 'SVM con kernel RBF with a C value of {c}'.format(c=st.session_state['svm_value'])

    elif model_selector == 'MLP':
        st.session_state['to_simulate'] = True
        mlp_value = st.slider("Select the learning rate:", 0.0001, 0.1000, value=st.session_state['mlp_value'], step=0.0001)
        if 'mlp' not in st.session_state or mlp_value != st.session_state['mlp_value']:
            st.session_state['mlp_value'] = mlp_value
            t0 = time.time()
            st.session_state['mlp'] = MLPRegressor(max_iter=10000, early_stopping=True, n_iter_no_change=20,learning_rate='adaptive',random_state=0,activation='relu', hidden_layer_sizes=200, learning_rate_init=st.session_state['mlp_value'])
            st.session_state['mlp'].fit(st.session_state['X_train'], st.session_state['y_train'].squeeze())
            st.session_state['mlptime'] = time.time() - t0
        train_time = st.session_state['mlptime']
        model_score = st.session_state['mlp'].score(st.session_state['X_test'], st.session_state['y_test'].squeeze())
        model_cross = np.array(cross_val_score(st.session_state['mlp'], st.session_state['X_test'], st.session_state['y_test'].squeeze()))
        title = 'MLP Regressor with a learning rate value of {learn}'.format(learn=st.session_state['mlp_value'])

    res_dict = {'Training Time': train_time, 'R^2': model_score, 'Cross Validation': np.mean(model_cross)}
    cross_dict = {i: model_cross[i] for i in range(len(model_cross))}
    res_table = pd.DataFrame(res_dict, index=[0])
    cross_table = pd.DataFrame(cross_dict, index=[0])
    st.markdown("<h5>{title}</h5>".format(title=title), unsafe_allow_html=True)
    st.table(res_table)
    st.write('The following table shows the individual results for the cross validation.')
    st.table(cross_table)

if os.path.isfile(path_to_teams) and 'teams' not in st.session_state:
    st.session_state['teams'] = pd.read_csv(path_to_teams, delimiter=';')

if 'dataset' in st.session_state and 'teams' in st.session_state:
    st.markdown("<h2 style='text-align: center'>Match Prediction</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([4, 1, 4])
    
    with col1:
        home_team = st.selectbox(
                'Select the home team:',
                set(st.session_state['teams']['name'].to_list())
            )
        home_list = st.session_state['teams'][st.session_state['teams']['name'] == home_team].values[0]
        st.markdown('<img src="{url}" width="100%" height="100%">'.format(url=home_list[2]), unsafe_allow_html=True)

        home_dict = {'Goalkeeper': home_list[5], 'Defense': home_list[6], 'Midfield': home_list[7], 'Offense': home_list[8], 'Total': home_list[9]}
        home_table = pd.DataFrame(home_dict, index=[0])
        st.table(home_table)
        if st.session_state['home_team'] is None or st.session_state['home_team'][0] != home_team:
            st.session_state['home_team'] = home_list
            st.session_state['to_simulate'] = True

    with col3:
        away_team = st.selectbox(
                'Select the away team:',
                set(st.session_state['teams']['name'].to_list())
            )
        away_list = st.session_state['teams'][st.session_state['teams']['name'] == away_team].values[0]

        st.markdown('<img src="{url}" width="100%" height="100%">'.format(url=away_list[2]), unsafe_allow_html=True)

        away_dict = {'Goalkeeper': away_list[5], 'Defense': away_list[6], 'Midfield': away_list[7], 'Offense': away_list[8], 'Total': away_list[9]}
        away_table = pd.DataFrame(away_dict, index=[0])
        st.table(away_table)
        if st.session_state['away_team'] is None or st.session_state['away_team'][0] != away_team:
            st.session_state['away_team'] = away_list
            st.session_state['to_simulate'] = True

    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            width: 100%;
        }
        </style>""", unsafe_allow_html=True)

    if st.session_state['to_simulate'] and st.button('Simular'):
        home = st.session_state['home_team']
        away = st.session_state['away_team']
        x = [[away[3], away[4], away[5], away[6], away[7], away[8], away[9], home[3], home[4], home[5], home[6], home[7], home[8], home[9], False]]
        x = st.session_state['scaler'].transform(x)
        st.session_state['to_simulate'] = False
        if model_selector == 'Linear Regression':
            st.session_state['simulation_results'] = st.session_state['lr'].predict(x)

        elif model_selector == 'Ridge':
            st.session_state['simulation_results'] = st.session_state['rd'].predict(x)

        elif model_selector == 'KNeighbors Regressor':
            st.session_state['simulation_results'] = st.session_state['kd'].predict(x)

        elif model_selector == 'Random Forest':
            st.session_state['simulation_results'] = st.session_state['rf'].predict(x)

        elif model_selector == 'SVM con kernel RBF':
            st.session_state['simulation_results'] = st.session_state['svm'].predict(x)

        elif model_selector == 'MLP':
            st.session_state['simulation_results'] = st.session_state['mlp'].predict(x)
    
    if not st.session_state['to_simulate']:
        result = int(st.session_state['simulation_results'])
        if result == 0:
            home_result = 0
            away_result = 0
        elif result < 0:
            home_result = 0
            away_result = result
        else:
            home_result = result
            away_result = 0
        col1, col2, col3 = st.columns([4, 1, 4])
        with col1:
            st.markdown("<h1 style='text-align: center'>{home}</h1>".format(home=home_result), unsafe_allow_html=True)
        with col3:
            st.markdown("<h1 style='text-align: center'>{away}</h1>".format(away=away_result), unsafe_allow_html=True)

    
