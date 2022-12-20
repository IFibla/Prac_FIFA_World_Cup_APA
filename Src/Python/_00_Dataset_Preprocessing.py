import copy
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def print_information_about_dataset(df):
    for it, key in enumerate(df.keys()): print(it, key, len(df[df[key].notna()]))
    print('From the dataset, after deleting the NaN values, we have found that we have only', 
            len(df[~df.isnull().any(axis=1)]), 'rows remaining.')

def delete_from_dataframe(df, delete_keys):
    for key in delete_keys:
        del df[key]
    return df

def replace_string_to_int_dataframe(df, keys, input):
    for keys, inp in zip(keys, input):
        df[keys].replace(inp, list(range(len(inp))), inplace=True)
    return df

def dataframe_to_dictionary(df, key, continue_list=None):
    result = dict()
    for _key, _val in zip(df[key].keys(), df[key].values):
        if _key not in continue_list:
            result[_key] = _val
    return result

def make_plots(df, plot=0):
    corr_matrix = df.corr()

    home_team = dataframe_to_dictionary(corr_matrix, 'home_team_score', ['home_team_score', 'away_team_score'])
    away_team = dataframe_to_dictionary(corr_matrix, 'away_team_score', ['home_team_score', 'away_team_score'])

    home_team = dict(sorted(home_team.items(), key=lambda item: item[1]))
    away_team = dict(sorted(away_team.items(), key=lambda item: item[1]))
    if plot == 0:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        ax0.matshow(corr_matrix)
        ax0.set_title('Correlation Matrix', fontsize=8)
        ax0.set_xticklabels(list(range(len(corr_matrix.keys()))), fontsize=5)
        ax0.set_yticklabels(list(corr_matrix.keys()), fontsize=5)

        ax1.set_title('Home Team Variables Correlation', fontsize=8)
        ax1.barh(list(home_team.keys()), list(home_team.values()))
        ax1.set_yticklabels(list(home_team.keys()), fontsize=5)
        
        ax2.set_title('Away Team Variables Correlation', fontsize=8)
        ax2.barh(list(away_team.keys()), list(away_team.values()))
        ax2.set_yticklabels(list(away_team.keys()), fontsize=5)

        fig.tight_layout()

    elif plot == 1:
        plt.matshow(corr_matrix)
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.colorbar()

    elif plot == 2:
        plt.title('Home Team Variables Correlation')
        plt.barh(list(home_team.keys()), list(home_team.values()))
        plt.yticks(list(home_team.keys()))
    elif plot == 3:
        plt.title('Away Team Variables Correlation')
        plt.barh(list(away_team.keys()), list(away_team.values()))
        plt.yticks(list(away_team.keys()))

    plt.show()


# df = pd.read_csv('../../Data/international_matches.csv', delimiter=';')
df = pd.read_csv('/Users/ignasi/Documents/_Q7_/_APA_/Prac_FIFA_World_Cup_APA/Data/international_matches.csv', delimiter=';')

df = df[~df.isnull().any(axis=1)]

delete_keys = ['date', 'home_team', 'away_team', 'tournament', 'city', 'country', 'neutral_location', 'home_team_result']
df = delete_from_dataframe(df, delete_keys)

replace_keys = ['home_team_continent', 'away_team_continent', 'shoot_out']
replace_input = [['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'], ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'], ['No', 'Yes']]
df = replace_string_to_int_dataframe(df, replace_keys, replace_input)

make_plots(df)






# df_without_NaN_UnusedVars.to_csv('../../Data/international_matches_clean.csv')