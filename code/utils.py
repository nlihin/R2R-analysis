import numpy as np
import pandas as pd
import ast
from itertools import combinations
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

def get_files(url):
    """
    get the files from github
    """
    response = requests.get(url)
    data = response.json()
    csv_files = [file['download_url'] for file in data if file['name'].endswith('.csv')]
    return csv_files

# Function to parse list_rank column
def parse_list_rank(row):
    return ast.literal_eval(row)

# Function to calculate lag values directly
def calculate_lag(group1, group2):
    return abs(group1 - group2)

all_combined_dfs = []

# function to process each CSV file
def process_files(csv_file_list):
    """
    process each file
    """
    for session_id, csv_url in enumerate(csv_file_list, start=1):
        print("running ", session_id)
        # Load the CSV file
        data = pd.read_csv(csv_url)

    # #fix 12.2.26: before it read files from github in this order: 10,1,2,3...
    # for session_id in csv_file_list:
    #     print("running ", session_id)
    #     # Load the CSV file
    #     csv_url = f".../{session_id}_rank.csv"
    #     data = pd.read_csv(csv_url)
        
        # Parse the list_rank column
        data['parsed_rank'] = data['list_rank'].apply(parse_list_rank)
    
        # Extract participant ID and their rankings
        parsed_data = []
    
        for _, row in data.iterrows():
            username = row['username']
            ranks = row['parsed_rank']
            
            for group, score in ranks:
                parsed_data.append({
                    'username': username,
                    'group': group,
                    'score': score
                })
    
        # Create a DataFrame from the parsed data
        parsed_df = pd.DataFrame(parsed_data)

        print("usernames: ", len(parsed_df["username"].unique()))
    
        # Find pairs with the same score and calculate lag
        lag_data = []
    
        for username, group_df in parsed_df.groupby('username'):
            same_score_groups = group_df.groupby('score')
            
            for score, score_df in same_score_groups:
                # Get all combinations of pairs
                for row1, row2 in combinations(score_df.itertuples(index=False), 2):
                    lag_value = calculate_lag(row1.group, row2.group)
                    
                    # Determine primacy or recency
                    if row1.group > row2.group:
                        bias = 'recency'
                    else:
                        bias = 'primacy'
                    
                    lag_data.append({
                        'username': username,
                        'group1': row1.group,
                        'group2': row2.group,
                        'score': score,
                        'lag': lag_value,
                        'bias': bias
                    })
    
        # Create a DataFrame for the lag data
        lag_df = pd.DataFrame(lag_data)
    
        # Split the bias column into 'primacy' and 'recency' columns
        lag_df['primacy'] = lag_df['bias'].apply(lambda x: 1 if x == 'primacy' else 0)
        lag_df['recency'] = lag_df['bias'].apply(lambda x: 1 if x == 'recency' else 0)
    
        # Final table structure
        final_df = lag_df[['username', 'score', 'lag', 'primacy', 'recency']].copy()
        final_df.loc[:, 'session_id'] = session_id  # Add the session_id column
    
        # Group by 'username', 'score', 'lag', and 'session_id' and aggregate 'primacy' and 'recency' columns
        combined_df = final_df.groupby(['username', 'score', 'lag', 'session_id']).agg({
            'primacy': 'sum',
            'recency': 'sum'
        }).reset_index()

        all_combined_dfs.append(combined_df) 
    
    final_combined  = pd.concat(all_combined_dfs, ignore_index=False)   
    return final_combined

def normalize_PR(df, path_name_for_save):
    df["total_comparisons"] =df["primacy"] + df["recency"]
    df["norm_primacy"] = df["primacy"]/df["total_comparisons"]
    df["norm_recency"] = df["recency"]/df["total_comparisons"]    
    return df

def viz_barplot(case_study_number, norm_df, scores, path_name_for_save,cpal):      
    # Loop through each score, create the melted DataFrame, and save the plot
    for score in scores:
        # Melt the DataFrame for the given score
        norm_df_score = norm_df[norm_df['score'] == score].melt(
            id_vars='lag', 
            value_vars=['norm_primacy', 'norm_recency'], 
            var_name='metric', 
            value_name='value'
        )
        
        # Rename the variables in the 'metric' column
        norm_df_score['metric'] = norm_df_score['metric'].replace({
            'norm_primacy': 'primacy', 
            'norm_recency': 'recency'
        })
        
        # Create the figure and save it
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=norm_df_score, x='lag', y='value', hue='metric', ax=ax, palette=cpal)
        #ax.set_title(f'Score = {score}')
        ax.legend(title='')  # Remove the title from the legend
        fig.savefig(path_name_for_save + "/" + f"{case_study_number}_score_{score}.png")  # Save figure
        plt.show()
        plt.close(fig)  # Close the figure to free up memory


def viz_lineplots(norm_df):

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    sns.lineplot(
        norm_df, x='lag', y='norm_primacy', hue='score', 
        style="score",
        markers=True, dashes=False, 
        ax=axes[0]
    )
    
    sns.lineplot(
        norm_df, x='lag', y='norm_recency', hue='score', 
        style="score",
        markers=True, dashes=False, 
        ax=axes[1]
    )
    
    # Set the same y-axis limits for both axes
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    
    # Titles
    axes[0].set_title('Primacy')
    axes[1].set_title('Recency')
    
    plt.show()

def replace_session_ids(df, my_date_session_letters):
    unique_sessions = sorted(df["session_id"].unique())
    if len(unique_sessions) != len(my_date_session_letters):
        raise ValueError(
            f"Length mismatch: {len(unique_sessions)} unique session_ids "
            f"vs {len(my_date_session_letters)} items in list."
        )
    mapping = dict(zip(unique_sessions, my_date_session_letters))
    new_df = df.copy()
    new_df["session_id"] = new_df["session_id"].replace(mapping)
    return new_df
