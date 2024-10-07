import streamlit as st
import pandas as pd
import numpy as np

# Function to load previous tested creatives
def load_tested_creatives(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.DataFrame(columns=['creative_id'])

# Function to extract the creative identifier (game code, concept number, or raw recording, and version number)
def extract_creative_id(name, game_code):
    parts = name.split('_')
    for i in range(len(parts) - 2):
        if parts[i].upper() == game_code.upper() and (parts[i+1].startswith('C') or parts[i+1].startswith('R')) and parts[i+2].startswith('V'):
            return '_'.join([parts[i], parts[i+1], parts[i+2]])
    return '_'.join(parts[:3])

# Function to categorize creatives
def categorize_creative(row, average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold):
    if row['network_impressions'] < impressions_threshold:
        return 'Testing'
    elif row['cost'] >= cost_threshold * average_cost and row['IPM'] > ipm_threshold * average_ipm:
        return 'High Performance'
    elif row['IPM'] >= ipm_threshold * average_ipm:
        return 'Potential Creative'
    elif row['IPM'] < average_ipm:
        return 'Low Performance'
    else:
        return 'Testing'

# Function to calculate z-scores manually
def calculate_zscore(series):
    if series.std(ddof=0) == 0:
        return pd.Series([0]*len(series))
    else:
        return (series - series.mean()) / series.std(ddof=0)

# Sigmoid function
def sigmoid(x):
    return 100 / (1 + np.exp(-x))

# Streamlit app
st.title("Creative Performance Analyzer")

# File upload section
st.sidebar.header("Upload Files")
prev_file = st.sidebar.file_uploader("Upload Previous Tested Creatives CSV", type="csv")
new_file = st.sidebar.file_uploader("Upload New Report CSV", type="csv")

# Game code input
st.sidebar.header("Game Code")
game_code = st.sidebar.text_input("Enter the 3-letter game code (e.g., CRC)")

# Target ROAS D0 input
st.sidebar.header("Target ROAS D0")
target_roas_d0 = st.sidebar.number_input("Enter the Target ROAS D0", min_value=0.0, value=0.5, step=0.1)

# Threshold settings
st.sidebar.header("Threshold Settings")
impressions_threshold = st.sidebar.number_input("Impressions Threshold", min_value=1000, value=2000, step=100)
cost_threshold = st.sidebar.slider("Cost Threshold Multiplier", min_value=0.0, max_value=5.0, value=1.1, step=0.1)
ipm_threshold = st.sidebar.slider("IPM Threshold Multiplier", min_value=0.0, max_value=5.0, value=1.1, step=0.1)

# Weights settings
st.sidebar.header("Weights Settings")
st.sidebar.write("Adjust the weights for each metric used in the Lumina Score calculation.")

# Spend weight is fixed at +1 (positive to promote scalability)

# Input fields for other weights
weight_roas_diff = st.sidebar.number_input("Weight for ROAS Difference", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
weight_roas_mat_d3 = st.sidebar.number_input("Weight for ROAS Maturation D3", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
weight_ipm = st.sidebar.number_input("Weight for IPM", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# First-time run toggle
first_time_run = st.sidebar.checkbox("First-time run (No Previous Tested Creatives CSV)")

if new_file and game_code:
    # Step 1: Load previous and new data
    prev_data = load_tested_creatives(prev_file) if not first_time_run else pd.DataFrame(columns=['creative_id'])
    new_data = pd.read_csv(new_file)

    if 'creative_network' not in new_data.columns:
        st.error("The uploaded new report CSV does not contain a 'creative_network' column.")
    else:
        # Step 2: Filter out irrelevant creatives
        exclude_creative_ids = [
            'Search SearchPartners', 'Search GoogleSearch', 'Youtube YouTubeVideos',
            'Display', 'TTCC_0021_Ship Craft - Gaming App'
        ]
        new_data = new_data[~new_data['creative_network'].isin(exclude_creative_ids)]
        new_data = new_data[~new_data['creative_network'].str.startswith('TTCC')]

        # Step 3: Extract creative IDs using the game code
        new_data['creative_id'] = new_data.apply(lambda row: extract_creative_id(str(row['creative_network']), game_code), axis=1)
        new_data = new_data[new_data['creative_id'] != 'unknown']

        # Step 4: Ensure required columns exist before aggregation
        required_columns = [
            'network_impressions', 'cost', 'installs',
            'retention_rate_d1', 'retention_rate_d3', 'retention_rate_d7',
            'custom_cohorted_total_revenue_d0', 'custom_cohorted_total_revenue_d3', 'custom_cohorted_total_revenue_d7'
        ]
        missing_columns = [col for col in required_columns if col not in new_data.columns]

        if missing_columns:
            st.error(f"The uploaded CSV is missing the following columns: {', '.join(missing_columns)}")
        else:
            # Step 5: Aggregate data at the creative level
            aggregated_data = new_data.groupby('creative_id').agg({
                'network_impressions': 'sum',
                'cost': 'sum',
                'installs': 'sum',
                'retention_rate_d1': 'mean',
                'retention_rate_d3': 'mean',
                'retention_rate_d7': 'mean',
                'custom_cohorted_total_revenue_d0': 'sum',
                'custom_cohorted_total_revenue_d3': 'sum',
                'custom_cohorted_total_revenue_d7': 'sum'
            }).reset_index()

            # Step 6: Calculate LTV using custom_cohorted_total_revenue
            aggregated_data['LTV_D0'] = np.where(aggregated_data['installs'] != 0,
                                                 aggregated_data['custom_cohorted_total_revenue_d0'] / aggregated_data['installs'],
                                                 0)
            aggregated_data['LTV_D3'] = np.where(aggregated_data['installs'] != 0,
                                                 (aggregated_data['custom_cohorted_total_revenue_d0'] + aggregated_data['custom_cohorted_total_revenue_d3']) / aggregated_data['installs'],
                                                 0)
            aggregated_data['LTV_D7'] = np.where(aggregated_data['installs'] != 0,
                                                 (aggregated_data['custom_cohorted_total_revenue_d0'] + aggregated_data['custom_cohorted_total_revenue_d3'] + aggregated_data['custom_cohorted_total_revenue_d7']) / aggregated_data['installs'],
                                                 0)

            # Step 7: Calculate ROAS using LTV and CPI (CPI = cost / installs)
            aggregated_data['CPI'] = np.where(aggregated_data['installs'] != 0,
                                              aggregated_data['cost'] / aggregated_data['installs'],
                                              0)
            # Handle division by zero
            aggregated_data['ROAS_d0'] = np.where(aggregated_data['CPI'] != 0,
                                                  aggregated_data['LTV_D0'] / aggregated_data['CPI'],
                                                  0)
            aggregated_data['ROAS_d3'] = np.where(aggregated_data['CPI'] != 0,
                                                  aggregated_data['LTV_D3'] / aggregated_data['CPI'],
                                                  0)
            aggregated_data['ROAS_d7'] = np.where(aggregated_data['CPI'] != 0,
                                                  aggregated_data['LTV_D7'] / aggregated_data['CPI'],
                                                  0)

            # Step 8: Calculate IPM using network_impressions
            aggregated_data['IPM'] = (aggregated_data['installs'] / aggregated_data['network_impressions']) * 1000
            aggregated_data['IPM'].replace([float('inf'), -float('inf')], 0, inplace=True)
            aggregated_data['IPM'] = aggregated_data['IPM'].round(2)

            # Step 9: Exclude outliers in IPM
            Q1 = aggregated_data['IPM'].quantile(0.25)
            Q3 = aggregated_data['IPM'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            aggregated_data = aggregated_data[(aggregated_data['IPM'] >= lower_bound) & (aggregated_data['IPM'] <= upper_bound)]

            # Step 10: Calculate ROAS diff using calculated ROAS_d0
            aggregated_data['ROAS_diff'] = aggregated_data['ROAS_d0'] - target_roas_d0

            # Step 11: Calculate ROAS Mat D3 using calculated ROAS_d3 and ROAS_d0
            aggregated_data['ROAS_Mat_D3'] = np.where(
                aggregated_data['ROAS_d0'] != 0,
                aggregated_data['ROAS_d3'] / aggregated_data['ROAS_d0'],
                0
            )
            aggregated_data['ROAS_Mat_D3'].replace([float('inf'), -float('inf'), np.nan], 0, inplace=True)
            aggregated_data['ROAS_Mat_D3'] = aggregated_data['ROAS_Mat_D3'].round(2)

            # Step 12: Handle NaN values and check for zero variance before calculating z-scores
            for col in ['ROAS_Mat_D3', 'cost', 'ROAS_diff', 'IPM']:
                # Replace spaces and dots in column names
                col_name = col.replace(" ", "_").replace(".", "")
                if aggregated_data[col].var(ddof=0) == 0:
                    aggregated_data[f'z_{col_name}'] = 0
                else:
                    aggregated_data[col].fillna(aggregated_data[col].mean(), inplace=True)
                    aggregated_data[f'z_{col_name}'] = calculate_zscore(aggregated_data[col])

            # Step 13: Use weights on z-scores
            weights = {
                'z_cost': 1.0,  # Fixed weight to promote scalability
                'z_ROAS_diff': weight_roas_diff,
                'z_ROAS_Mat_D3': weight_roas_mat_d3,
                'z_IPM': weight_ipm
            }

            # Step 14: Calculate Lumina Score using weighted sum of z-scores and apply 15% penalty for installs < 5
            def calculate_lumina_score(row):
                weighted_sum = (
                    row['z_cost'] * weights['z_cost'] +
                    row['z_ROAS_diff'] * weights['z_ROAS_diff'] +
                    row['z_ROAS_Mat_D3'] * weights['z_ROAS_Mat_D3'] +
                    row['z_IPM'] * weights['z_IPM']
                )
                lumina_score = sigmoid(weighted_sum)
                if row['installs'] < 5:
                    lumina_score *= 0.85  # Penalize by 15%
                return lumina_score

            aggregated_data['Lumina_Score'] = aggregated_data.apply(calculate_lumina_score, axis=1)

            # Step 15: Categorize creatives
            average_ipm = aggregated_data['IPM'].mean()
            average_cost = aggregated_data['cost'].mean()
            aggregated_data['Category'] = aggregated_data.apply(lambda row: categorize_creative(row, average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold), axis=1)

            # Step 16: Output the overall creative performance data as CSV
            overall_output = aggregated_data.to_csv(index=False)
            st.download_button("Download Overall Creative Performance CSV", overall_output.encode('utf-8'), "Overall_Creative_Performance.csv")
