import streamlit as st
import pandas as pd
import numpy as np

# Function to load previous tested creatives
def load_tested_creatives(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.DataFrame(columns=['creative_id', 'Facebook', 'Google Ads', 'Google Organic Search', 'Organic', 'Snapchat', 'TikTok for Business', 'Untrusted Devices'])

# Function to extract the creative identifier (game code, concept number, or raw recording, and version number)
def extract_creative_id(name, game_code):
    parts = name.split('_')
    for i in range(len(parts) - 2):
        if parts[i] == game_code and (parts[i+1].startswith('C') or parts[i+1].startswith('R')) and parts[i+2].startswith('V'):
            return '_'.join([game_code, parts[i+1], parts[i+2]])
    return '_'.join(parts[:3])

# Function to categorize creatives
def categorize_creative(row, average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold):
    if row['impressions'] < impressions_threshold:
        return 'Testing'
    elif row['cost'] >= cost_threshold * average_cost and row['IPM'] > ipm_threshold * average_ipm:
        return 'High Performance'
    elif row['IPM'] >= ipm_threshold * average_ipm:
        return 'Potential Creative'
    elif row['IPM'] < average_ipm:
        return 'Low Performance'
    else:
        return 'Testing'

# Function to calculate robust z-scores
def calculate_robust_zscore(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    return (series - median) / (mad if mad else 1)

def min_max_scale(series, epsilon=1e-8):
    return (series - series.min()) / (series.max() - series.min() + epsilon)


# Sigmoid function
def sigmoid(x, slope=2):
    return 1 / (1 + np.exp(-slope * x))

# Function to aggregate and process data for Auto-Testing CNCP
def autotesting_aggregate(new_data, target_roas_d0, target_cpi):
    # Adjusted column names specific to Auto-Testing CNCP
    new_data.rename(columns={
        'ecpi_all': 'CPI',
        'network_impressions': 'impressions',
        'installs': 'installs'
    }, inplace=True)

    # Check if all required columns are present
    required_columns = ['impressions', 'cost', 'installs', 'roas_d0', 'roas_d3', 'roas_d7', 'retention_rate_d1',
                        'retention_rate_d3', 'retention_rate_d7', 'lifetime_value_d0', 'lifetime_value_d3', 
                        'lifetime_value_d7', 'CPI']
    missing_columns = [col for col in required_columns if col not in new_data.columns]
    
    if missing_columns:
        st.error(f"Missing columns in the uploaded CSV: {', '.join(missing_columns)}")
        return pd.DataFrame()  # Return an empty DataFrame to avoid further errors

    aggregated_data = new_data.groupby('creative_id').agg({
        'impressions': 'sum',
        'cost': 'sum',
        'installs': 'sum',
        'roas_d0': 'mean',
        'roas_d3': 'mean',
        'roas_d7': 'mean',
        'retention_rate_d1': 'mean',
        'retention_rate_d3': 'mean',
        'retention_rate_d7': 'mean',
        'lifetime_value_d0': 'mean',
        'lifetime_value_d3': 'mean',
        'lifetime_value_d7': 'mean',
        'custom_cohorted_total_revenue_d0': 'sum',
        'custom_cohorted_total_revenue_d3': 'sum',
        'custom_cohorted_total_revenue_d7': 'sum',
        'CPI': 'mean'
    }).reset_index()
    
         # CPI = Total Aggregated Cost / Total Aggregated Installs
    aggregated_data['CPI'] = np.where(aggregated_data['installs'] != 0, 
                                      aggregated_data['cost'] / aggregated_data['installs'], 
                                      0)
    
    # LTV calculations for D0, D3, D7
    aggregated_data['LTV_D0'] = np.where(aggregated_data['installs'] != 0, 
                                         aggregated_data['custom_cohorted_total_revenue_d0'] / aggregated_data['installs'], 
                                         0)
    aggregated_data['LTV_D3'] = np.where(aggregated_data['installs'] != 0, 
                                         aggregated_data['custom_cohorted_total_revenue_d3'] / aggregated_data['installs'], 
                                         0)
    aggregated_data['LTV_D7'] = np.where(aggregated_data['installs'] != 0, 
                                         aggregated_data['custom_cohorted_total_revenue_d7'] / aggregated_data['installs'], 
                                         0)
    
    # ROAS calculation using LTV and CPI
    aggregated_data['ROAS_d0'] = np.where(aggregated_data['CPI'] != 0, 
                                          aggregated_data['LTV_D0'] / aggregated_data['CPI'], 
                                          0)
    aggregated_data['ROAS_d3'] = np.where(aggregated_data['CPI'] != 0, 
                                          aggregated_data['LTV_D3'] / aggregated_data['CPI'], 
                                          0)
    aggregated_data['ROAS_d7'] = np.where(aggregated_data['CPI'] != 0, 
                                          aggregated_data['LTV_D7'] / aggregated_data['CPI'], 
                                          0)
    
    # Handle Inf and NaN values in ROAS calculations
    aggregated_data['ROAS_d0'].replace([np.inf, -np.inf], np.nan, inplace=True)
    aggregated_data['ROAS_d3'].replace([np.inf, -np.inf], np.nan, inplace=True)
    aggregated_data['ROAS_d7'].replace([np.inf, -np.inf], np.nan, inplace=True)
    aggregated_data['ROAS_d0'].fillna(0, inplace=True)
    aggregated_data['ROAS_d3'].fillna(0, inplace=True)
    aggregated_data['ROAS_d7'].fillna(0, inplace=True)
    
    # Calculate ROAS Mat. D3 with error handling for division
    aggregated_data['ROAS Mat. D3'] = np.where(
        aggregated_data['ROAS_d0'] != 0,
        aggregated_data['ROAS_d3'] / aggregated_data['ROAS_d0'],
        0  # Set to 0 when ROAS_d0 is zero to avoid division by zero
    )
    aggregated_data['ROAS Mat. D3'].replace([np.inf, -np.inf], np.nan, inplace=True)
    aggregated_data['ROAS Mat. D3'].fillna(0, inplace=True)
    
    # Calculate ROAS_diff safely
    aggregated_data['ROAS_diff'] = aggregated_data['ROAS_d0'] - target_roas_d0
    aggregated_data['ROAS_diff'].replace([np.inf, -np.inf], np.nan, inplace=True)
    aggregated_data['ROAS_diff'].fillna(0, inplace=True)
    
    # Calculate CPI_diff safely
    aggregated_data['CPI_diff'] = target_cpi - aggregated_data['CPI']
    aggregated_data['CPI_diff'].replace([np.inf, -np.inf], np.nan, inplace=True)
    aggregated_data['CPI_diff'].fillna(0, inplace=True)
    
    # Calculate IPM
    aggregated_data['IPM'] = (aggregated_data['installs'] / aggregated_data['impressions']) * 1000
    aggregated_data['IPM'].replace([float('inf'), -float('inf')], 0, inplace=True)
    aggregated_data['IPM'] = aggregated_data['IPM'].round(2)
    
    # Apply robust z-scores, ensuring no NaN or Inf values
    aggregated_data['z_ROAS_Mat_D3'] = calculate_robust_zscore(aggregated_data['ROAS Mat. D3'])
    aggregated_data['z_cost'] = calculate_robust_zscore(aggregated_data['cost'])
    aggregated_data['z_ROAS_diff'] = calculate_robust_zscore(aggregated_data['ROAS_diff'])
    aggregated_data['z_IPM'] = calculate_robust_zscore(aggregated_data['IPM'])
    aggregated_data['z_CPI_diff'] = calculate_robust_zscore(aggregated_data['CPI_diff'])
    
    # Min-Max Scaling on z-scores, handling potential zero division
    aggregated_data['scaled_ROAS_Mat_D3'] = min_max_scale(aggregated_data['z_ROAS_Mat_D3'])
    aggregated_data['scaled_cost'] = min_max_scale(aggregated_data['z_cost'])
    aggregated_data['scaled_ROAS_diff'] = min_max_scale(aggregated_data['z_ROAS_diff'])
    aggregated_data['scaled_IPM'] = min_max_scale(aggregated_data['z_IPM'])
    aggregated_data['scaled_CPI_diff'] = min_max_scale(aggregated_data['z_CPI_diff'])



    # Define weights for each component
    weights = {
        'scaled_cost': 0.7,
        'scaled_ROAS_diff': 1.5,
        'scaled_ROAS_Mat_D3': 1.0,
        'scaled_IPM': 1.0,
        'scaled_CPI_diff': 1.2
    }

    # Lumina Score calculation using weighted sum and sigmoid transformation
    def calculate_lumina_score(row):
        weighted_sum = (
            row['scaled_cost'] * weights['scaled_cost'] + 
            row['scaled_ROAS_diff'] * weights['scaled_ROAS_diff'] + 
            row['scaled_ROAS_Mat_D3'] * weights['scaled_ROAS_Mat_D3'] + 
            row['scaled_IPM'] * weights['scaled_IPM'] + 
            row['scaled_CPI_diff'] * weights['scaled_CPI_diff']
        )
        lumina_score = sigmoid(weighted_sum)
        # Penalize if installs are low or unreliable
        if row['installs'] < 5 or row['IPM'] < 0.5:
            return lumina_score * 0.8  # Penalize by 20%
        return lumina_score

    valid_creatives = aggregated_data[aggregated_data['installs'] >= 1]
    valid_creatives['Lumina_Score'] = valid_creatives.apply(calculate_lumina_score, axis=1)

    # Normalize the Lumina score to 0-100
    min_score = valid_creatives['Lumina_Score'].min()
    max_score = valid_creatives['Lumina_Score'].max()
    valid_creatives['Lumina_Score'] = (valid_creatives['Lumina_Score'] - min_score) / (max_score - min_score) * 100
    return valid_creatives

# Streamlit app
st.title("Lumina - Auto-Testing Analyzer")

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

# Target CPI input
st.sidebar.header("Target CPI")
target_cpi = st.sidebar.number_input("Enter the Target CPI", min_value=0.0, value=2.0, step=0.1)

# Threshold settings
st.sidebar.header("Threshold Settings")
impressions_threshold = st.sidebar.number_input("Impressions Threshold", min_value=1000, value=2000, step=100)
cost_threshold = st.sidebar.slider("Cost Threshold Multiplier", min_value=0.0, max_value=2.0, value=1.1, step=0.1)
ipm_threshold = st.sidebar.slider("IPM Threshold Multiplier", min_value=0.0, max_value=2.0, value=1.1, step=0.1)

# First-time run toggle
first_time_run = st.sidebar.checkbox("First-time run (No Previous Tested Creatives CSV)")

if new_file and game_code:
    # Step 1: Load previous and new data
    prev_data = load_tested_creatives(prev_file) if not first_time_run else pd.DataFrame(columns=['creative_id', 'Facebook', 'Google Ads', 'Google Organic Search', 'Organic', 'Snapchat', 'TikTok for Business', 'Untrusted Devices'])
    new_data = pd.read_csv(new_file)
    
    if 'creative_network' not in new_data.columns:
        st.error("The uploaded new report CSV does not contain a 'creative_network' column.")
    else:
        # Step 2: Filter out irrelevant creatives
        exclude_creative_ids = [
            'Search SearchPartners', 'Search GoogleSearch', 'Youtube YouTubeVideos',
            'Display', 'TTCC'
        ]
        new_data = new_data[~new_data['creative_network'].isin(exclude_creative_ids)]
        new_data = new_data[~new_data['creative_network'].str.startswith('TTCC')]

        # Step 3: Extract creative IDs
        new_data['creative_id'] = new_data.apply(lambda row: extract_creative_id(row['creative_network'], game_code), axis=1)
        new_data = new_data[new_data['creative_id'] != 'unknown']

        # Step 4: Apply Auto-Testing aggregation logic
        aggregated_data = autotesting_aggregate(new_data, target_roas_d0, target_cpi)

        if not aggregated_data.empty:
            # Step 5: Categorize creatives
            average_ipm = aggregated_data['IPM'].mean()
            average_cost = aggregated_data['cost'].mean()
            aggregated_data['Category'] = aggregated_data.apply(lambda row: categorize_creative(row, average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold), axis=1)
            
            # Step 6: Output the overall creative performance data as CSV
            overall_output = aggregated_data.to_csv(index=False)
            st.download_button("Download Creative Level Performance CSV", overall_output.encode('utf-8'), "Creative_Performance.csv")
