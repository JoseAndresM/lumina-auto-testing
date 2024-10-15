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

# Function to aggregate and process data based on the selected data source
def autotesting_aggregate(new_data, target_roas_d0, target_cpi, data_source):
    # Adjusted column names specific to Auto-Testing CNCP
    new_data.rename(columns={
        'ecpi_all': 'CPI',
        'network_impressions': 'impressions',
        'installs': 'installs'
    }, inplace=True)

    # Ensure required columns exist
    required_columns = ['impressions', 'cost', 'installs', 'channel',
                        'custom_cohorted_total_revenue_d0', 'custom_cohorted_total_revenue_d3',
                        'custom_cohorted_total_revenue_d7']
    missing_columns = [col for col in required_columns if col not in new_data.columns]
    
    if missing_columns:
        st.error(f"Missing columns in the uploaded CSV: {', '.join(missing_columns)}")
        return pd.DataFrame()

    # Extract data based on channel
    facebook_data = new_data[new_data['channel'] == 'Facebook']
    unity_data = new_data[new_data['channel'] == 'Unity']

    # Initialize an empty DataFrame for aggregated data
    aggregated_data = pd.DataFrame()

    # Calculate CPI and ROAS metrics using Facebook data
    if data_source in ['All', 'Facebook'] and not facebook_data.empty:
        fb_aggregated = facebook_data.groupby('creative_id').agg({
            'impressions': 'sum',
            'cost': 'sum',
            'installs': 'sum',
            'custom_cohorted_total_revenue_d0': 'sum',
            'custom_cohorted_total_revenue_d3': 'sum',
            'custom_cohorted_total_revenue_d7': 'sum'
        }).reset_index()

        # CPI = Total Cost / Total Installs
        fb_aggregated['CPI'] = np.where(fb_aggregated['installs'] != 0, 
                                        fb_aggregated['cost'] / fb_aggregated['installs'], 
                                        0)

        # LTV calculations using custom_cohorted_total_revenue
        fb_aggregated['LTV_D0'] = np.where(fb_aggregated['installs'] != 0, 
                                           fb_aggregated['custom_cohorted_total_revenue_d0'] / fb_aggregated['installs'], 
                                           0)
        fb_aggregated['LTV_D3'] = np.where(fb_aggregated['installs'] != 0, 
                                           (fb_aggregated['custom_cohorted_total_revenue_d0'] + fb_aggregated['custom_cohorted_total_revenue_d3']) / fb_aggregated['installs'], 
                                           0)
        fb_aggregated['LTV_D7'] = np.where(fb_aggregated['installs'] != 0, 
                                           (fb_aggregated['custom_cohorted_total_revenue_d0'] + fb_aggregated['custom_cohorted_total_revenue_d3'] + fb_aggregated['custom_cohorted_total_revenue_d7']) / fb_aggregated['installs'], 
                                           0)

        # ROAS calculation using LTV and CPI
        fb_aggregated['ROAS_d0'] = np.where(fb_aggregated['CPI'] != 0, 
                                            fb_aggregated['LTV_D0'] / fb_aggregated['CPI'], 
                                            0)
        fb_aggregated['ROAS_d3'] = np.where(fb_aggregated['CPI'] != 0, 
                                            fb_aggregated['LTV_D3'] / fb_aggregated['CPI'], 
                                            0)
        fb_aggregated['ROAS_d7'] = np.where(fb_aggregated['CPI'] != 0, 
                                            fb_aggregated['LTV_D7'] / fb_aggregated['CPI'], 
                                            0)

        # Calculate ROAS_diff and CPI_diff
        fb_aggregated['ROAS_diff'] = fb_aggregated['ROAS_d0'] - target_roas_d0
        fb_aggregated['CPI_diff'] = target_cpi - fb_aggregated['CPI']

        # Handle Inf and NaN values
        fb_aggregated.replace([np.inf, -np.inf], np.nan, inplace=True)
        fb_aggregated.fillna(0, inplace=True)

        # Calculate ROAS_Mat_D3
        fb_aggregated['ROAS_Mat_D3'] = np.where(
            fb_aggregated['ROAS_d0'] != 0,
            fb_aggregated['ROAS_d3'] / fb_aggregated['ROAS_d0'],
            0
        )

        # Keep only necessary columns
        fb_metrics = fb_aggregated[['creative_id', 'CPI', 'CPI_diff', 'ROAS_d0', 'ROAS_diff', 'ROAS_Mat_D3', 'cost']]
    else:
        # If no Facebook data, create an empty DataFrame for metrics
        fb_metrics = pd.DataFrame(columns=['creative_id', 'CPI', 'CPI_diff', 'ROAS_d0', 'ROAS_diff', 'ROAS_Mat_D3', 'cost'])

    # Calculate IPM using data from both Facebook and Unity (or selected data source)
    if data_source == 'All':
        # Combine Facebook and Unity data for IPM calculation
        combined_data = pd.concat([facebook_data, unity_data], ignore_index=True)
    elif data_source == 'Facebook':
        combined_data = facebook_data
    elif data_source == 'Unity':
        combined_data = unity_data
    else:
        combined_data = pd.DataFrame()

    if not combined_data.empty:
        ipm_aggregated = combined_data.groupby('creative_id').agg({
            'impressions': 'sum',
            'installs': 'sum'
        }).reset_index()

        # Calculate IPM
        ipm_aggregated['IPM'] = (ipm_aggregated['installs'] / ipm_aggregated['impressions']) * 1000
        ipm_aggregated['IPM'].replace([np.inf, -np.inf], 0, inplace=True)
        ipm_aggregated['IPM'] = ipm_aggregated['IPM'].round(2)
    else:
        ipm_aggregated = pd.DataFrame(columns=['creative_id', 'IPM'])

    # Merge Facebook metrics and IPM data
    aggregated_data = pd.merge(fb_metrics, ipm_aggregated, on='creative_id', how='outer')

    # Fill NaN values for missing metrics
    aggregated_data.fillna({'CPI': 0, 'CPI_diff': 0, 'ROAS_d0': 0, 'ROAS_diff': 0, 'ROAS_Mat_D3': 0, 'cost': 0, 'IPM': 0}, inplace=True)

    # Handle cases where only Unity data is available
    if data_source == 'Unity' and fb_metrics.empty:
        # For Unity-only data, set CPI and ROAS metrics to zero or NaN
        aggregated_data['CPI'] = np.nan
        aggregated_data['CPI_diff'] = np.nan
        aggregated_data['ROAS_d0'] = np.nan
        aggregated_data['ROAS_diff'] = np.nan
        aggregated_data['ROAS_Mat_D3'] = np.nan
        aggregated_data['cost'] = combined_data.groupby('creative_id')['cost'].sum().reset_index(drop=True)
        aggregated_data['installs'] = combined_data.groupby('creative_id')['installs'].sum().reset_index(drop=True)

    # Apply robust z-scores, ensuring no NaN or Inf values
    aggregated_data['z_ROAS_Mat_D3'] = calculate_robust_zscore(aggregated_data['ROAS_Mat_D3'])
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
        'scaled_cost': 1.0,
        'scaled_ROAS_diff': 1.5,
        'scaled_ROAS_Mat_D3': 1.0,
        'scaled_IPM': 1.0,
        'scaled_CPI_diff': 1.3
    }

    # Adjust weights if only Unity data is selected (since ROAS and CPI metrics are not available)
    if data_source == 'Unity':
        weights = {
            'scaled_cost': 0.0,
            'scaled_ROAS_diff': 0.0,
            'scaled_ROAS_Mat_D3': 0.0,
            'scaled_IPM': 1.0,
            'scaled_CPI_diff': 0.0
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

    # Filter out creatives with no installs
    valid_creatives = aggregated_data[aggregated_data['installs'] >= 1]
    valid_creatives['Lumina_Score'] = valid_creatives.apply(calculate_lumina_score, axis=1)

    # Normalize the Lumina score to 0-100
    min_score = valid_creatives['Lumina_Score'].min()
    max_score = valid_creatives['Lumina_Score'].max()
    valid_creatives['Lumina_Score'] = (valid_creatives['Lumina_Score'] - min_score) / (max_score - min_score + 1e-8) * 100
    return valid_creatives

# Streamlit app
st.title("Lumina - Auto-Testing Analyzer with Data Source Selection")

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
cost_threshold = st.sidebar.slider("Cost Threshold Multiplier", min_value=0.0, max_value=5.0, value=1.1, step=0.1)
ipm_threshold = st.sidebar.slider("IPM Threshold Multiplier", min_value=0.0, max_value=5.0, value=1.1, step=0.1)

# Data Source Selection
st.sidebar.header("Data Source Selection")
data_source = st.sidebar.selectbox("Select Data Source", options=['All', 'Facebook', 'Unity'])

# First-time run toggle
first_time_run = st.sidebar.checkbox("First-time run (No Previous Tested Creatives CSV)")

if new_file and game_code:
    # Step 1: Load previous and new data
    prev_data = load_tested_creatives(prev_file) if not first_time_run else pd.DataFrame(columns=['creative_id'])
    new_data = pd.read_csv(new_file)
    
    # Ensure the 'channel' column exists
    if 'channel' not in new_data.columns:
        st.error("The uploaded new report CSV does not contain a 'channel' column.")
        st.stop()
    
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
        new_data['creative_id'] = new_data.apply(lambda row: extract_creative_id(str(row['creative_network']), game_code), axis=1)
        new_data = new_data[new_data['creative_id'] != 'unknown']

        # Step 4: Apply Auto-Testing aggregation logic
        aggregated_data = autotesting_aggregate(new_data, target_roas_d0, target_cpi, data_source)

        if not aggregated_data.empty:
            # Step 5: Categorize creatives
            average_ipm = aggregated_data['IPM'].mean()
            average_cost = aggregated_data['cost'].mean()
            aggregated_data['Category'] = aggregated_data.apply(lambda row: categorize_creative(row, average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold), axis=1)

            # Step 6: Output the overall creative performance data as CSV
            overall_output = aggregated_data.to_csv(index=False)
            st.download_button("Download Creative Level Performance CSV", overall_output.encode('utf-8'), "Creative_Performance.csv")

            # Optional: Display aggregated data in the app
            st.header("Aggregated Creative Performance Data")
            st.write(aggregated_data)
