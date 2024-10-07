import streamlit as st
import pandas as pd
import numpy as np

# Sigmoid function for Lumina Score calculation
def sigmoid(x, slope=2):
    return 1 / (1 + np.exp(-slope * x))

# Function to calculate robust z-scores
def calculate_robust_zscore(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    return (series - median) / (mad if mad else 1)

# Min-Max Scaling function
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

# Streamlit sidebar for user input
st.sidebar.header("Select Test Type")
unity_test_active = st.sidebar.checkbox("Unity Test Active (Include Unity Ads Data)")

# File upload section
new_file = st.sidebar.file_uploader("Upload New Report CSV", type="csv")

# Target ROAS D0 and CPI inputs
target_roas_d0 = st.sidebar.number_input("Enter Target ROAS D0", min_value=0.0, value=0.5, step=0.1)
target_cpi = st.sidebar.number_input("Enter Target CPI", min_value=0.0, value=2.0, step=0.1)

if new_file:
    # Load new data
    new_data = pd.read_csv(new_file)
    
    # Step 1: Aggregate Facebook data for CPI and ROAS calculations
    facebook_data = new_data[new_data['Channel'] == 'Facebook']

    facebook_aggregated = facebook_data.groupby('creative_id').agg({
        'cost': 'sum',
        'installs': 'sum',
        'custom_cohorted_total_revenue_d0': 'sum',
        'custom_cohorted_total_revenue_d3': 'sum',
        'custom_cohorted_total_revenue_d7': 'sum',
        'impressions': 'sum'  # Impressions for IPM
    }).reset_index()

    # Calculate Facebook CPI
    facebook_aggregated['CPI'] = np.where(facebook_aggregated['installs'] != 0, 
                                          facebook_aggregated['cost'] / facebook_aggregated['installs'], 
                                          0)

    # Facebook LTV calculations for D0, D3, D7
    facebook_aggregated['LTV_D0'] = np.where(facebook_aggregated['installs'] != 0, 
                                             facebook_aggregated['custom_cohorted_total_revenue_d0'] / facebook_aggregated['installs'], 
                                             0)
    facebook_aggregated['LTV_D3'] = np.where(facebook_aggregated['installs'] != 0, 
                                             (facebook_aggregated['custom_cohorted_total_revenue_d0'] + facebook_aggregated['custom_cohorted_total_revenue_d3']) / facebook_aggregated['installs'], 
                                             0)
    facebook_aggregated['LTV_D7'] = np.where(facebook_aggregated['installs'] != 0, 
                                             (facebook_aggregated['custom_cohorted_total_revenue_d0'] + facebook_aggregated['custom_cohorted_total_revenue_d3'] + facebook_aggregated['custom_cohorted_total_revenue_d7']) / facebook_aggregated['installs'], 
                                             0)

    # Calculate Facebook ROAS (D0, D3, D7)
    facebook_aggregated['ROAS_d0'] = np.where(facebook_aggregated['CPI'] != 0, 
                                              facebook_aggregated['LTV_D0'] / facebook_aggregated['CPI'], 
                                              0)
    facebook_aggregated['ROAS_d3'] = np.where(facebook_aggregated['CPI'] != 0, 
                                              facebook_aggregated['LTV_D3'] / facebook_aggregated['CPI'], 
                                              0)
    facebook_aggregated['ROAS_d7'] = np.where(facebook_aggregated['CPI'] != 0, 
                                              facebook_aggregated['LTV_D7'] / facebook_aggregated['CPI'], 
                                              0)

    # ROAS Maturation calculation
    facebook_aggregated['ROAS_Mat_D3'] = np.where(facebook_aggregated['ROAS_d0'] != 0, 
                                                  facebook_aggregated['ROAS_d3'] / facebook_aggregated['ROAS_d0'], 
                                                  0)

    # Step 2: Aggregate Unity Ads data for IPM if Unity Test is active
    if unity_test_active:
        unity_data = new_data[new_data['Channel'] == 'Unity Ads']
        
        # Concatenate Facebook and Unity Ads data for combined IPM
        combined_data = pd.concat([facebook_data[['creative_id', 'impressions', 'installs']], 
                                   unity_data[['creative_id', 'impressions', 'installs']]])

        # Aggregate combined data for IPM
        combined_aggregated = combined_data.groupby('creative_id').agg({
            'impressions': 'sum',
            'installs': 'sum'
        }).reset_index()

        # Calculate combined IPM
        combined_aggregated['IPM'] = np.where(combined_aggregated['impressions'] != 0, 
                                              (combined_aggregated['installs'] / combined_aggregated['impressions']) * 1000, 
                                              0)
    else:
        # Use only Facebook data for IPM if Unity Test is not active
        combined_aggregated = facebook_aggregated[['creative_id', 'impressions', 'installs']].copy()
        combined_aggregated['IPM'] = np.where(combined_aggregated['impressions'] != 0, 
                                              (combined_aggregated['installs'] / combined_aggregated['impressions']) * 1000, 
                                              0)

    # Step 3: Merge Facebook CPI/ROAS data with combined IPM data
    final_data = pd.merge(facebook_aggregated[['creative_id', 'CPI', 'ROAS_d0', 'ROAS_d3', 'ROAS_d7', 'ROAS_Mat_D3']], 
                          combined_aggregated[['creative_id', 'IPM']], 
                          on='creative_id', 
                          how='left')

    # Handle cases where a creative was not tested on Unity Ads (IPM NaN)
    final_data['IPM'].fillna(final_data['IPM'], inplace=True)

    # Step 4: Calculate the final score (Lumina Score) based on the merged data
    # Calculate CPI_diff and ROAS_diff based on Facebook data
    final_data['ROAS_diff'] = final_data['ROAS_d0'] - target_roas_d0
    final_data['CPI_diff'] = target_cpi - final_data['CPI']

    # Apply robust z-scores to each component
    final_data['z_ROAS_diff'] = calculate_robust_zscore(final_data['ROAS_diff'])
    final_data['z_CPI_diff'] = calculate_robust_zscore(final_data['CPI_diff'])
    final_data['z_IPM'] = calculate_robust_zscore(final_data['IPM'])

    # Min-Max Scaling for normalization
    final_data['scaled_ROAS_diff'] = min_max_scale(final_data['z_ROAS_diff'])
    final_data['scaled_CPI_diff'] = min_max_scale(final_data['z_CPI_diff'])
    final_data['scaled_IPM'] = min_max_scale(final_data['z_IPM'])

    # Define weights for Lumina score calculation
    weights = {
        'scaled_CPI_diff': 1.2,
        'scaled_ROAS_diff': 1.5,
        'scaled_IPM': 1.0
    }

    # Lumina Score calculation using the weighted sum of scaled z-scores
    def calculate_lumina_score(row):
        weighted_sum = (
            row['scaled_CPI_diff'] * weights['scaled_CPI_diff'] +
            row['scaled_ROAS_diff'] * weights['scaled_ROAS_diff'] +
            row['scaled_IPM'] * weights['scaled_IPM']
        )
        return sigmoid(weighted_sum)

    # Calculate the final Lumina score
    final_data['Lumina_Score'] = final_data.apply(calculate_lumina_score, axis=1)

    # Step 5: Display final report with download button in Streamlit
    final_output = final_data.to_csv(index=False)
    st.download_button("Download Final Report", final_output.encode('utf-8'), "Final_Report.csv")

