import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def load_clinical_data(filepath):
    """
    Loads clinical (voice acoustic) data, separates labels, and normalizes features.
    
    Args:
        filepath (str): Path to 'parkinsons.data'.
        
    Returns:
        dict: A dictionary containing 'HC' and 'PD' keys, each with their PyTorch feature tensors.
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
        
    df = pd.read_csv(filepath)
    
    # The 'status' column is 1 for PD and 0 for HC
    # The 'name' column is dropped as it's a string identifier
    labels = df['status'].values
    features_df = df.drop(columns=['name', 'status'])
    features = features_df.values
    
    # Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split into PD and HC groups
    hc_mask = (labels == 0)
    pd_mask = (labels == 1)
    
    hc_features = features_scaled[hc_mask]
    pd_features = features_scaled[pd_mask]
    
    data = {
        'HC': torch.tensor(hc_features, dtype=torch.float32),
        'PD': torch.tensor(pd_features, dtype=torch.float32)
    }
    
    return data

if __name__ == "__main__":
    # Simple test
    test_filepath = os.path.join('..', '..', 'data', 'raw', 'parkinsons.data')
    data = load_clinical_data(test_filepath)
    print("Clinical Preprocessing Complete.")
    if data:
        print(f"HC Tensor Shape: {data['HC'].shape}")
        print(f"PD Tensor Shape: {data['PD'].shape}")
