import os
import torch
import numpy as np

def load_gait_data(directory_path, disease_prefix='hunt', control_prefix='control', max_len=100):
    """
    Loads gait .ts files from the PhysioNet dataset.
    Returns dictionaries of tensors for the disease class and healthy controls.
    """
    disease_data = []
    control_data = []
    
    if not os.path.exists(directory_path):
        print(f"Warning: Gait directory {directory_path} not found.")
        return {'disease': disease_data, 'HC': control_data}

    for filename in os.listdir(directory_path):
        if not filename.endswith('.ts'):
            continue
            
        filepath = os.path.join(directory_path, filename)
        
        try:
            # PhysioNet .ts files are typically space or tab separated text files
            # containing time and stride interval data.
            data = np.loadtxt(filepath)
            
            # Pad or truncate to max_len
            if len(data) == 0:
                continue
                
            # Usually shape is (N, 2) or (N, 3). We want to adapt it to the SensorAgent input.
            # SensorAgent expects (channels=16, seq_len=100) by default.
            # We will tile or pad to fit (16, max_len) for simplicity in this mock integration.
            channels = data.shape[1] if len(data.shape) > 1 else 1
            
            if len(data) > max_len:
                data = data[:max_len]
            else:
                pad_width = ((0, max_len - len(data)), (0, 0)) if len(data.shape) > 1 else (0, max_len - len(data))
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
                
            # Make it 16 channels by repeating
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
                
            # Shape is now (max_len, channels). Let's transpose to (channels, max_len)
            data = data.T
            
            # Pad channels to 16 to match the SensorAgent default
            if data.shape[0] < 16:
                pad_channels = 16 - data.shape[0]
                data = np.pad(data, ((0, pad_channels), (0, 0)), mode='constant')
            elif data.shape[0] > 16:
                data = data[:16, :]
                
            tensor_data = torch.tensor(data, dtype=torch.float32)
            
            if filename.startswith(disease_prefix):
                disease_data.append(tensor_data)
            elif filename.startswith(control_prefix):
                control_data.append(tensor_data)
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return {'disease': disease_data, 'HC': control_data}

if __name__ == '__main__':
    test_path = os.path.join(os.path.dirname(__file__), '../../data/gaitndd/gait-in-neurodegenerative-disease-database-1.0.0')
    data = load_gait_data(test_path)
    print(f"Loaded {len(data['disease'])} HD samples and {len(data['HC'])} Control samples.")
    if data['disease']:
        print(f"Sample shape: {data['disease'][0].shape}")
