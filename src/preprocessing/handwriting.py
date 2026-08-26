import os
import torch
import numpy as np

def load_handwriting_data(data_dir, max_len=2000):
    """
    Loads handwriting .txt files and processes them into padded PyTorch tensors.
    
    Args:
        data_dir (str): Path to 'hw_dataset' directory.
        max_len (int): Maximum sequence length to pad/truncate to.
        
    Returns:
        dict: A dictionary with 'PD' and 'HC' keys, each containing a list of PyTorch tensors.
    """
    control_dir = os.path.join(data_dir, 'control')
    parkinson_dir = os.path.join(data_dir, 'parkinson')
    
    data = {'HC': [], 'PD': []}
    
    def process_folder(folder_path, label):
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} does not exist.")
            return
            
        for file in os.listdir(folder_path):
            if file.endswith('.txt'):
                filepath = os.path.join(folder_path, file)
                try:
                    # Read lines, ignoring potential headers or empty lines
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                    
                    parsed_data = []
                    for line in lines:
                        parts = line.strip().split(';')
                        # Ensure we have at least X, Y, Z, Pressure, GripAngle
                        if len(parts) >= 5:
                            try:
                                # Extract X, Y, Z, Pressure, GripAngle
                                values = [float(p) for p in parts[:5]]
                                parsed_data.append(values)
                            except ValueError:
                                continue # Skip header or invalid lines
                    
                    if not parsed_data:
                        continue
                        
                    tensor_data = torch.tensor(parsed_data, dtype=torch.float32)
                    
                    # Pad or truncate to max_len
                    seq_len = tensor_data.shape[0]
                    if seq_len > max_len:
                        tensor_data = tensor_data[:max_len, :]
                    elif seq_len < max_len:
                        padding = torch.zeros((max_len - seq_len, 5), dtype=torch.float32)
                        tensor_data = torch.cat((tensor_data, padding), dim=0)
                        
                    data[label].append(tensor_data)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    
    process_folder(control_dir, 'HC')
    process_folder(parkinson_dir, 'PD')
    
    # Stack into single tensors
    if data['HC']:
        data['HC'] = torch.stack(data['HC'])
    if data['PD']:
        data['PD'] = torch.stack(data['PD'])
        
    return data

if __name__ == "__main__":
    # Simple test
    test_dir = os.path.join('..', '..', 'data', 'raw', 'hw_dataset')
    data = load_handwriting_data(test_dir)
    print("Handwriting Preprocessing Complete.")
    if len(data['HC']) > 0:
        print(f"HC Tensor Shape: {data['HC'].shape}")
    if len(data['PD']) > 0:
        print(f"PD Tensor Shape: {data['PD'].shape}")
