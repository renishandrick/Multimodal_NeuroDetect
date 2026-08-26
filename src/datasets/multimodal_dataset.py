import torch
from torch.utils.data import Dataset
import random

class PairedMultimodalDataset(Dataset):
    """
    Synthetically pairs handwriting data with clinical data based on their class (PD or HC).
    """
    def __init__(self, handwriting_data, clinical_data):
        """
        Args:
            handwriting_data (dict): Dictionary with keys 'PD' and 'HC' containing handwriting tensors.
            clinical_data (dict): Dictionary with keys 'PD' and 'HC' containing clinical tensors.
        """
        self.hw_pd = handwriting_data['PD']
        self.hw_hc = handwriting_data['HC']
        
        self.clin_pd = clinical_data['PD']
        self.clin_hc = clinical_data['HC']
        
        self.num_pd = len(self.hw_pd)
        self.num_hc = len(self.hw_hc)
        
        self.total_samples = self.num_pd + self.num_hc
        
        # Determine labels: 1 for PD, 0 for HC
        self.labels = [1] * self.num_pd + [0] * self.num_hc
        
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if label == 1: # PD
            # Get the handwriting sample based on index
            hw_tensor = self.hw_pd[idx]
            # Synthetically pair with a random PD clinical profile
            random_clin_idx = random.randint(0, len(self.clin_pd) - 1)
            clin_tensor = self.clin_pd[random_clin_idx]
        else: # HC
            # Adjust index for HC
            hc_idx = idx - self.num_pd
            hw_tensor = self.hw_hc[hc_idx]
            # Synthetically pair with a random HC clinical profile
            random_clin_idx = random.randint(0, len(self.clin_hc) - 1)
            clin_tensor = self.clin_hc[random_clin_idx]
            
        return (hw_tensor, clin_tensor), torch.tensor(label, dtype=torch.long)
