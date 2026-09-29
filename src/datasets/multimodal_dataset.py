import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class PairedMultimodalDataset(Dataset):
    """
    Synthetically pairs data from 6 modalities based on class (disease vs HC).
    Modalities: hw, clin, img, speech, sensor, genetic
    """
    def __init__(self, hw_data=None, clin_data=None, img_data=None, speech_data=None, sensor_data=None, genetic_data=None):
        
        self.data_dict = {
            'hw': hw_data or {'disease': [], 'HC': []},
            'clin': clin_data or {'disease': [], 'HC': []},
            'img': img_data or {'disease': [], 'HC': []},
            'speech': speech_data or {'disease': [], 'HC': []},
            'sensor': sensor_data or {'disease': [], 'HC': []},
            'genetic': genetic_data or {'disease': [], 'HC': []}
        }
        
        # Calculate max samples (assuming clinical or hw has the most if not all are present)
        disease_lens = [len(v['disease']) for v in self.data_dict.values() if len(v['disease']) > 0]
        hc_lens = [len(v['HC']) for v in self.data_dict.values() if len(v['HC']) > 0]
        
        self.num_disease = max(disease_lens) if disease_lens else 0
        self.num_hc = max(hc_lens) if hc_lens else 0
        
        self.total_samples = self.num_disease + self.num_hc
        self.labels = [1] * self.num_disease + [0] * self.num_hc
        
        # Define default shapes for missing modalities
        self.default_shapes = {
            'hw': (5, 2000),      # Conv1D expects channels first: (5, 2000) but agent does permute, so return (2000, 5)
            'clin': (22,),
            'img': (3, 224, 224),
            'speech': (21,),
            'sensor': (16, 100),
            'genetic': (1000,)
        }

    def __len__(self):
        return self.total_samples
        
    def _get_random_sample(self, mod_dict, class_key, default_shape):
        if mod_dict and len(mod_dict[class_key]) > 0:
            idx = random.randint(0, len(mod_dict[class_key]) - 1)
            return mod_dict[class_key][idx]
        else:
            return torch.zeros(default_shape, dtype=torch.float32)

    def __getitem__(self, idx):
        label = self.labels[idx]
        class_key = 'disease' if label == 1 else 'HC'
        
        # Specifically get sequential hw data to preserve old logic
        if len(self.data_dict['hw'][class_key]) > 0:
            local_idx = idx if label == 1 else idx - self.num_disease
            # Handle out of bounds if modalities have different lengths
            if local_idx < len(self.data_dict['hw'][class_key]):
                hw_tensor = self.data_dict['hw'][class_key][local_idx]
            else:
                hw_tensor = self._get_random_sample(self.data_dict['hw'], class_key, (2000, 5))
        else:
            hw_tensor = self._get_random_sample(self.data_dict['hw'], class_key, (2000, 5))

        clin_tensor = self._get_random_sample(self.data_dict['clin'], class_key, self.default_shapes['clin'])
        img_tensor = self._get_random_sample(self.data_dict['img'], class_key, self.default_shapes['img'])
        speech_tensor = self._get_random_sample(self.data_dict['speech'], class_key, self.default_shapes['speech'])
        sensor_tensor = self._get_random_sample(self.data_dict['sensor'], class_key, self.default_shapes['sensor'])
        genetic_tensor = self._get_random_sample(self.data_dict['genetic'], class_key, self.default_shapes['genetic'])
            
        inputs = {
            'hw': hw_tensor,
            'clin': clin_tensor,
            'img': img_tensor,
            'speech': speech_tensor,
            'sensor': sensor_tensor,
            'genetic': genetic_tensor
        }
        return inputs, torch.tensor(label, dtype=torch.long)
