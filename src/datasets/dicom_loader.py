import SimpleITK as sitk
import numpy as np
import torch
import os

def load_dicom_series(directory_path, target_shape=(3, 224, 224)):
    """
    Reads a DICOM series from a directory using SimpleITK,
    resamples/crops it, and converts it to a normalized PyTorch tensor.
    
    Args:
        directory_path (str): Path to the folder containing DICOM files for one series.
        target_shape (tuple): Expected output shape for the model.
        
    Returns:
        torch.Tensor: The processed MRI volume.
    """
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
        if not dicom_names:
            print(f"Warning: No DICOM files found in {directory_path}")
            return torch.zeros(target_shape, dtype=torch.float32)

        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Convert to numpy array
        img_array = sitk.GetArrayFromImage(image)
        
        # Basic normalization
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)
        
        # For our multimodal pipeline, we typically need a fixed shape (e.g. 3, 224, 224)
        # In a real scenario, you'd use torchvision transforms or torch.nn.functional.interpolate
        # Here we do a simple resize/pad to fit target_shape to prevent crashing.
        tensor = torch.tensor(img_array, dtype=torch.float32)
        
        # Add batch and channel dimensions for interpolation: (1, 1, D, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        if len(target_shape) == 3:
            # If target shape is (C, H, W), we interpolate to that size
            import torch.nn.functional as F
            tensor = F.interpolate(tensor, size=(target_shape[0], target_shape[1], target_shape[2]), mode='trilinear', align_corners=False)
            tensor = tensor.squeeze(0).squeeze(0)
        
        return tensor

    except Exception as e:
        print(f"Error loading DICOM series {directory_path}: {e}")
        return torch.zeros(target_shape, dtype=torch.float32)

if __name__ == '__main__':
    # Test the loader on the downloaded MS dataset
    test_path = os.path.join(os.path.dirname(__file__), '../../data/ms_data2/ST000001/SE000001')
    if os.path.exists(test_path):
        tensor = load_dicom_series(test_path)
        print(f"Loaded tensor shape: {tensor.shape}")
    else:
        print("Data path not found for testing.")
