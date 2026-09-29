import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.preprocessing.handwriting import load_handwriting_data
from src.preprocessing.clinical import load_clinical_data
from src.datasets.multimodal_dataset import PairedMultimodalDataset
from src.datasets.dicom_loader import load_dicom_series
from src.datasets.gait_loader import load_gait_data

from src.agents.handwriting_agent import HandwritingAgent
from src.agents.clinical_agent import ClinicalAgent
from src.agents.imaging_agent import ImagingAgent
from src.agents.speech_agent import SpeechAgent
from src.agents.sensor_agent import SensorAgent
from src.agents.genetic_agent import GeneticAgent
from src.agents.fusion_agent import FusionAgent
from src.agents.classification_agent import ClassificationAgent
from src.evaluation.metrics import calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Run the Explainable Agentic AI Framework")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--disease", type=str, default="PD", choices=["PD", "MS", "HD"], help="Target disease to train for")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Starting pipeline for {config['project']['name']}")
    
    # 1. Device Setup
    device_str = config['hardware']['device']
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # 2. Data Pipeline
    print(f"Loading data for {args.disease}...")
    data_dir = os.path.join('data')
    
    # Initialize empty modalities
    hw_data, clin_data, img_data, speech_data, sensor_data, genetic_data = None, None, None, None, None, None
    
    if args.disease == "PD":
        hw_dir = os.path.join(data_dir, 'raw', 'hw_dataset')
        clinical_file = os.path.join(data_dir, 'raw', 'parkinsons.data')
        hw_data = load_handwriting_data(hw_dir, max_len=2000)
        clin_data = load_clinical_data(clinical_file)
        
    elif args.disease == "MS":
        # Load Multiple Sclerosis DICOM MRI
        dicom_path = os.path.join(data_dir, 'ms_data2', 'ST000001', 'SE000001')
        print("Loading MS DICOM MRI series...")
        ms_tensor = load_dicom_series(dicom_path)
        # Mocking 50 samples of MS and 50 controls from the single patient scan to run training loop
        img_data = {'disease': [ms_tensor for _ in range(50)], 'HC': [torch.zeros_like(ms_tensor) for _ in range(50)]}
        
    elif args.disease == "HD":
        # Load Huntington's Disease Gait Sensor Data
        gait_dir = os.path.join(data_dir, 'gaitndd', 'gait-in-neurodegenerative-disease-database-1.0.0')
        print("Loading HD PhysioNet Gait data...")
        sensor_data = load_gait_data(gait_dir, disease_prefix='hunt', control_prefix='control')
    
    dataset = PairedMultimodalDataset(
        hw_data=hw_data, clin_data=clin_data, img_data=img_data, 
        speech_data=speech_data, sensor_data=sensor_data, genetic_data=genetic_data
    )
    
    batch_size = config['training']['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model Instantiation
    print("Initializing all 6 modality agents, fusion, and classification...")
    embed_dim = config['models']['fusion']['embedding_dim']
    num_heads = config['models']['fusion']['num_heads']
    num_classes = 2
    
    agents = {
        'hw': HandwritingAgent(in_channels=5, embedding_dim=embed_dim).to(device),
        'clin': ClinicalAgent(in_features=22, embedding_dim=embed_dim).to(device),
        'img': ImagingAgent(in_channels=3, embedding_dim=embed_dim).to(device),
        'speech': SpeechAgent(in_features=21, embedding_dim=embed_dim).to(device),
        'sensor': SensorAgent(in_channels=16, embedding_dim=embed_dim).to(device),
        'genetic': GeneticAgent(in_features=1000, embedding_dim=embed_dim).to(device)
    }
    
    fusion_agent = FusionAgent(embedding_dim=embed_dim, num_heads=num_heads).to(device)
    # 6 modalities means fused shape is embedding_dim * 6
    classification_agent = ClassificationAgent(in_features=embed_dim * 6, num_classes=num_classes).to(device)
    
    # 4. Training Setup
    criterion = nn.CrossEntropyLoss()
    
    all_params = list(fusion_agent.parameters()) + list(classification_agent.parameters())
    for agent in agents.values():
        all_params += list(agent.parameters())
        
    learning_rate = config['training']['learning_rate']
    optimizer = optim.Adam(all_params, lr=learning_rate)
    
    epochs = config['training']['epochs']
    
    # 5. Training Loop
    print(f"Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        for agent in agents.values():
            agent.train()
        fusion_agent.train()
        classification_agent.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch_idx, (inputs_dict, labels) in enumerate(dataloader):
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Forward Pass: Extract features for all modalities
            embeddings = []
            for mod, agent in agents.items():
                inputs_dict[mod] = inputs_dict[mod].to(device)
                embeddings.append(agent(inputs_dict[mod]))
            
            # Forward Pass: Fuse
            fused_repr = fusion_agent(embeddings)
            
            # Forward Pass: Classify
            logits = classification_agent(fused_repr)
            
            # Compute Loss
            loss = criterion(logits, labels)
            
            # Backward Pass and Optimize
            loss.backward()
            optimizer.step()
            
            # Metrics Collection
            running_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1] # prob of positive class
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
        epoch_loss = running_loss / len(dataset)
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - "
              f"Acc: {metrics['accuracy'] * 100:.2f}% - F1: {metrics['f1_score']:.4f} - AUC: {metrics['roc_auc']:.4f}")
        
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
