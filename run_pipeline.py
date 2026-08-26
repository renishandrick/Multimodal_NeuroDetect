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

from src.agents.handwriting_agent import HandwritingAgent
from src.agents.clinical_agent import ClinicalAgent
from src.agents.fusion_agent import FusionAgent

def main():
    parser = argparse.ArgumentParser(description="Run the Explainable Agentic AI Framework")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
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
    print("Loading data...")
    data_dir = os.path.join('data', 'raw')
    hw_dir = os.path.join(data_dir, 'hw_dataset')
    clinical_file = os.path.join(data_dir, 'parkinsons.data')
    
    hw_data = load_handwriting_data(hw_dir, max_len=2000)
    clin_data = load_clinical_data(clinical_file)
    
    dataset = PairedMultimodalDataset(hw_data, clin_data)
    
    batch_size = config['training']['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model Instantiation
    print("Initializing agents...")
    embed_dim = config['models']['fusion']['embedding_dim']
    num_heads = config['models']['fusion']['num_heads']
    
    # We use 2 classes: PD (1) and HC (0)
    num_classes = 2
    
    hw_agent = HandwritingAgent(in_channels=5, embedding_dim=embed_dim).to(device)
    clin_agent = ClinicalAgent(in_features=22, embedding_dim=embed_dim).to(device)
    fusion_agent = FusionAgent(embedding_dim=embed_dim, num_heads=num_heads, num_classes=num_classes).to(device)
    
    # 4. Training Setup
    criterion = nn.CrossEntropyLoss()
    
    # Combine all parameters for the optimizer
    all_params = list(hw_agent.parameters()) + list(clin_agent.parameters()) + list(fusion_agent.parameters())
    learning_rate = config['training']['learning_rate']
    optimizer = optim.Adam(all_params, lr=learning_rate)
    
    epochs = config['training']['epochs']
    
    # 5. Training Loop
    print(f"Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        hw_agent.train()
        clin_agent.train()
        fusion_agent.train()
        
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            hw_inputs, clin_inputs = inputs
            
            # Move to device
            hw_inputs = hw_inputs.to(device)
            clin_inputs = clin_inputs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward Pass: Extract features
            hw_embed = hw_agent(hw_inputs)
            clin_embed = clin_agent(clin_inputs)
            
            # Forward Pass: Fuse and Classify
            logits = fusion_agent(hw_embed, clin_embed)
            
            # Compute Loss
            loss = criterion(logits, labels)
            
            # Backward Pass and Optimize
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item() * hw_inputs.size(0)
            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds / total_samples * 100
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
