# Explainable Agentic AI Framework for Neurological Disorders

This repository implements a multi-agent AI system for the early detection and progression prediction of neurological disorders (Parkinson's, Huntington's, Multiple Sclerosis) using multimodal data.

## Architecture Mapping

| Component | Paper Reference | Implementation Path |
| :--- | :--- | :--- |
| **Imaging Agent** | Section 4.2 (MRI) | `src/agents/imaging_agent.py` |
| **Speech Agent** | Section 4.3 (Voice) | `src/agents/speech_agent.py` |
| **Sensor Agent** | Section 4.4 (Gait/Wearables) | `src/agents/sensor_agent.py` |
| **Handwriting Agent** | Section 4.5 (Spirals) | `src/agents/handwriting_agent.py` |
| **Clinical Agent** | Section 4.6 (Records) | `src/agents/clinical_agent.py` |
| **Genetic Agent** | Section 4.7 (Markers) | `src/agents/genetic_agent.py` |
| **Fusion Agent** | Section 5 (Cross-Attention) | `src/agents/fusion_agent.py` |
| **Classification** | Section 6 (Softmax) | `src/agents/classification_agent.py` |
| **Explainability** | Section 7 (Grad-CAM, SHAP) | `src/agents/explainability_agent.py` |

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your downloaded datasets in the `data/raw/` directory.
3. Update `configs/default.yaml` as needed.
4. Run the pipeline:
   ```bash
   python run_pipeline.py --config configs/default.yaml
   ```

> **Note**: This pipeline is optimized for constrained hardware (4GB VRAM) by using lightweight variants of the proposed feature extractors.
