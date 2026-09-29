from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates standard classification metrics.
    
    Args:
        y_true (list or np.array): True labels
        y_pred (list or np.array): Predicted labels
        y_prob (list or np.array, optional): Predicted probabilities for ROC-AUC
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, and roc_auc.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # In case only one class is present in y_true
            metrics['roc_auc'] = 0.0
            
    return metrics
