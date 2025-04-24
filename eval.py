import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re
from matplotlib.colors import LinearSegmentedColormap
import math
import json
from datetime import datetime
from config import *

def evaluate_hme_model(model, data_loader, criterion, latex_vocab_reverse, device="cuda"):
    """
    Evaluate the Handwritten Math Expression recognition model
    
    Args:
    - model:                  Trained Seq2Seq model
    - data_loader:            DataLoader for evaluation data
    - criterion:              Loss function
    - latex_vocab_reverse:    Dictionary mapping indices to LaTeX tokens
    - device:                 Device to run evaluation on
        
    Returns:
      dict: Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    exact_matches = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    character_error_rates = []
    token_accuracies = []
    
    # For analysis of common errors
    token_errors = {}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs, lengths, targets = batch
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            
            # Forward pass without teacher forcing
            outputs = model(inputs, lengths, targets, teacher_forcing_ratio=0.0)
            
            # Calculate loss
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            targets_flat = targets[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            # Calculate metrics
            predictions = outputs.argmax(dim=-1)
            batch_size = targets.size(0)
            
            for i in range(batch_size):
                pred_seq = predictions[i, 1:].cpu().tolist()  # Skip SOS token
                true_seq = targets[i, 1:].cpu().tolist()
                
                # Find end of sequences (EOS token or end of padded sequence)
                pred_end = pred_seq.index(LATEX_VOCAB['<eos>']) if LATEX_VOCAB['<eos>'] in pred_seq else len(pred_seq)
                true_end = true_seq.index(LATEX_VOCAB['<eos>']) if LATEX_VOCAB['<eos>'] in true_seq else len(true_seq)
                
                # Truncate sequences at EOS
                pred_seq = pred_seq[:pred_end]
                true_seq = true_seq[:true_end]
                
                # Convert to LaTeX strings
                pred_tokens = [latex_vocab_reverse.get(idx, '<unk>') for idx in pred_seq]
                true_tokens = [latex_vocab_reverse.get(idx, '<unk>') for idx in true_seq]
                
                pred_latex = ''.join(pred_tokens)
                true_latex = ''.join(true_tokens)
                
                # Store for analysis
                all_predictions.append(pred_latex)
                all_targets.append(true_latex)
                
                # Calculate exact match
                if pred_latex == true_latex:
                    exact_matches += 1
                
                # Calculate CER (Character Error Rate)
                cer = calculate_cer(true_latex, pred_latex)
                character_error_rates.append(cer)
                
                # Calculate token accuracy
                correct_tokens = sum(1 for p, t in zip(pred_tokens, true_tokens) if p == t)
                max_len = max(len(pred_tokens), len(true_tokens))
                token_acc = correct_tokens / max_len if max_len > 0 else 1.0
                token_accuracies.append(token_acc)
                
                # Track token-level errors
                for j, (t, p) in enumerate(zip(true_tokens, pred_tokens)):
                    if t != p:
                        error_key = f"{t} → {p}"
                        token_errors[error_key] = token_errors.get(error_key, 0) + 1
                
                # Track missing or extra tokens
                if len(true_tokens) > len(pred_tokens):
                    for j in range(len(pred_tokens), len(true_tokens)):
                        error_key = f"{true_tokens[j]} → [MISSING]"
                        token_errors[error_key] = token_errors.get(error_key, 0) + 1
                elif len(pred_tokens) > len(true_tokens):
                    for j in range(len(true_tokens), len(pred_tokens)):
                        error_key = f"[EXTRA] → {pred_tokens[j]}"
                        token_errors[error_key] = token_errors.get(error_key, 0) + 1
                
                total_samples += 1
    
    # Calculate final metrics
    avg_loss = total_loss / len(data_loader)
    exact_match_accuracy = exact_matches / total_samples if total_samples > 0 else 0
    avg_cer = sum(character_error_rates) / len(character_error_rates) if character_error_rates else 0
    avg_token_accuracy = sum(token_accuracies) / len(token_accuracies) if token_accuracies else 0
    
    # Sort token errors by frequency
    sorted_token_errors = {k: v for k, v in sorted(token_errors.items(), key=lambda item: item[1], reverse=True)}
    
    # Compute symbol-level confusion matrix for most common tokens
    symbol_confusion = compute_symbol_confusion(all_targets, all_predictions)
    
    return {
        'loss': avg_loss,
        'exact_match_accuracy': exact_match_accuracy,
        'character_error_rate': avg_cer,
        'token_accuracy': avg_token_accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'token_errors': sorted_token_errors,
        'symbol_confusion': symbol_confusion
    }

def tokenize_expression(latex_str):
    """Transform a Latex math string into a list of tokens"""
    command_re = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\\*|[a-zA-Z]+|.)')
    tokens = []
    s = latex_str
    
    while s:
        if s[0] == '\\':
            match = command_re.match(s)
            if match:
                tokens.append(match.group(0))
                s = s[len(tokens[-1]):]
            else:
                tokens.append(s[0])
                s = s[1:]
        else:
            tokens.append(s[0])
            s = s[1:]
    
    return tokens

def calculate_cer(ref, hyp):
    """
    Calculate Character Error Rate
    
    Args:
    - ref: Reference string
    - hyp: Hypothesis string
        
    Returns:
        float: Character Error Rate between 0 and 1
    """
    # Convert to token lists if they aren't already
    ref_tokens = tokenize_expression(ref) if isinstance(ref, str) else ref
    hyp_tokens = tokenize_expression(hyp) if isinstance(hyp, str) else hyp
    
    # Dynamic programming for edit distance
    d = np.zeros((len(ref_tokens) + 1, len(hyp_tokens) + 1))
    
    # Initialize first row and column
    for i in range(len(ref_tokens) + 1):
        d[i, 0] = i
    for j in range(len(hyp_tokens) + 1):
        d[0, j] = j
    
    # Compute edit distance
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                d[i, j] = d[i-1, j-1]  # No operation needed
            else:
                d[i, j] = min(
                    d[i-1, j] + 1,    # Deletion
                    d[i, j-1] + 1,    # Insertion
                    d[i-1, j-1] + 1   # Substitution
                )
    
    # Calculate CER
    if len(ref_tokens) == 0:
        return 1.0 if len(hyp_tokens) > 0 else 0.0
    
    return d[len(ref_tokens), len(hyp_tokens)] / len(ref_tokens)

def compute_symbol_confusion(targets, predictions, top_k=20):
    """
    Compute symbol-level confusion statistics
    
    Args:
    - targets:     List of ground truth LaTeX strings
    - predictions: List of predicted LaTeX strings
    - top_k:       Number of most common tokens to include
        
    Returns:
        dict: Symbol confusion statistics
    """
    # Tokenize all strings
    tokenized_targets = [tokenize_expression(t) for t in targets]
    tokenized_predictions = [tokenize_expression(p) for p in predictions]
    
    # Count token frequencies in ground truth
    token_counts = {}
    for tokens in tokenized_targets:
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Get most common tokens
    top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_token_set = {token for token, _ in top_tokens}
    
    # Initialize confusion counts
    confusion = {}
    
    # Count confusions
    for target_tokens, pred_tokens in zip(tokenized_targets, tokenized_predictions):
        for i, target_token in enumerate(target_tokens):
            if target_token in top_token_set:
                if i < len(pred_tokens):
                    pred_token = pred_tokens[i]
                    confusion_key = (target_token, pred_token)
                    confusion[confusion_key] = confusion.get(confusion_key, 0) + 1
                else:
                    # Missing prediction
                    confusion_key = (target_token, "[MISSING]")
                    confusion[confusion_key] = confusion.get(confusion_key, 0) + 1
    
    return {
        'top_tokens': [token for token, _ in top_tokens],
        'confusion_counts': confusion
    }

def visualize_evaluation_results(results, save_dir="evaluation_results"):
    """
    Create visualizations for evaluation metrics
    
    Args:
    - results:  Dictionary of evaluation results
    - save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Create a summary text file
    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Loss: {results['loss']:.4f}\n")
        f.write(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f} ({results['exact_match_accuracy']*100:.2f}%)\n")
        f.write(f"Character Error Rate: {results['character_error_rate']:.4f}\n")
        f.write(f"Token Accuracy: {results['token_accuracy']:.4f}\n\n")
        
        f.write(f"Top Token Errors:\n")
        for i, (error, count) in enumerate(list(results['token_errors'].items())[:20]):
            f.write(f"  {error}: {count}\n")
    
    # 2. Plot metrics
    plt.figure(figsize=(12, 6))
    metrics = ['exact_match_accuracy', 'token_accuracy']
    values = [results['exact_match_accuracy'], results['token_accuracy']]
    colors = ['#4CAF50', '#2196F3']
    
    plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1.0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
    
    plt.ylabel('Accuracy')
    plt.title('Model Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_metrics.png"))
    plt.close()
    
    # 3. Plot Character Error Rate
    plt.figure(figsize=(6, 6))
    plt.pie([results['character_error_rate'], 1-results['character_error_rate']], 
            labels=['Error', 'Correct'], colors=['#F44336', '#4CAF50'],
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Character Error Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "character_error_rate.png"))
    plt.close()
    
    # 4. Plot top token errors
    plt.figure(figsize=(12, 6))
    top_errors = list(results['token_errors'].items())[:15]
    labels = [error for error, _ in top_errors]
    counts = [count for _, count in top_errors]
    
    plt.barh(range(len(labels)), counts, color='#FF5722')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Count')
    plt.title('Top Token Errors')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "token_errors.png"))
    plt.close()
    
    # 5. Plot confusion matrix for top tokens
    if results['symbol_confusion']['confusion_counts']:
        top_tokens = results['symbol_confusion']['top_tokens']
        conf_matrix = np.zeros((len(top_tokens), len(top_tokens) + 1))
        
        # Map token to index
        token_to_idx = {token: i for i, token in enumerate(top_tokens)}
        
        # Fill confusion matrix
        for (true_token, pred_token), count in results['symbol_confusion']['confusion_counts'].items():
            if true_token in token_to_idx:
                true_idx = token_to_idx[true_token]
                if pred_token in token_to_idx:
                    pred_idx = token_to_idx[pred_token]
                else:
                    # Last column for "other" or missing
                    pred_idx = len(top_tokens)
                conf_matrix[true_idx, pred_idx] += count
        
        # Normalize by row
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_norm = np.zeros_like(conf_matrix)
        nonzero_rows = row_sums > 0
        conf_matrix_norm[nonzero_rows.flatten()] = conf_matrix[nonzero_rows.flatten()] / row_sums[nonzero_rows]
        
        # Plot
        plt.figure(figsize=(12, 10))
        labels = top_tokens + ["Other"]
        sns.heatmap(conf_matrix_norm, annot=False, cmap="YlGnBu", 
                    xticklabels=labels, yticklabels=top_tokens,
                    vmin=0, vmax=1)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Symbol Confusion Matrix (Normalized)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()
    
    # 6. Save all prediction examples to CSV
    df = pd.DataFrame({
        'target': results['targets'],
        'prediction': results['predictions'],
        'correct': [t == p for t, p in zip(results['targets'], results['predictions'])],
        'cer': [calculate_cer(t, p) for t, p in zip(results['targets'], results['predictions'])]
    })
    df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    
    # 7. Create histogram of CER
    cers = [calculate_cer(t, p) for t, p in zip(results['targets'], results['predictions'])]
    plt.figure(figsize=(10, 6))
    plt.hist(cers, bins=20, color='#3F51B5', alpha=0.7)
    plt.axvline(np.mean(cers), color='r', linestyle='dashed', linewidth=2, label=f'Mean CER: {np.mean(cers):.4f}')
    plt.xlabel('Character Error Rate')
    plt.ylabel('Count')
    plt.title('Distribution of Character Error Rates')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cer_distribution.png"))
    plt.close()
    
    print(f"Evaluation visualizations saved to {save_dir}")

def evaluate_by_complexity(results, save_dir="evaluation_results"):
    """
    Analyze performance by expression complexity
    
    Args:
    - results:  Dictionary of evaluation results
    - save_dir: Directory to save visualizations
    """
    # Define complexity metrics
    def expression_complexity(latex):
        tokens = tokenize_expression(latex)
        
        # Count special elements that indicate complexity
        fractions = sum(1 for t in tokens if t == '\\frac')
        subscripts = sum(1 for i, t in enumerate(tokens) if t == '_' and i < len(tokens)-1)
        superscripts = sum(1 for i, t in enumerate(tokens) if t == '^' and i < len(tokens)-1)
        special_symbols = sum(1 for t in tokens if t.startswith('\\') and t not in ['\\frac', '\\sum', '\\int'])
        brackets = sum(1 for t in tokens if t in ['{', '}', '(', ')', '[', ']'])
        
        # Weighted complexity score
        complexity = len(tokens) + 2*fractions + subscripts + superscripts + special_symbols + 0.5*brackets
        
        return complexity
    
    # Calculate complexity for each expression
    complexities = [expression_complexity(target) for target in results['targets']]
    
    # Create dataframe with complexity and error metrics
    df = pd.DataFrame({
        'target': results['targets'],
        'prediction': results['predictions'],
        'complexity': complexities,
        'correct': [t == p for t, p in zip(results['targets'], results['predictions'])],
        'cer': [calculate_cer(t, p) for t, p in zip(results['targets'], results['predictions'])]
    })
    
    # Define complexity bins
    max_complexity = max(complexities)
    bins = np.linspace(0, max_complexity, 6)
    labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins)-1)]
    
    df['complexity_bin'] = pd.cut(df['complexity'], bins=bins, labels=labels)
    
    # Calculate metrics by complexity bin
    bin_metrics = df.groupby('complexity_bin').agg({
        'correct': 'mean',
        'cer': 'mean',
        'target': 'count'
    }).reset_index()
    
    bin_metrics = bin_metrics.rename(columns={
        'correct': 'exact_match_rate',
        'cer': 'avg_cer',
        'target': 'count'
    })
    
    # Plot metrics by complexity
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(bin_metrics['complexity_bin'], bin_metrics['exact_match_rate'], color='#4CAF50')
    ax1.set_xlabel('Expression Complexity')
    ax1.set_ylabel('Exact Match Rate')
    ax1.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(bin_metrics['complexity_bin'], bin_metrics['avg_cer'], color='#F44336')
    ax2.set_xlabel('Expression Complexity')
    ax2.set_ylabel('Average Character Error Rate')
    ax2.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_by_complexity.png"))
    plt.close()
    
    # Save to CSV
    bin_metrics.to_csv(os.path.join(save_dir, "performance_by_complexity.csv"), index=False)
    
    # Also plot sample counts for reference
    plt.figure(figsize=(8, 5))
    plt.bar(bin_metrics['complexity_bin'], bin_metrics['count'], color='#2196F3')
    plt.xlabel('Expression Complexity')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Expression Complexity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "complexity_distribution.png"))
    plt.close()

def visualize_sample_predictions(model, test_dataset, latex_vocab_reverse, num_samples=5, save_dir="sample_predictions"):
    """
    Visualize sample predictions from the model
    
    Args:
    - model:               Trained model
    - test_dataset:        Test dataset
    - latex_vocab_reverse: Dictionary mapping indices to LaTeX tokens
    - num_samples:         Number of samples to visualize
    - save_dir:            Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get random sample indices
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        # Get sample
        features, ground_truth = test_dataset[idx]
        ink_file = os.path.join(test_dataset.split_dir, test_dataset.ink_files[idx])
        ink = read_inkml_file(ink_file)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            # Add batch dimension
            input_tensor = features.unsqueeze(0).to(DEVICE)
            input_length = torch.tensor([features.size(0)]).to(DEVICE)
            
            # Initialize with SOS token
            input_token = torch.tensor([LATEX_VOCAB['<sos>']]).to(DEVICE)
            
            # Encoder forward pass
            encoder_outputs, (hidden, cell) = model.encoder(input_tensor, input_length)
            
            if model.encoder.bidirectional:
                hidden = hidden.view(model.encoder.num_layers, 2, hidden.size(1), hidden.size(2)).sum(dim=1)
                cell = cell.view(model.encoder.num_layers, 2, cell.size(1), cell.size(2)).sum(dim=1)
            
            # Create mask for attention
            mask = model.create_mask(input_tensor)
            
            # Generate sequence
            output_indices = [LATEX_VOCAB['<sos>']]
            attention_weights = []
            
            for _ in range(100):  # Max length
                prediction, hidden, cell, attn_weights = model.decoder(
                    input_token, hidden, cell, encoder_outputs, mask
                )
                
                attention_weights.append(attn_weights.squeeze(0).cpu())
                
                top_token = prediction.argmax(1).item()
                output_indices.append(top_token)
                
                if top_token == LATEX_VOCAB['<eos>']:
                    break
                
                input_token = torch.tensor([top_token]).to(DEVICE)
        
        # Convert indices to LaTeX
        filtered_indices = [idx for idx in output_indices if idx not in [
            LATEX_VOCAB['<pad>'], LATEX_VOCAB['<sos>'], LATEX_VOCAB['<eos>']
        ]]
        predicted_latex = ''.join([latex_vocab_reverse.get(idx, '<unk>') for idx in filtered_indices])
        
        # Calculate error rate
        cer = calculate_cer(ground_truth, predicted_latex)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # Plot the ink
        ax1 = fig.add_subplot(211)
        for stroke in ink.strokes:
            ax1.plot(stroke[0], stroke[1])
        ax1.set_title("Input Ink")
        ax1.invert_yaxis()
        ax1.set_aspect('equal')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Plot attention heatmap
        if attention_weights:
            ax2 = fig.add_subplot(212)
            attention_matrix = torch.stack(attention_weights).numpy()
            
            # Limit to reasonable size
            max_shown = min(30, attention_matrix.shape[0])
            attention_display = attention_matrix[:max_shown, :min(50, attention_matrix.shape[1])]
            
            im = ax2.imshow(attention_display, cmap='viridis', aspect='auto')
            ax2.set_title("Attention Weights")
            ax2.set_xlabel("Input Sequence")
            ax2.set_ylabel("Output Sequence")
            fig.colorbar(im, ax=ax2)
        
        # Add text for the predictions
        match_status = "CORRECT" if predicted_latex == ground_truth else "ERROR"
        status_color = "green" if match_status == "CORRECT" else "red"
        
        plt.figtext(0.1, 0.02, f"Ground Truth: {ground_truth}", fontsize=12)
        plt.figtext(0.1, 0.05, f"Prediction: {predicted_latex}", fontsize=12)
        plt.figtext(0.1, 0.08, f"Status: {match_status} (CER: {cer:.4f})", fontsize=12, color=status_color)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
        plt.close()
    
    print(f"Sample predictions saved to {save_dir}")

def run_complete_evaluation(model, test_loader, test_dataset, criterion, latex_vocab_reverse, device="cuda"):
    """
    Run a complete evaluation of the model
    
    Args:
    - model:               Trained model
    - test_loader:         DataLoader for test data
    - test_dataset:        Test dataset
    - criterion:           Loss function
    - latex_vocab_reverse: Dictionary mapping indices to LaTeX tokens
    - device:              Device to run evaluation on
        
    Returns:
        dict: Evaluation results
    """
    # Create main evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = f"evaluation_{timestamp}"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Run main evaluation
    print("Evaluating model performance...")
    results = evaluate_hme_model(model, test_loader, criterion, latex_vocab_reverse, device)
    
    # Save raw results
    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        # Convert non-serializable items
        serializable_results = {
            'loss': results['loss'],
            'exact_match_accuracy': results['exact_match_accuracy'],
            'character_error_rate': results['character_error_rate'],
            'token_accuracy': results['token_accuracy'],
            'token_errors': {str(k): v for k, v in results['token_errors'].items()},
            'sample_predictions': [{
                'target': results['targets'][i],
                'prediction': results['predictions'][i]
            } for i in range(min(100, len(results['targets'])))]
        }
        json.dump(serializable_results, f, indent=2)
    
    # Generate visualizations
    print("Generating evaluation visualizations...")
    visualize_evaluation_results(results, save_dir=os.path.join(eval_dir, "metrics"))
    
    # Analyze by complexity
    print("Analyzing performance by expression complexity...")
    evaluate_by_complexity(results, save_dir=os.path.join(eval_dir, "complexity_analysis"))
    
    # Visualize sample predictions
    print("Visualizing sample predictions...")
    visualize_sample_predictions(
        model, test_dataset, latex_vocab_reverse, 
        num_samples=10, save_dir=os.path.join(eval_dir, "samples")
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f} ({results['exact_match_accuracy']*100:.2f}%)")
    print(f"Character Error Rate: {results['character_error_rate']:.4f}")
    print(f"Token Accuracy: {results['token_accuracy']:.4f}")
    print(f"\nEvaluation results saved to {eval_dir}")
    
    return results
