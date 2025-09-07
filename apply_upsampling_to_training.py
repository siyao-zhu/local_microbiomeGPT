"""
Script to demonstrate how to apply upsampling to training data before tokenization
in the microbiomeGPT pipeline.
"""

import numpy as np
from sklearn.model_selection import train_test_split

# The gaussian_noise_upsampling function (already in your notebook)
def gaussian_noise_upsampling(X, y, noise_std=0.1, random_state=42):
    """
    Generate synthetic samples for minority class using Gaussian noise.
    
    Args:
        X: Feature matrix (samples x features)
        y: Target labels
        noise_std: Standard deviation for Gaussian noise
        random_state: Random seed for reproducibility
    
    Returns:
        X_balanced: Balanced feature matrix
        y_balanced: Balanced target labels
        stats: Dictionary with balancing statistics
    """
    np.random.seed(random_state)
    
    # Find minority and majority classes
    unique_labels, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    
    print(f"Original class distribution: {label_counts}")
    
    if len(unique_labels) == 1:
        print("Warning: Only one class found. No balancing needed.")
        return X, y, {"message": "No balancing performed - single class"}
    
    minority_label = min(label_counts, key=label_counts.get)
    majority_count = max(label_counts.values())
    minority_count = label_counts[minority_label]
    
    # Get minority samples and generate synthetic ones
    minority_mask = (y == minority_label)
    minority_X = X[minority_mask].copy()
    samples_needed = majority_count - minority_count
    
    synthetic_X_list = []
    synthetic_y_list = []
    
    for i in range(samples_needed):
        # Pick random minority sample as base
        base_idx = np.random.randint(0, len(minority_X))
        base_sample = minority_X[base_idx].copy()
        
        # Add noise only to non-zero species
        nonzero_mask = base_sample > 0
        noise = np.random.normal(0, noise_std, size=base_sample.shape)
        noise[~nonzero_mask] = 0  # Zero out noise for zero abundance species
        
        synthetic_sample = base_sample + noise
        synthetic_sample = np.maximum(synthetic_sample, 0)  # No negative values
        
        # For binned data, we need to ensure integer values and reasonable bounds
        synthetic_sample = np.round(synthetic_sample).astype(int)
        # Assuming config.n_bins exists, otherwise use a reasonable default
        n_bins = 51  # Update this based on your config.n_bins value
        synthetic_sample = np.clip(synthetic_sample, 0, n_bins - 1)  # Keep within bin range
        
        synthetic_X_list.append(synthetic_sample)
        synthetic_y_list.append(minority_label)
    
    # Combine original + synthetic
    if samples_needed > 0:
        X_balanced = np.vstack([X, np.array(synthetic_X_list)])
        y_balanced = np.hstack([y, np.array(synthetic_y_list)])
    else:
        X_balanced = X
        y_balanced = y
    
    final_unique, final_counts = np.unique(y_balanced, return_counts=True)
    final_label_counts = dict(zip(final_unique, final_counts))
    print(f"After up-sampling: {final_label_counts}")
    
    stats = {
        "synthetic_generated": samples_needed,
        "final_distribution": final_label_counts,
        "original_distribution": label_counts
    }
    return X_balanced, y_balanced, stats


# IMPORTANT: Integration point in your notebook
# You need to modify your code as follows:

"""
# Assuming you have labels available (e.g., from your dataset)
# If labels are in adata.obs, you might have something like:
y_all = adata.obs['label'].values  # or whatever column contains your labels

# When splitting the data, split BOTH features and labels with the same random state
train_data, valid_data, y_train, y_valid = train_test_split(
    all_counts, 
    y_all,
    test_size=0.1, 
    shuffle=True,
    random_state=42  # Add random_state for reproducibility
)

# Apply upsampling to training data BEFORE tokenization
train_data_balanced, y_train_balanced, upsampling_stats = gaussian_noise_upsampling(
    train_data,
    y_train,
    noise_std=0.1,
    random_state=42
)

# Now use train_data_balanced in your tokenization instead of train_data
tokenized_train = tokenize_and_pad_batch(
    train_data_balanced,  # Use balanced data here
    organism_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=config.append_cls,
    include_zero_gene=config.include_zero_gene,
)

# Validation data remains unchanged
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    organism_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=config.append_cls,
    include_zero_gene=config.include_zero_gene,
)
"""

# Example of what the integration looks like:
def demonstrate_integration():
    # Simulated data for demonstration
    n_samples = 1000
    n_features = 313
    
    # Create synthetic data
    all_counts = np.random.randint(0, 10, size=(n_samples, n_features))
    
    # Create imbalanced labels (70% class 0, 30% class 1)
    y_all = np.zeros(n_samples, dtype=int)
    y_all[int(0.7 * n_samples):] = 1
    np.random.shuffle(y_all)
    
    # Split data AND labels together
    train_data, valid_data, y_train, y_valid = train_test_split(
        all_counts, 
        y_all,
        test_size=0.1, 
        shuffle=True,
        random_state=42
    )
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(valid_data)}")
    
    # Apply upsampling to training data
    train_data_balanced, y_train_balanced, stats = gaussian_noise_upsampling(
        train_data,
        y_train,
        noise_std=0.1,
        random_state=42
    )
    
    print(f"\nAfter upsampling:")
    print(f"Balanced training set size: {len(train_data_balanced)}")
    print(f"Synthetic samples generated: {stats['synthetic_generated']}")
    
    return train_data_balanced, y_train_balanced, valid_data, y_valid


if __name__ == "__main__":
    # Run demonstration
    train_balanced, y_train_balanced, valid, y_valid = demonstrate_integration()