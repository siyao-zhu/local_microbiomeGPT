# Strategy 1: Up-sampling with Gaussian Noise - Implementation Instructions

## Overview
You need to implement Strategy 1 for balancing classes during fine-tuning for external validation. This strategy uses Gaussian noise to generate synthetic samples for the minority class.

## Key Requirements from the Paper
1. **Generate enough samples** for minority group to equal majority group size
2. **Add Gaussian noise only to non-zero abundance species**
3. **Start with std=0.1** but explore increasing up to **std=5.0**
4. **Apply only during fine-tuning for external validation** (not during pre-training)

## Step-by-Step Implementation

### Step 1: Add the Up-sampling Function to Your Notebook

Add this function to your notebook (preferably after the data loading section):

```python
def gaussian_noise_upsampling(
    abundance_data: np.ndarray,
    labels: np.ndarray,
    noise_std: float = 0.1,
    random_state: int = 42
) -> tuple:
    """
    Strategy 1: Up-sampling with Gaussian Noise for Class Balancing
    
    Generate enough samples for the minority group to make the number of samples 
    in that group equal to that of the majority group using Gaussian noise added 
    to non-zero abundance species.
    
    Args:
        abundance_data: Array of shape (n_samples, n_species) with relative abundances
        labels: Array of shape (n_samples,) with class labels (0=healthy, 1=diseased)
        noise_std: Standard deviation for Gaussian noise (start with 0.1, can explore up to 5.0)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (upsampled_data, upsampled_labels)
    """
    np.random.seed(random_state)
    
    # Identify class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    majority_class = unique_labels[np.argmax(counts)]
    minority_class = unique_labels[np.argmin(counts)]
    majority_count = counts[np.argmax(counts)]
    minority_count = counts[np.argmin(counts)]
    
    print(f"Class distribution before up-sampling:")
    print(f"  Class {majority_class}: {majority_count} samples (majority)")
    print(f"  Class {minority_class}: {minority_count} samples (minority)")
    print(f"  Imbalance ratio: {majority_count / minority_count:.2f}")
    
    # Calculate how many synthetic samples needed for minority class
    samples_needed = majority_count - minority_count
    print(f"  Synthetic samples needed: {samples_needed}")
    
    # Get minority class samples
    minority_indices = np.where(labels == minority_class)[0]
    minority_data = abundance_data[minority_indices]
    
    # Start with original data
    upsampled_data = abundance_data.copy()
    upsampled_labels = labels.copy()
    
    # Generate synthetic samples
    synthetic_data = []
    synthetic_labels = []
    
    for i in range(samples_needed):
        # Randomly select a minority sample as template
        template_idx = np.random.choice(len(minority_data))
        template_sample = minority_data[template_idx].copy()
        
        # Add Gaussian noise only to non-zero abundance species
        noise_mask = template_sample > 0  # Only add noise to non-zero species
        noise = np.random.normal(0, noise_std, size=template_sample.shape)
        
        # Apply noise only to non-zero positions
        synthetic_sample = template_sample.copy()
        synthetic_sample[noise_mask] += noise[noise_mask]
        
        # Ensure no negative values (clamp to 0)
        synthetic_sample = np.maximum(synthetic_sample, 0)
        
        # Optional: normalize to maintain relative abundance properties
        # if synthetic_sample.sum() > 0:
        #     synthetic_sample = synthetic_sample * (template_sample.sum() / synthetic_sample.sum())
        
        synthetic_data.append(synthetic_sample)
        synthetic_labels.append(minority_class)
    
    # Combine original and synthetic data
    if synthetic_data:
        synthetic_data = np.array(synthetic_data)
        synthetic_labels = np.array(synthetic_labels)
        
        upsampled_data = np.vstack([upsampled_data, synthetic_data])
        upsampled_labels = np.hstack([upsampled_labels, synthetic_labels])
    
    # Final class distribution
    unique_final, counts_final = np.unique(upsampled_labels, return_counts=True)
    print(f"\nClass distribution after up-sampling:")
    for label, count in zip(unique_final, counts_final):
        print(f"  Class {label}: {count} samples")
    print(f"  Balance ratio: {counts_final[0] / counts_final[1]:.2f}")
    
    return upsampled_data, upsampled_labels
```

### Step 2: Add Class Labels to Your Data

You need to add class labels to your dataset. Based on typical microbiome studies, you'll need:

```python
# Example: Add class labels based on sample names or metadata
# You'll need to replace this with your actual labeling logic

def add_class_labels_to_adata(adata):
    """
    Add class labels to AnnData object based on sample names or metadata.
    
    Modify this function based on how your data is organized:
    - If samples have disease/condition information in their names
    - If you have separate metadata files
    - If labels are encoded in sample IDs
    """
    
    # EXAMPLE 1: If sample names contain disease information
    sample_names = adata.obs.index.tolist()
    labels = []
    
    for sample in sample_names:
        # Modify these conditions based on your data
        if any(disease in sample.lower() for disease in ['disease', 'sick', 'patient', 'case']):
            labels.append(1)  # diseased
        elif any(healthy in sample.lower() for healthy in ['healthy', 'control', 'normal']):
            labels.append(0)  # healthy
        else:
            # Default or unknown - you may need to handle this
            labels.append(0)  # or raise an error
    
    adata.obs['class_label'] = labels
    return adata

# Apply the labeling
adata = add_class_labels_to_adata(adata)
print("Class distribution:")
print(adata.obs['class_label'].value_counts())
```

### Step 3: Prepare Data for External Validation

When you're ready to fine-tune for external validation:

```python
def prepare_balanced_external_validation_data(adata, noise_std=0.1, test_size=0.2, random_state=42):
    """
    Prepare balanced data for external validation using up-sampling.
    
    Args:
        adata: AnnData object with abundance data and class labels
        noise_std: Standard deviation for Gaussian noise
        test_size: Proportion of data to use for external validation
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Dictionary with train and validation data (both balanced)
    """
    from sklearn.model_selection import train_test_split
    
    # Extract data
    X = adata.X  # abundance data
    y = adata.obs['class_label'].values  # class labels
    
    # Split into train and external validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("Original data splits:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print()
    
    # Apply up-sampling to validation set (external validation)
    print("Applying Strategy 1 up-sampling to validation set:")
    X_val_balanced, y_val_balanced = gaussian_noise_upsampling(
        X_val, y_val, noise_std=noise_std, random_state=random_state
    )
    
    return {
        'train': {'X': X_train, 'y': y_train},
        'val_original': {'X': X_val, 'y': y_val},
        'val_balanced': {'X': X_val_balanced, 'y': y_val_balanced},
        'noise_std': noise_std
    }
```

### Step 4: Test Multiple Noise Levels

```python
def test_upsampling_noise_levels(adata, noise_levels=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    Test up-sampling with different noise levels to find optimal setting.
    """
    print("Testing different noise levels for Strategy 1:")
    print("=" * 50)
    
    X = adata.X
    y = adata.obs['class_label'].values
    
    results = {}
    
    for noise_std in noise_levels:
        print(f"\nNoise std = {noise_std}")
        print("-" * 20)
        
        X_balanced, y_balanced = gaussian_noise_upsampling(
            X, y, noise_std=noise_std, random_state=42
        )
        
        results[noise_std] = {
            'X_balanced': X_balanced,
            'y_balanced': y_balanced,
            'total_samples': len(X_balanced),
            'balance_ratio': np.sum(y_balanced == 0) / np.sum(y_balanced == 1)
        }
    
    return results
```

### Step 5: Integration with Fine-tuning Pipeline

When you create your DataLoader for fine-tuning, use the balanced data:

```python
# During fine-tuning setup
balanced_data = prepare_balanced_external_validation_data(adata, noise_std=0.1)

# Create datasets for training
train_dataset = YourDatasetClass(
    balanced_data['train']['X'], 
    balanced_data['train']['y']
)

# Create datasets for external validation (balanced)
val_dataset = YourDatasetClass(
    balanced_data['val_balanced']['X'], 
    balanced_data['val_balanced']['y']
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
```

## Implementation Notes

### 1. **When to Apply**
- Apply **only during fine-tuning** for external validation
- **Do NOT apply during pre-training** - keep pre-training data unchanged
- Apply to the external validation set to balance classes

### 2. **Noise Level Selection**
- Start with `noise_std = 0.1` (conservative)
- Test higher values: 0.5, 1.0, 2.0, 5.0
- Choose based on validation performance
- Higher noise = more variation but potentially less biological plausibility

### 3. **Quality Checks**
- Verify that synthetic samples maintain reasonable abundance ranges
- Check that total abundances remain within expected bounds
- Monitor that no species get unrealistic abundance values

### 4. **Biological Plausibility**
- The method only adds noise to non-zero species (biologically sound)
- Clamps negative values to 0 (maintains non-negativity)
- Optionally normalize to preserve total abundance properties

## Example Usage in Your Workflow

```python
# 1. After loading your data and before fine-tuning
print("=== Implementing Strategy 1: Up-sampling with Gaussian Noise ===")

# 2. Add labels to your data (customize based on your data structure)
adata = add_class_labels_to_adata(adata)

# 3. Test different noise levels
noise_test_results = test_upsampling_noise_levels(adata)

# 4. Choose optimal noise level and prepare balanced data
optimal_noise_std = 0.1  # or based on your testing
balanced_data = prepare_balanced_external_validation_data(
    adata, noise_std=optimal_noise_std
)

# 5. Use balanced data in your fine-tuning pipeline
# (integrate with your existing DataLoader creation)
```

## Next Steps

1. **Add the functions** to your notebook
2. **Implement class labeling** based on your data structure
3. **Test the up-sampling** with different noise levels
4. **Integrate with your fine-tuning pipeline**
5. **Evaluate performance** to choose optimal noise level
6. **Document results** for your analysis

The key is to apply this strategy specifically for external validation during fine-tuning, not during pre-training, and to start conservatively with low noise levels before exploring higher values.