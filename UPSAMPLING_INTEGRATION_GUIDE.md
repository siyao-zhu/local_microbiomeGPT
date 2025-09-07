# Upsampling Integration Guide for MicrobiomeGPT

## Overview
This guide explains how to integrate the Gaussian noise upsampling strategy into your microbiomeGPT pretraining pipeline to balance minority classes in the training data.

## Key Changes Required

### 1. **Labels are Required**
The upsampling strategy requires class labels (y) to identify minority and majority classes. You need to:
- Identify where your labels come from (e.g., disease status, health condition)
- Ensure labels are available during the pretraining phase
- If labels aren't available during pretraining, consider applying upsampling only during fine-tuning

### 2. **Modified Data Splitting**
Change your current `train_test_split` to split both features AND labels:

```python
# Original code (splits only features):
(train_data, valid_data) = train_test_split(all_counts, test_size=0.1, shuffle=True)

# Updated code (splits features and labels together):
# First, get your labels (example):
y_all = adata.obs['disease_status'].values  # or your label column

# Then split both:
(train_data, valid_data, y_train, y_valid) = train_test_split(
    all_counts, 
    y_all,
    test_size=0.1, 
    shuffle=True,
    random_state=42  # for reproducibility
)
```

### 3. **Apply Upsampling to Training Data**
Add the upsampling step AFTER splitting but BEFORE tokenization:

```python
# Apply Gaussian noise upsampling to training data only
train_data_balanced, y_train_balanced, upsampling_stats = gaussian_noise_upsampling(
    train_data,
    y_train,
    noise_std=0.1,
    random_state=42
)

# Print statistics to verify balancing
print(f"Original training size: {len(train_data)}")
print(f"Balanced training size: {len(train_data_balanced)}")
print(f"Class distribution: {upsampling_stats['final_distribution']}")
```

### 4. **Update Tokenization**
Use the balanced training data for tokenization:

```python
# Change from:
tokenized_train = tokenize_and_pad_batch(
    train_data_balanced,  # This variable didn't exist before
    ...
)

# To (after upsampling):
tokenized_train = tokenize_and_pad_batch(
    train_data_balanced,  # Now properly defined from upsampling
    organism_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=config.append_cls,
    include_zero_gene=config.include_zero_gene,
)
```

## Complete Integration Example

```python
# 1. Load your data and labels
input_layer_key = "X_binned"
all_counts = adata.layers[input_layer_key].A if issparse(adata.layers[input_layer_key]) else adata.layers[input_layer_key]
organisms = adata.var["organism_name"].tolist()

# Get labels (adjust based on your data)
y_all = adata.obs['your_label_column'].values  # CHANGE THIS

# 2. Split data and labels together
(train_data, valid_data, y_train, y_valid) = train_test_split(
    all_counts, 
    y_all,
    test_size=0.1, 
    shuffle=True,
    random_state=42
)

# 3. Apply upsampling to training data
train_data_balanced, y_train_balanced, upsampling_stats = gaussian_noise_upsampling(
    train_data,
    y_train,
    noise_std=0.1,
    random_state=42
)

# 4. Continue with tokenization using balanced data
organism_ids = np.array(vocab(organisms), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data_balanced,  # Balanced data
    organism_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=config.append_cls,
    include_zero_gene=config.include_zero_gene,
)

tokenized_valid = tokenize_and_pad_batch(
    valid_data,  # Validation data remains unbalanced
    organism_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=config.append_cls,
    include_zero_gene=config.include_zero_gene,
)
```

## Important Notes

1. **Binned Data Handling**: The upsampling function has been modified to handle binned data:
   - Synthetic samples are rounded to integers
   - Values are clipped to stay within the bin range (0 to n_bins-1)

2. **Validation Set**: The validation set is NOT upsampled - only the training set is balanced

3. **Label Availability**: If you don't have labels during pretraining:
   - Option 1: Skip upsampling during pretraining, apply it during fine-tuning
   - Option 2: Create synthetic labels based on some criteria
   - Option 3: Use unsupervised pretraining without upsampling

4. **Memory Considerations**: Upsampling increases the size of your training data, which may impact memory usage

## Verification Steps

After integration, verify:
1. Check the class distribution before and after upsampling
2. Confirm the training set size has increased appropriately
3. Ensure the validation set remains unchanged
4. Monitor training to ensure the model learns from both original and synthetic samples

## Files Created
- `apply_upsampling_to_training.py`: Demonstration script with the full implementation
- `upsampling_integration_code.py`: The exact code to integrate into your notebook
- This guide: Step-by-step integration instructions