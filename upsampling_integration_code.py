# Integration code for upsampling in microbiomeGPT notebook

# STEP 1: Make sure you have labels for your data
# You need to identify where your labels come from. Common sources:
# - adata.obs['label'] or adata.obs['condition'] 
# - A separate label file that matches your samples
# - From your fine-tuning dataset

# For example, if your labels are disease status:
# y_all = adata.obs['disease_status'].values  # 0 for healthy, 1 for diseased

# STEP 2: Modify your train_test_split to include labels
# Replace your current train_test_split code with:

# First, ensure you have labels
# If you don't have labels in your pretraining data, you might need to:
# 1. Load them from a separate file
# 2. Or create dummy labels for pretraining (all zeros)
# 3. Or only apply upsampling during fine-tuning when you have labels

# Assuming you have labels:
y_all = np.zeros(len(all_counts))  # Replace with your actual labels

# Split both features and labels
(
    train_data,
    valid_data,
    y_train,
    y_valid
) = train_test_split(
    all_counts,
    y_all,
    test_size=0.1,
    shuffle=True,
    random_state=42  # Add for reproducibility
)

# STEP 3: Apply upsampling to training data
train_data_balanced, y_train_balanced, upsampling_stats = gaussian_noise_upsampling(
    train_data,
    y_train,
    noise_std=0.1,
    random_state=42
)

print(f"Original training size: {len(train_data)}")
print(f"Balanced training size: {len(train_data_balanced)}")
print(f"Upsampling statistics: {upsampling_stats}")

# STEP 4: Update tokenization to use balanced data
# Replace tokenized_train = tokenize_and_pad_batch(train_data_balanced, ...)
# The rest of your code remains the same

tokenized_train = tokenize_and_pad_batch(
    train_data_balanced,  # Now using balanced data
    organism_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=config.append_cls,
    include_zero_gene=config.include_zero_gene,
)

# Validation data tokenization remains unchanged
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

logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)