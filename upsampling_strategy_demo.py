#!/usr/bin/env python3
"""
Strategy 1: Up-sampling with Gaussian Noise for Class Balancing

This script demonstrates how to implement the up-sampling strategy for balancing 
classes in microbiome data for external validation during fine-tuning.

Based on the paper's method description, this strategy:
1. Generates enough samples for the minority group to make the number of samples 
   in that group equal to that of the majority group
2. Uses Gaussian noise added to non-zero abundance species
3. Starts with std=0.1 but can explore increasing up to std=5.0
"""

import pandas as pd
import numpy as np
from copy import deepcopy
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def gaussian_noise_upsampling(
    abundance_data: np.ndarray,
    labels: np.ndarray,
    noise_std: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
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
        
        # Optionally normalize to maintain relative abundance properties
        # (uncomment if you want to preserve total abundance)
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


def test_multiple_noise_levels(
    abundance_data: np.ndarray,
    labels: np.ndarray,
    noise_std_list: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    random_state: int = 42
) -> dict:
    """
    Test up-sampling with different noise levels.
    
    Args:
        abundance_data: Array of shape (n_samples, n_species) with relative abundances
        labels: Array of shape (n_samples,) with class labels (0=healthy, 1=diseased)
        noise_std_list: List of standard deviations to test for Gaussian noise
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing balanced datasets for each noise level
    """
    print("=== Strategy 1: Up-sampling with Gaussian Noise ===")
    print("Testing different noise levels...")
    print()
    
    balanced_datasets = {}
    
    for std in noise_std_list:
        print(f"Testing noise std = {std}")
        X_balanced, y_balanced = gaussian_noise_upsampling(
            abundance_data, labels, noise_std=std, random_state=random_state
        )
        balanced_datasets[f'std_{std}'] = {
            'X': X_balanced,
            'y': y_balanced,
            'noise_std': std
        }
        print("-" * 50)
    
    return balanced_datasets


def plot_noise_impact(original_data, synthetic_data, species_names=None, n_species_show=10):
    """
    Plot the impact of noise on synthetic samples compared to original data.
    
    Args:
        original_data: Original abundance data
        synthetic_data: Synthetic data generated with noise
        species_names: Names of species (optional)
        n_species_show: Number of top species to show in plots
    """
    # Calculate mean abundances
    orig_means = np.mean(original_data, axis=0)
    synth_means = np.mean(synthetic_data, axis=0)
    
    # Get top species by abundance
    top_species_idx = np.argsort(orig_means)[-n_species_show:]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Mean abundance comparison
    axes[0, 0].scatter(orig_means[top_species_idx], synth_means[top_species_idx])
    axes[0, 0].plot([0, orig_means[top_species_idx].max()], 
                    [0, orig_means[top_species_idx].max()], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('Original Mean Abundance')
    axes[0, 0].set_ylabel('Synthetic Mean Abundance')
    axes[0, 0].set_title('Mean Abundance Comparison (Top Species)')
    
    # Plot 2: Distribution of differences
    differences = synth_means - orig_means
    axes[0, 1].hist(differences, bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Abundance Difference (Synthetic - Original)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Abundance Differences')
    
    # Plot 3: Sample variance comparison
    orig_vars = np.var(original_data, axis=0)
    synth_vars = np.var(synthetic_data, axis=0)
    axes[1, 0].scatter(orig_vars[top_species_idx], synth_vars[top_species_idx])
    axes[1, 0].plot([0, orig_vars[top_species_idx].max()], 
                    [0, orig_vars[top_species_idx].max()], 'r--', alpha=0.5)
    axes[1, 0].set_xlabel('Original Variance')
    axes[1, 0].set_ylabel('Synthetic Variance')
    axes[1, 0].set_title('Variance Comparison (Top Species)')
    
    # Plot 4: Relative abundance distributions for a sample species
    top_species = top_species_idx[-1]  # Most abundant species
    axes[1, 1].hist(original_data[:, top_species], alpha=0.5, label='Original', bins=30)
    axes[1, 1].hist(synthetic_data[:, top_species], alpha=0.5, label='Synthetic', bins=30)
    axes[1, 1].set_xlabel('Relative Abundance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Distribution for Most Abundant Species')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


def demonstrate_upsampling_workflow():
    """
    Demonstrate the complete up-sampling workflow with example data.
    """
    print("=" * 60)
    print("STRATEGY 1: UP-SAMPLING WITH GAUSSIAN NOISE DEMONSTRATION")
    print("=" * 60)
    print()
    
    print("STEP 1: Create example imbalanced dataset")
    print("-" * 40)
    
    # Create example data (this would be replaced with your actual data)
    np.random.seed(42)
    
    # Simulate microbiome data with class imbalance
    n_healthy = 1000  # majority class
    n_diseased = 400  # minority class  
    n_species = 50
    
    # Healthy samples (tend to have more diverse microbiome)
    healthy_data = np.random.dirichlet(np.ones(n_species) * 2, n_healthy) * 100
    healthy_labels = np.zeros(n_healthy)
    
    # Diseased samples (less diverse, some species more abundant)
    diseased_alpha = np.ones(n_species)
    diseased_alpha[:10] = 5  # First 10 species more abundant in disease
    diseased_data = np.random.dirichlet(diseased_alpha, n_diseased) * 100
    diseased_labels = np.ones(n_diseased)
    
    # Combine data
    X = np.vstack([healthy_data, diseased_data])
    y = np.hstack([healthy_labels, diseased_labels])
    
    print(f"Original dataset:")
    print(f"  Total samples: {len(X)}")
    print(f"  Healthy (class 0): {np.sum(y == 0)} samples")
    print(f"  Diseased (class 1): {np.sum(y == 1)} samples")
    print(f"  Imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}")
    print()
    
    print("STEP 2: Apply up-sampling with different noise levels")
    print("-" * 50)
    
    # Test different noise levels
    noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = test_multiple_noise_levels(X, y, noise_levels)
    
    print("STEP 3: Analyze results")
    print("-" * 25)
    
    for noise_level in noise_levels:
        key = f'std_{noise_level}'
        X_balanced = results[key]['X']
        y_balanced = results[key]['y']
        
        # Check balance
        unique, counts = np.unique(y_balanced, return_counts=True)
        print(f"Noise std {noise_level}: Balance ratio = {counts[0] / counts[1]:.2f}")
    
    print()
    print("STEP 4: Recommendations for implementation")
    print("-" * 45)
    print("1. Start with noise_std = 0.1 for conservative augmentation")
    print("2. Increase noise_std gradually (0.5, 1.0, 2.0, 5.0) if needed")
    print("3. Evaluate model performance with each noise level")
    print("4. Choose the noise level that gives best validation performance")
    print("5. Apply only to external validation set, not training set")
    print()
    print("For your microbiomeGPT fine-tuning:")
    print("- Apply this before creating DataLoader for external validation")
    print("- Keep original training data unchanged")
    print("- Monitor that synthetic samples maintain biological plausibility")


if __name__ == "__main__":
    demonstrate_upsampling_workflow()