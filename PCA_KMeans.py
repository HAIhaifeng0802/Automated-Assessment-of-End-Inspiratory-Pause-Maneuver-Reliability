"""
A two-stage dynamic clustering approach for high-dimensional waveform data with:
1. Dimensionality reduction using PCA
2. Initial coarse clustering with MiniBatchKMeans
3. Dynamic cluster expansion based on distance thresholds
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import time
import joblib
import os
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess(data_path: str) -> tuple:
    """Load and preprocess waveform data.
    
    Args:
        data_path: Directory containing 'all_waveforms.npy' and 'metadata.csv'
        
    Returns:
        Tuple of (flattened_waveforms, metadata_df)
    """
    print("Loading data...")
    waveforms = np.load(os.path.join(data_path, "all_waveforms.npy"))  # Shape: (n_samples, 300, 3)
    metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))
    
    # Flatten waveforms to (n_samples × 900)
    X = waveforms.reshape(waveforms.shape[0], -1)
    return X, metadata

def perform_pca(X: np.ndarray, 
               variance_threshold: float = 0.99, 
               plot: bool = True) -> tuple:
    """Dimensionality reduction with PCA.
    
    Args:
        X: Input data matrix
        variance_threshold: Minimum explained variance to retain
        plot: Whether to generate explained variance plot
        
    Returns:
        Tuple of (transformed_data, pca_model, scaler)
    """
    print("\n=== PCA Dimensionality Reduction ===")
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Automated PCA with variance threshold
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {pca.n_components_} (Explained variance: {np.sum(pca.explained_variance_ratio_):.2%})")
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), 'b-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.axhline(y=variance_threshold, color='r', linestyle='--')
        plt.title('PCA Explained Variance')
        plt.savefig(os.path.join(data_path, 'pca_variance.png'))
        plt.close()
    
    return X_pca, pca, scaler

def min_distance_to_centroids(X: np.ndarray, 
                            centroids: np.ndarray, 
                            chunk_size: int = 1000) -> np.ndarray:
    """Compute minimum distances to centroids with memory optimization.
    
    Args:
        X: Data matrix
        centroids: Cluster centers
        chunk_size: Processing batch size
        
    Returns:
        Array of minimum distances for each sample
    """
    min_dists = np.zeros(len(X))
    for i in tqdm(range(0, len(X), chunk_size), desc="Computing distances"):
        batch = X[i:i+chunk_size]
        dists = np.sqrt(((batch[:, np.newaxis] - centroids)**2).sum(axis=2))
        min_dists[i:i+chunk_size] = np.min(dists, axis=1)
    return min_dists

def dynamic_two_stage_clustering(X_pca: np.ndarray,
                               initial_clusters: int = 5337,
                               threshold_quantile: float = 0.95,
                               batch_size: int = 5000,
                               distance_chunk_size: int = 1000) -> tuple:
    """Two-stage dynamic clustering with automatic cluster expansion.
    
    Args:
        X_pca: PCA-transformed data
        initial_clusters: Number of initial clusters
        threshold_quantile: Percentile for distance threshold
        batch_size: Processing batch size
        distance_chunk_size: Memory-optimized distance computation chunk size
        
    Returns:
        Tuple of (cluster_labels, final_centroids)
    """
    # --- Stage 1: Coarse Clustering ---
    print("\n=== Stage 1: Initial Coarse Clustering ===")
    np.random.seed(42)
    idx = np.random.permutation(len(X_pca))
    split = len(X_pca) // 2
    X_train = X_pca[idx[:split]]
    
    # Mini-batch K-Means clustering
    mbk = MiniBatchKMeans(
        n_clusters=initial_clusters,
        batch_size=batch_size,
        init='k-means++',
        max_iter=100,
        random_state=42
    ).fit(X_train)
    centroids = mbk.cluster_centers_
    
    # --- Stage 2: Dynamic Expansion ---
    print("\n=== Stage 2: Dynamic Cluster Expansion ===")
    X_extend = X_pca[idx[split:]]
    
    # Calculate dynamic threshold
    train_dists = min_distance_to_centroids(X_train, centroids, distance_chunk_size)
    threshold = np.percentile(train_dists, threshold_quantile * 100)
    
    # Initialize dynamic expansion
    dynamic_centroids = list(centroids)
    extend_labels = np.zeros(len(X_extend), dtype=int)
    
    # Process extension set in batches
    for i in tqdm(range(0, len(X_extend), batch_size), desc="Dynamic expansion"):
        batch = X_extend[i:i+batch_size]
        
        # Compute distances using current centroids
        distances = np.zeros((len(batch), len(dynamic_centroids)))
        for j in range(0, len(batch), distance_chunk_size):
            chunk = batch[j:j+distance_chunk_size]
            distances[j:j+distance_chunk_size] = np.sqrt(
                ((chunk[:, np.newaxis] - np.array(dynamic_centroids))**2).sum(axis=2))
        
        min_distances = np.min(distances, axis=1)
        nearest_indices = np.argmin(distances, axis=1)
        
        # Identify samples needing new clusters
        new_samples = batch[min_distances > threshold]
        if len(new_samples) > 0:
            dynamic_centroids.extend(new_samples)
            new_distances = np.sqrt(((batch[:, np.newaxis] - np.array(new_samples))**2).sum(axis=2))
            updated_min_indices = np.argmin(np.hstack([distances, new_distances]), axis=1)
            nearest_indices = updated_min_indices
        
        extend_labels[i:i+batch_size] = nearest_indices
    
    final_centroids = np.array(dynamic_centroids)
    print(f"Final cluster count: {len(final_centroids)}")
    
    final_labels = np.zeros(len(X_pca), dtype=int)
    final_labels[idx[:split]] = mbk.labels_  # Stage 1 samples
    final_labels[idx[split:]] = extend_labels  # Stage 2 samples
    
    return final_labels, final_centroids

def analyze_results(labels: np.ndarray,
                   metadata: pd.DataFrame,
                   centroids: np.ndarray,
                   output_path: str) -> None:
    """Analyze and visualize clustering results.
    
    Args:
        labels: Cluster assignments
        metadata: Sample metadata
        centroids: Final cluster centers
        output_path: Output directory
    """
    print("\n=== Results Analysis ===")
    metadata['cluster_label'] = labels
    cluster_counts = metadata['cluster_label'].value_counts()
    
    # Basic statistics
    print(f"Total clusters: {len(cluster_counts)}")
    print(f"Cluster sizes - Max: {cluster_counts.max()}, Min: {cluster_counts.min()}, Mean: {cluster_counts.mean():.1f}")
    
    # Save distribution plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(cluster_counts, bins=50, log=True)
    plt.title('Cluster Size Distribution (Log Scale)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(cluster_counts)), sorted(cluster_counts, reverse=True), s=1)
    plt.title('Cluster Size Rank')
    plt.savefig(os.path.join(output_path, 'cluster_distribution.png'))
    plt.close()
    
    # Save outputs
    metadata.to_csv(os.path.join(output_path, 'clustered_metadata.csv'), index=False)
    np.save(os.path.join(output_path, 'cluster_labels.npy'), labels)
    np.save(os.path.join(output_path, 'final_centroids.npy'), centroids)

if __name__ == "__main__":
    # Configuration
    data_path = r".\Data"
    output_path = os.path.join(data_path, "results")
    os.makedirs(output_path, exist_ok=True)
    
    # Pipeline execution
    X, metadata = load_and_preprocess(data_path)
    X_pca, pca, scaler = perform_pca(X)
    cluster_labels, final_centroids = dynamic_two_stage_clustering(
        X_pca,
        initial_clusters=5337,
        threshold_quantile=0.99,
        batch_size=5000,
        distance_chunk_size=1000
    )
    analyze_results(cluster_labels, metadata, final_centroids, output_path)
    
    # Save models
    joblib.dump(pca, os.path.join(output_path, 'pca_model.joblib'))
    joblib.dump(scaler, os.path.join(output_path, 'scaler.joblib'))
    
    print("\n=== Processing Complete ===")
    print(f"Results saved to: {output_path}")