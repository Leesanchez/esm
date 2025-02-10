import torch
import esm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define output directory for embeddings
output_dir = os.path.expanduser("~/Desktop/ESM_Project/ESM_Predictions")
os.makedirs(output_dir, exist_ok=True)

# Load ESM-2 model
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # Set model to evaluation mode

# Example protein sequence
protein_name = "test_protein"
sequence = "MKTWTEAI"  # Replace this with your own sequence
data = [(protein_name, sequence)]

# Convert sequence to tokens
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Generate embeddings
print(f"Generating embeddings for {protein_name}...")
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33])  # Get representations from layer 33
    embedding = results["representations"][33].numpy()  # Convert to NumPy array

# Save embedding
output_path = os.path.join(output_dir, f"{protein_name}_embedding.npy")
np.save(output_path, embedding)
print(f"✅ Protein embedding saved at: {output_path}")

# ---- PCA for Visualization ----
print("Performing PCA to visualize embeddings...")

# Flatten the 3D embedding into 2D for PCA
embedding_reshaped = embedding.reshape(embedding.shape[0], -1)

# Check if PCA is possible
n_samples, n_features = embedding_reshaped.shape
if n_samples < 2:
    print("⚠️ Not enough samples for PCA. Skipping PCA visualization.")
else:
    # Perform PCA
    pca = PCA(n_components=min(2, n_samples))  # Ensure valid PCA components
    embedding_pca = pca.fit_transform(embedding_reshaped)

    # Save PCA visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], color='blue', label=protein_name)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA of {protein_name} Embedding")
    plt.legend()
    pca_plot_path = os.path.join(output_dir, f"{protein_name}_pca_plot.png")
    plt.savefig(pca_plot_path)
    plt.close()
    print(f"✅ PCA plot saved at: {pca_plot_path}")

print("✅ All tasks completed successfully!")