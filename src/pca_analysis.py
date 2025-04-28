"""
PCA analysis module for dimensionality reduction and visualization.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from .config import VIS_CONFIG, OUTPUTS_DIR

# Get logger instead of basic config
logger = logging.getLogger(__name__)

class PCAAnalyzer:
    def __init__(self, n_components=2):
        """Initialize PCA analyzer."""
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def fit_transform(self, features):
        """Fit PCA and transform features."""
        self.logger.info("Performing PCA analysis")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply PCA
        pca_features = self.pca.fit_transform(scaled_features)
        
        return pca_features
        
    def plot_pca(self, pca_features, labels, class_names, title="PCA Visualization"):
        """Plot PCA results."""
        self.logger.info("Plotting PCA visualization")
        
        plt.figure(figsize=VIS_CONFIG["plot_size"])
        
        # Create scatter plot
        scatter = plt.scatter(
            pca_features[:, 0],
            pca_features[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6
        )
        
        # Add legend
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=class_names,
            title="Classes"
        )
        
        # Add explained variance ratio
        explained_variance = self.pca.explained_variance_ratio_
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "pca_visualization.png", dpi=VIS_CONFIG["dpi"])
        plt.close()
        
    def plot_variance_explained(self):
        """Plot cumulative explained variance."""
        self.logger.info("Plotting explained variance")
        
        plt.figure(figsize=VIS_CONFIG["plot_size"])
        
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        
        # Plot
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "pca_variance.png", dpi=VIS_CONFIG["dpi"])
        plt.close()
        
    def get_feature_importance(self, feature_names):
        """Get feature importance from PCA components."""
        self.logger.info("Calculating feature importance")
        
        # Get absolute values of PCA components
        components = np.abs(self.pca.components_)
        
        # Calculate feature importance
        feature_importance = np.sum(components, axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        return {
            'features': [feature_names[i] for i in sorted_idx],
            'importance': feature_importance[sorted_idx]
        } 