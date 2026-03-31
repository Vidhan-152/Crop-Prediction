from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS
from .transformers import DomainFeatureGenerator


sns.set_theme(style="whitegrid")


def save_eda_plots(df: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)

    for feature in FEATURE_COLUMNS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df[feature], kde=True, ax=axes[0], color="#2E8B57")
        axes[0].set_title(f"Distribution of {feature}")
        sns.boxplot(data=df, x="label", y=feature, ax=axes[1], color="#87CEEB")
        axes[1].tick_params(axis="x", rotation=90)
        axes[1].set_title(f"{feature} by crop")
        fig.tight_layout()
        fig.savefig(output_dir / f"dist_{feature}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pairplot_sample = df.sample(min(400, len(df)), random_state=42)
    pair_grid = sns.pairplot(pairplot_sample, vars=FEATURE_COLUMNS, hue="label", corner=True, plot_kws={"alpha": 0.65, "s": 20})
    pair_grid.savefig(output_dir / "pairplot_sample.png", dpi=180)
    plt.close("all")


def save_pca_and_cluster_plots(df: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    X = DomainFeatureGenerator().fit_transform(df[FEATURE_COLUMNS])
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
    pca_df["label"] = df["label"].values

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="label", s=45, ax=ax, legend=False)
    ax.set_title("PCA Projection of Crop Classes")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"], s=12, alpha=0.7)
    ax.set_title("3D PCA Scatter")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_3d_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    kmeans = KMeans(n_clusters=df["label"].nunique(), random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, cluster_labels)
    ari = adjusted_rand_score(df["label"], cluster_labels)
    pd.DataFrame(
        [{"algorithm": "kmeans", "silhouette_score": float(silhouette), "adjusted_rand_index": float(ari)}]
    ).to_csv(output_dir / "cluster_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(pca_df["PC1"], pca_df["PC2"], c=cluster_labels, cmap="tab20", s=35)
    ax.set_title("K-Means Clusters in PCA Space")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_clusters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    linkage_matrix = linkage(X_scaled[: min(500, len(X_scaled))], method="ward")
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(linkage_matrix, ax=ax, no_labels=True)
    ax.set_title("Hierarchical Clustering Dendrogram")
    fig.tight_layout()
    fig.savefig(output_dir / "hierarchical_dendrogram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
