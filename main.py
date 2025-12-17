import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import umap
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def save_3d_points_to_csv(
    embeddings_3d: np.ndarray,
    labels: np.ndarray,
    out_csv_path: str = "ms1mv2_embeddings_3d.csv",
    include_label: bool = True,
    include_sample_index: np.ndarray | None = None,
    float_format: str = "%.6f"
):
    """
    Save final 3D points (and optional labels / sample indices) to CSV.

    Columns:
      - x, y, z
      - label (optional)
      - sample_index (optional; index into original arrays before sampling)
    """
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    # Ensure shapes are consistent
    if embeddings_3d.ndim != 2 or embeddings_3d.shape[1] != 3:
        raise ValueError(f"embeddings_3d must be shape (N, 3), got {embeddings_3d.shape}")
    if len(labels) != len(embeddings_3d):
        raise ValueError(f"labels length {len(labels)} != embeddings_3d length {len(embeddings_3d)}")
    if include_sample_index is not None and len(include_sample_index) != len(embeddings_3d):
        raise ValueError("include_sample_index must have the same length as embeddings_3d")

    header = ["x", "y", "z"]
    if include_label:
        header.append("label")
    if include_sample_index is not None:
        header.append("sample_index")

    # Write CSV
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Write rows (avoid building a huge intermediate array)
        for i in range(len(embeddings_3d)):
            x, y, z = embeddings_3d[i]
            row = [
                float_format % x,
                float_format % y,
                float_format % z,
            ]
            if include_label:
                row.append(int(labels[i]))
            if include_sample_index is not None:
                row.append(int(include_sample_index[i]))
            writer.writerow(row)

    print(f"Saved 3D points to CSV: '{out_csv_path}' ({len(embeddings_3d):,} rows)")


def create_3d_face_visualization(
    embeddings_path='embedding_ir101_adaface/embedding_ms1mv2_embeddings.npy',
    labels_path='embedding_ir101_adaface/embedding_ms1mv2_labels.npy',
    n_samples=30000,
    n_classes_to_show=50,
    reduction_method='umap',
    interactive=True,
    save_csv=True,
    csv_path="ms1mv2_embeddings_3d.csv",
    csv_include_label=True,
    csv_include_sample_index=True 
):
    """
    3D visualization for MS1MV2 face embeddings with multiple viewing options
    """
    
    print("Loading data...")
    # Load with memory mapping to handle large files
    embeddings = np.load(embeddings_path, mmap_mode='r')
    labels = np.load(labels_path)
    
    print(f"Original dataset: {embeddings.shape[0]:,} samples, {len(np.unique(labels)):,} identities")
    
    # 1. Intelligent Sampling
    print(f"Sampling {n_samples:,} points...")
    
    # Strategy: Sample from diverse identities
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_samples < n_classes:
        # If sampling fewer points than classes, take 1-2 from each of some classes
        selected_classes = np.random.choice(unique_labels, n_samples//2, replace=False)
        indices = []
        for cls in selected_classes:
            cls_indices = np.where(labels == cls)[0]
            if len(cls_indices) > 0:
                indices.append(np.random.choice(cls_indices, min(2, len(cls_indices)), replace=False))
        indices = np.concatenate(indices)[:n_samples]
    else:
        # Sample proportionally
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
    
    embeddings_sample = embeddings[indices]
    labels_sample = labels[indices]
    
    # 2. Normalize (critical for face embeddings)
    print("Normalizing embeddings...")
    embeddings_norm = embeddings_sample / np.linalg.norm(embeddings_sample, axis=1, keepdims=True)
    
    # 3. Dimensionality Reduction to 3D
    print(f"Reducing to 3D using {reduction_method}...")
    
    if reduction_method.lower() == 'pca':
        # Fast PCA
        pca = PCA(n_components=3, random_state=42)
        embeddings_3d = pca.fit_transform(embeddings_norm)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
        
    elif reduction_method.lower() == 'umap':
        # UMAP with cosine distance (best for face embeddings)
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,  # Adjust based on density
            min_dist=0.1,    # Controls clustering (0.0-1.0)
            metric='cosine',
            random_state=42,
            low_memory=True if n_samples > 10000 else False
        )
        embeddings_3d = reducer.fit_transform(embeddings_norm)
        
    elif reduction_method.lower() == 'pca+umap':
        # Two-stage: PCA to 50D then UMAP to 3D
        pca50 = PCA(n_components=min(50, embeddings_norm.shape[1]))
        embeddings_50d = pca50.fit_transform(embeddings_norm)
        
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        embeddings_3d = reducer.fit_transform(embeddings_50d)

    # ---- SAVE FINAL 3D POINTS ----
    if save_csv:
        save_3d_points_to_csv(
            embeddings_3d=embeddings_3d,
            labels=labels_sample,
            out_csv_path=csv_path,
            include_label=csv_include_label,
            include_sample_index=indices if csv_include_sample_index else None,
            float_format="%.6f"
        )

    # 4. Create Visualization
    if interactive:
        return create_interactive_3d_plotly(embeddings_3d, labels_sample, n_classes_to_show)
    else:
        return create_static_3d_matplotlib(embeddings_3d, labels_sample, n_classes_to_show)


def create_interactive_3d_plotly(embeddings_3d, labels, n_classes_to_show=50):
    """
    Interactive 3D visualization with Plotly
    """
    print("Creating interactive 3D visualization with Plotly...")
    
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    
    # Sort by count (show most populated identities)
    sorted_indices = np.argsort(label_counts)[::-1]
    unique_labels = unique_labels[sorted_indices]
    label_counts = label_counts[sorted_indices]
    
    # Take top N classes
    if len(unique_labels) > n_classes_to_show:
        top_labels = unique_labels[:n_classes_to_show]
    else:
        top_labels = unique_labels
    
    # Create color palette
    colors = plt.cm.tab20c(np.linspace(0, 1, len(top_labels)))
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each identity
    for i, label in enumerate(top_labels):
        mask = labels == label
        count = np.sum(mask)
        
        if count > 0:
            color = colors[i]
            rgba_str = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'
            
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=rgba_str,
                    opacity=0.7,
                    line=dict(width=0)  # No border for cleaner look
                ),
                name=f'ID {label} (n={count})',
                hovertemplate=(
                    f'Identity: {label}<br>'
                    'X: %{x:.3f}<br>'
                    'Y: %{y:.3f}<br>'
                    'Z: %{z:.3f}<br>'
                    '<extra></extra>'
                )
            ))
    
    # Add a trace for "Other" identities if we have more than n_classes_to_show
    if len(unique_labels) > n_classes_to_show:
        other_mask = ~np.isin(labels, top_labels)
        if np.sum(other_mask) > 0:
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[other_mask, 0],
                y=embeddings_3d[other_mask, 1],
                z=embeddings_3d[other_mask, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='rgba(150, 150, 150, 0.3)',  # Gray for others
                    opacity=0.3
                ),
                name=f'Other identities ({len(unique_labels)-n_classes_to_show})',
                hovertemplate='Other identities<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'MS1MV2 Face Embeddings 3D Visualization<br><sub>{len(embeddings_3d):,} samples, {len(np.unique(labels)):,} identities shown</sub>',
            x=0.5,
            y=0.95,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(
                title='Dimension 1',
                backgroundcolor='rgba(240, 240, 240, 0.1)',
                gridcolor='white',
                showbackground=True
            ),
            yaxis=dict(
                title='Dimension 2',
                backgroundcolor='rgba(240, 240, 240, 0.1)',
                gridcolor='white',
                showbackground=True
            ),
            zaxis=dict(
                title='Dimension 3',
                backgroundcolor='rgba(240, 240, 240, 0.1)',
                gridcolor='white',
                showbackground=True
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            title="Identities",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=1200,
        height=800,
        margin=dict(r=200, l=50, b=50, t=100)  # Make room for legend
    )
    
    # Add dropdown for view angles
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.5}}],
                        label="Default",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 2, "y": 0, "z": 0}}],
                        label="X View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 2, "z": 0}}],
                        label="Y View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2}}],
                        label="Z View",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=0.95,
                yanchor="top"
            )
        ]
    )
    
    # Save to HTML for sharing
    fig.write_html("ms1mv2_face_embeddings_3d_interactive.html")
    print("Saved interactive visualization to 'ms1mv2_face_embeddings_3d_interactive.html'")
    
    fig.show()
    return fig


def create_static_3d_matplotlib(embeddings_3d, labels, n_classes_to_show=30):
    """
    Static 3D visualization with Matplotlib
    """
    print("Creating static 3D visualization with Matplotlib...")
    
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    
    # Sort by count and take top N
    sorted_indices = np.argsort(label_counts)[::-1]
    unique_labels = unique_labels[sorted_indices]
    label_counts = label_counts[sorted_indices]
    
    if len(unique_labels) > n_classes_to_show:
        top_labels = unique_labels[:n_classes_to_show]
    else:
        top_labels = unique_labels
    
    # Create figure with 2 subplots: colored by identity and colored by density
    fig = plt.figure(figsize=(20, 8))
    
    # Subplot 1: Colored by identity
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create color map
    colors = plt.cm.tab20c(np.linspace(0, 1, len(top_labels)))
    
    for i, label in enumerate(top_labels):
        mask = labels == label
        if np.sum(mask) > 0:
            ax1.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c=[colors[i]],
                s=10,
                alpha=0.7,
                label=f'ID {label} (n={np.sum(mask)})'
            )
    
    # Plot "other" identities in gray
    other_mask = ~np.isin(labels, top_labels)
    if np.sum(other_mask) > 0:
        ax1.scatter(
            embeddings_3d[other_mask, 0],
            embeddings_3d[other_mask, 1],
            embeddings_3d[other_mask, 2],
            c='gray',
            s=3,
            alpha=0.2,
            label=f'Other ({len(unique_labels)-len(top_labels)} ids)'
        )
    
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_zlabel('Dimension 3')
    ax1.set_title(f'MS1MV2 Face Embeddings (Colored by Identity)\n{len(embeddings_3d):,} samples')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Colored by density (using z-axis as color)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Color by z-value (third dimension)
    scatter = ax2.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=embeddings_3d[:, 2],  # Color by z-value
        cmap='viridis',
        s=5,
        alpha=0.6
    )
    
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.set_zlabel('Dimension 3')
    ax2.set_title(f'MS1MV2 Face Embeddings (Colored by Density)\n{len(embeddings_3d):,} samples')
    plt.colorbar(scatter, ax=ax2, label='Z-value (Dimension 3)', pad=0.1)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'MS1MV2 Dataset: {len(embeddings_3d):,} samples from {len(np.unique(labels)):,} identities', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('ms1mv2_face_embeddings_3d_static.png', dpi=150, bbox_inches='tight')
    print("Saved static visualization to 'ms1mv2_face_embeddings_3d_static.png'")
    
    plt.show()
    return fig


def visualize_identity_clusters(embeddings_3d, labels, specific_identities=None, n_per_identity=10):
    """
    Visualize specific identities or clusters in detail
    """
    if specific_identities is None:
        # Pick 5 random identities
        unique_labels = np.unique(labels)
        specific_identities = np.random.choice(unique_labels, 5, replace=False)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(specific_identities)))
    
    for i, identity in enumerate(specific_identities):
        mask = labels == identity
        identity_samples = embeddings_3d[mask]
        
        # If too many samples, take random subset
        if len(identity_samples) > n_per_identity:
            identity_samples = identity_samples[np.random.choice(
                len(identity_samples), n_per_identity, replace=False
            )]
        
        ax.scatter(
            identity_samples[:, 0],
            identity_samples[:, 1],
            identity_samples[:, 2],
            c=[colors[i]],
            s=50,
            alpha=0.8,
            label=f'Identity {identity}',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add centroid
        centroid = np.mean(identity_samples, axis=0)
        ax.scatter(
            centroid[0], centroid[1], centroid[2],
            c=[colors[i]],
            s=200,
            marker='X',
            edgecolors='black',
            linewidth=1
        )
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(f'Detailed View of {len(specific_identities)} Identities\nwith Centroids (X markers)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_3d_animation(embeddings_3d, labels, output_path='3d_rotation.gif'):
    """
    Create a rotating 3D animation (requires imagemagick or pillow)
    """
    from matplotlib.animation import FuncAnimation
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Simple scatter (color by label for speed)
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=labels % 20,  # Use modulo for limited colors
        cmap='tab20',
        s=1,
        alpha=0.6
    )
    
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.set_title('MS1MV2 Face Embeddings 3D Rotation')
    
    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return scatter,
    
    # Create animation (slow for large datasets)
    print("Creating animation (this may take a while)...")
    anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=15, dpi=100)
    print(f"Animation saved to {output_path}")
    
    plt.close()
    return anim


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    # Example 1: Basic interactive visualization
    fig = create_3d_face_visualization(
        n_samples=10000,           # Start with 20K samples
        n_classes_to_show=500,      # Show 40 identities in color
        reduction_method='umap',   # Use UMAP (best for face embeddings)
        interactive=True          # Use Plotly for interactivity
    )

    # Example 2: Static visualization for publication
    # fig = create_3d_face_visualization(
    #     n_samples=50000,
    #     n_classes_to_show=30,
    #     reduction_method='pca',  # Faster for large samples
    #     interactive=False       # Use Matplotlib
    # )

    # Example 3: Load pre-computed 3D embeddings and visualize
    # embeddings_3d = np.load('embeddings_3d_umap.npy')
    # labels = np.load('labels.npy')
    # create_interactive_3d_plotly(embeddings_3d, labels)

    # Example 4: Visualize specific identity clusters
    # (After running create_3d_face_visualization)
    # embeddings_3d, labels_sample = get_sampled_data()  # You'd need to return these
    # visualize_identity_clusters(embeddings_3d, labels_sample, specific_identities=[42, 123, 456, 789, 1011])