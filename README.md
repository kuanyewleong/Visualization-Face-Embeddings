# Visualization — Face Embeddings (MS1MV2)

🔧 **Purpose**

This repository provides tools to sample, reduce, and visualize high-dimensional face embeddings (MS1MV2) in 3D. It supports interactive Plotly visualizations (saved as HTML) and publication-ready static Matplotlib figures (PNG), plus exports of 3D coordinates to CSV.

**Key Features**

- Intelligent sampling of large embedding sets (memory-mapped loading)
- Normalization and configurable dimensionality reduction (UMAP / PCA / PCA+UMAP)
- Interactive 3D visualization using Plotly (saved to `ms1mv2_face_embeddings_3d_interactive.html`)
- Static publication-quality Matplotlib plots (saved to `ms1mv2_face_embeddings_3d_static.png`)
- Exportable 3D point CSV (`ms1mv2_embeddings_3d.csv`) with optional labels and sample indices
- Utility functions for detailed cluster views and rotating 3D animations (GIF)

---

## Installation

Recommended: use a virtual environment (venv / conda).

```bash
pip install -r requirements.txt
# Or install directly:
pip install numpy matplotlib plotly umap-learn scikit-learn pillow
```

Note: UMAP (`umap-learn`) can be memory intensive for very large datasets; using `reduction_method='pca'` or `pca+umap` and the script's `low_memory` options can help.

---

## Expected Inputs

- `embedding_ir101_adaface/embedding_ms1mv2_embeddings.npy` — NumPy array of shape (N, D)
- `embedding_ir101_adaface/embedding_ms1mv2_labels.npy` — NumPy array of integer labels (length N)

(You can replace paths with your own arrays when calling the functions.)

---

## Usage (from `main.py`)

Primary entry point: `create_3d_face_visualization(...)`

Examples:

- Basic interactive visualization (default behavior in `main.py`):

```python
fig = create_3d_face_visualization(
    n_samples=10000,
    n_classes_to_show=500,
    reduction_method='umap',
    interactive=True
)
# An HTML file will be saved: ms1mv2_face_embeddings_3d_interactive.html
```

- Static (Matplotlib) visualization for publication:

```python
fig = create_3d_face_visualization(
    n_samples=50000,
    n_classes_to_show=30,
    reduction_method='pca',
    interactive=False
)
# PNG will be saved: ms1mv2_face_embeddings_3d_static.png
```

- Export pre-computed 3D embeddings and visualize:

```python
# If you already have 3D embeddings and labels saved as .npy
embeddings_3d = np.load('embeddings_3d_umap.npy')
labels = np.load('labels.npy')
create_interactive_3d_plotly(embeddings_3d, labels)
```

---

## Functions & Parameters (summary)

- `save_3d_points_to_csv(embeddings_3d, labels, out_csv_path, include_label=True, include_sample_index=None)`
  - Saves columns x,y,z and optional `label` and `sample_index` to CSV.

- `create_3d_face_visualization(embeddings_path, labels_path, n_samples=30000, n_classes_to_show=50, reduction_method='umap', interactive=True, save_csv=True, csv_path='ms1mv2_embeddings_3d.csv', csv_include_label=True, csv_include_sample_index=True)`
  - Loads embeddings (memory-mapped), samples, normalizes, reduces to 3D, saves CSV, and returns either a Plotly `Figure` or Matplotlib `Figure` depending on `interactive`.

- `create_interactive_3d_plotly(embeddings_3d, labels, n_classes_to_show=50)`
  - Creates and saves a Plotly interactive HTML with controls and predefined camera views.

- `create_static_3d_matplotlib(embeddings_3d, labels, n_classes_to_show=30)`
  - Creates a two-panel static figure (colored by identity and by density) and saves PNG.

- `visualize_identity_clusters(embeddings_3d, labels, specific_identities=None, n_per_identity=10)`
  - Shows detailed cluster views for specific identities and centroids.

- `create_3d_animation(embeddings_3d, labels, output_path='3d_rotation.gif')`
  - Rotates the 3D scatter and saves a GIF (requires `pillow` or `imagemagick`).

---

## Tips & Notes

- Use a lower `n_samples` when prototyping to speed up UMAP.
- For very large datasets, PCA first (`reduction_method='pca+umap'`) reduces memory/time.
- The script uses `np.load(..., mmap_mode='r')` to handle large `.npy` files without loading all data into RAM.
- When saving CSV, samples indices are the indices into the original arrays before sampling (if enabled).

---

## Output Files (default)

- `ms1mv2_embeddings_3d.csv` — Exported 3D coordinates
- `ms1mv2_face_embeddings_3d_interactive.html` — Interactive Plotly visualization
- `ms1mv2_face_embeddings_3d_static.png` — Static Matplotlib figure
- `3d_rotation.gif` (when generated) — Rotating animation GIF

---

## License & Credits

This repository includes a `LICENSE` file — please consult it for usage terms.

---

## Included examples & requirements ✅

- `requirements.txt` — lists the minimal packages to install.
- `examples/example_usage.py` — small script demonstrating the most common workflows. Run:

```bash
python examples/example_usage.py
```

---

