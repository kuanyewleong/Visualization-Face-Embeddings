"""
Simple examples showing common workflows for this project.
Run with: python examples/example_usage.py
"""

from main import create_3d_face_visualization, create_interactive_3d_plotly
import numpy as np


def main():
    # Example 1: Small interactive visualization (quick prototyping)
    fig = create_3d_face_visualization(
        n_samples=5000,
        n_classes_to_show=50,
        reduction_method='umap',
        interactive=True,
        save_csv=True,
        csv_path="examples/ms1mv2_embeddings_3d_example.csv",
        csv_include_sample_index=False
    )

    print("Interactive example complete.")
    print("Outputs (default locations):")
    print(" - ms1mv2_face_embeddings_3d_interactive.html")
    print(" - examples/ms1mv2_embeddings_3d_example.csv")

    # Example 2: If you already have pre-computed 3D embeddings
    # embeddings_3d = np.load('embeddings_3d_umap.npy')
    # labels = np.load('labels.npy')
    # create_interactive_3d_plotly(embeddings_3d, labels)


if __name__ == '__main__':
    main()
