#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA

DT_RECORD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<i4")])


def iter_batches(n_rows: int, batch_size: int):
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        yield start, end


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="embedding_ir101_adaface/embedding_ms1mv2_embeddings.npy")
    ap.add_argument("--lab", default="embedding_ir101_adaface/embedding_ms1mv2_labels.npy")
    ap.add_argument("--outdir", default="out_3d")
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--compression", default="snappy",
                    choices=["snappy", "zstd", "gzip", "brotli", "none"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows, else quick test")
    ap.add_argument("--norm", default="minmax01",
                    choices=["none", "minmax01", "minmax11"],
                    help="Normalization applied to PCA 3D output before saving")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    parquet_path = outdir / "embeddings_3d_with_labels.parquet"
    unity_dir = outdir / "unity_chunks"
    ensure_dir(unity_dir)
    index_path = outdir / "unity_chunks_index.json"

    # Memmap load
    X = np.load(args.emb, mmap_mode="r")
    y = np.load(args.lab, mmap_mode="r")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got {X.shape}")
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError(f"Row mismatch: embeddings {n}, labels {y.shape[0]}")

    if args.limit and args.limit > 0:
        n = min(n, args.limit)

    print(f"Embeddings: shape=({n},{d}), dtype={X.dtype}")
    print(f"Labels:     shape=({n},), dtype={y.dtype}")
    print(f"Batch size: {args.batch}, Unity chunk size: {args.chunk}")
    print(f"Normalization: {args.norm}")

    # -------------------------
    # Pass 1: Fit IncrementalPCA
    # -------------------------
    ipca = IncrementalPCA(n_components=3, batch_size=args.batch)

    print("\n[1/3] Fitting IncrementalPCA...")
    for i, (s, e) in enumerate(iter_batches(n, args.batch), start=1):
        batch = np.asarray(X[s:e], dtype=np.float32)
        ipca.partial_fit(batch)
        if i % 10 == 0:
            print(f"  fit progress: {e}/{n} ({(e/n)*100:.2f}%)")

    evr = getattr(ipca, "explained_variance_ratio_", None)
    if evr is not None:
        print(f"Explained variance ratio (3 comps): sum={evr.sum():.6f}")

    # -----------------------------------------
    # Pass 2: Find global bounds in PCA 3D space
    # -----------------------------------------
    if args.norm != "none":
        print("\n[2/3] Computing global min/max for normalization...")
        global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        for i, (s, e) in enumerate(iter_batches(n, args.batch), start=1):
            batch = np.asarray(X[s:e], dtype=np.float32)
            Z = ipca.transform(batch).astype(np.float32, copy=False)

            bmin = Z.min(axis=0).astype(np.float64)
            bmax = Z.max(axis=0).astype(np.float64)
            global_min = np.minimum(global_min, bmin)
            global_max = np.maximum(global_max, bmax)

            if i % 10 == 0:
                print(f"  bounds progress: {e}/{n} ({(e/n)*100:.2f}%)")

        ranges = global_max - global_min
        # Avoid divide-by-zero if some axis is constant
        ranges[ranges == 0] = 1.0

        print(f"  PCA global min: {global_min}")
        print(f"  PCA global max: {global_max}")

        def normalize(Z: np.ndarray) -> np.ndarray:
            Zf = Z.astype(np.float32, copy=False)
            Z01 = (Zf - global_min.astype(np.float32)) / ranges.astype(np.float32)
            if args.norm == "minmax01":
                return Z01
            # minmax11
            return (Z01 * 2.0) - 1.0
    else:
        print("\n[2/3] Skipping bounds pass (no normalization).")
        global_min = None
        global_max = None
        ranges = None

        def normalize(Z: np.ndarray) -> np.ndarray:
            return Z.astype(np.float32, copy=False)

    # ---------------------------------------------------
    # Pass 3: Transform again -> normalize -> write outputs
    # ---------------------------------------------------
    print("\n[3/3] Writing Parquet + Unity chunks (normalized)...")

    schema = pa.schema([
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("z", pa.float32()),
        ("label", pa.int32()),
    ])
    compression = None if args.compression == "none" else args.compression

    # Unity chunk buffering
    chunk_idx = 0
    chunk_buf = np.empty(args.chunk, dtype=DT_RECORD)
    chunk_fill = 0
    chunks_meta = []

    # For index bounds of SAVED data
    saved_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    saved_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    def flush_chunk():
        nonlocal chunk_idx, chunk_fill, saved_min, saved_max

        if chunk_fill == 0:
            return
        fn = f"chunk_{chunk_idx:05d}.bin"
        fp = unity_dir / fn
        data = chunk_buf[:chunk_fill]

        xs = data["x"].astype(np.float64, copy=False)
        ys = data["y"].astype(np.float64, copy=False)
        zs = data["z"].astype(np.float64, copy=False)
        cmin = np.array([xs.min(), ys.min(), zs.min()], dtype=np.float64)
        cmax = np.array([xs.max(), ys.max(), zs.max()], dtype=np.float64)

        saved_min = np.minimum(saved_min, cmin)
        saved_max = np.maximum(saved_max, cmax)

        with open(fp, "wb") as f:
            data.tofile(f)

        chunks_meta.append({
            "file": str(fp.relative_to(outdir)),
            "points": int(chunk_fill),
            "bounds_min": [float(cmin[0]), float(cmin[1]), float(cmin[2])],
            "bounds_max": [float(cmax[0]), float(cmax[1]), float(cmax[2])],
        })

        chunk_idx += 1
        chunk_fill = 0

    parquet_writer = pq.ParquetWriter(
        str(parquet_path),
        schema=schema,
        compression=compression,
        use_dictionary=False
    )

    try:
        for i, (s, e) in enumerate(iter_batches(n, args.batch), start=1):
            batch = np.asarray(X[s:e], dtype=np.float32)
            labels = np.asarray(y[s:e])
            if np.issubdtype(labels.dtype, np.integer):
                labels_i32 = labels.astype(np.int32, copy=False)
            else:
                labels_i32 = labels.astype(np.int32)

            Z = ipca.transform(batch).astype(np.float32, copy=False)
            ZN = normalize(Z)  # normalized (or passthrough)

            # Parquet write
            table = pa.Table.from_arrays(
                [
                    pa.array(ZN[:, 0], type=pa.float32()),
                    pa.array(ZN[:, 1], type=pa.float32()),
                    pa.array(ZN[:, 2], type=pa.float32()),
                    pa.array(labels_i32, type=pa.int32()),
                ],
                names=["x", "y", "z", "label"],
            )
            parquet_writer.write_table(table)

            # Unity chunk write
            pos = 0
            m = ZN.shape[0]
            while pos < m:
                space = args.chunk - chunk_fill
                take = min(space, m - pos)

                chunk_buf["x"][chunk_fill:chunk_fill + take] = ZN[pos:pos + take, 0]
                chunk_buf["y"][chunk_fill:chunk_fill + take] = ZN[pos:pos + take, 1]
                chunk_buf["z"][chunk_fill:chunk_fill + take] = ZN[pos:pos + take, 2]
                chunk_buf["label"][chunk_fill:chunk_fill + take] = labels_i32[pos:pos + take]

                chunk_fill += take
                pos += take

                if chunk_fill == args.chunk:
                    flush_chunk()

            if i % 10 == 0:
                print(f"  write progress: {e}/{n} ({(e/n)*100:.2f}%)")

        flush_chunk()

    finally:
        parquet_writer.close()

    # Index JSON
    index = {
        "version": 1,
        "normalization": args.norm,
        "pca_space_bounds_min": None if global_min is None else [float(global_min[0]), float(global_min[1]), float(global_min[2])],
        "pca_space_bounds_max": None if global_max is None else [float(global_max[0]), float(global_max[1]), float(global_max[2])],
        "saved_space_bounds_min": [float(saved_min[0]), float(saved_min[1]), float(saved_min[2])],
        "saved_space_bounds_max": [float(saved_max[0]), float(saved_max[1]), float(saved_max[2])],
        "record_format": {
            "endianness": "little",
            "stride_bytes": 16,
            "fields": [
                {"name": "x", "type": "float32"},
                {"name": "y", "type": "float32"},
                {"name": "z", "type": "float32"},
                {"name": "label", "type": "int32"},
            ],
        },
        "total_points": int(n),
        "chunk_points_target": int(args.chunk),
        "chunks": chunks_meta,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Parquet: {parquet_path}")
    print(f"Unity chunks dir: {unity_dir}")
    print(f"Unity index: {index_path}")


if __name__ == "__main__":
    main()


'''
python reduce_to_3d_parquet_and_unity_chunks.py \
  --emb "embedding_ir101_adaface/embedding_ms1mv2_embeddings.npy \
  --lab "embedding_ir101_adaface/embedding_ms1mv2_labels.npy" \
  --outdir out_3d \
  --batch 200000 \
  --chunk 250000 \
  --compression snappy \
  --norm minmax01
'''