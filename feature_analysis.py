import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse

# ── Import everything from new.py ─────────────────────────────────────────────
# new.py must live in the same directory as this file (or be on PYTHONPATH).
sys.path.insert(0, str(Path(__file__).parent))
from fusion_model import (
    DEFECT_CLASSES, NUM_CLASSES, IMG_SIZE, VEC_SIZE,
    load_and_preprocess,
    extract_classical_vector,
)

FEATURE_NAMES = [
    'dct_energy',    'edge_density', 'sat_mean',   'var_mean',
    'intensity',     'green_ratio',  'dark_ratio',  'hue_std',
    'lbp_entropy',   'glcm_contrast','glcm_corr',   'glcm_energy',
    'glcm_homo',     'sobel_mean',   'lab_a_mean',  'lab_b_mean',
]

# ─────────────────────────────────────────────────────────────────────────────
# METHOD A — GradientTape via tf.Variable
# ─────────────────────────────────────────────────────────────────────────────
def method_a_gradient(model, img: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Wraps vec in a tf.Variable so the tape tracks it automatically.
    Uses training=False to keep BatchNorm stable.
    May still return near-zero if BN absorbs the gradient — use Method B
    as primary in that case.
    """
    img_t = tf.constant(img[None, ...], dtype=tf.float32)
    vec_v = tf.Variable(vec[None, ...], dtype=tf.float32)

    grads = []
    for c in range(NUM_CLASSES):
        with tf.GradientTape() as tape:
            preds = model([img_t, vec_v], training=False)
            score = preds[:, c]
        g = tape.gradient(score, vec_v)
        grads.append(g.numpy()[0] if g is not None else np.zeros(VEC_SIZE))

    return np.array(grads)  # (NUM_CLASSES, VEC_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD B — Finite-difference perturbation  ← PRIMARY / MOST RELIABLE
# ─────────────────────────────────────────────────────────────────────────────
def method_b_finite_diff(model, img: np.ndarray, vec: np.ndarray,
                          epsilon: float = 1e-3) -> np.ndarray:
    """
    Central-difference estimate of ∂P(class_c)/∂feature_i.
    Bypasses BatchNorm gradient issues entirely — always produces
    non-zero values as long as the feature actually affects predictions.
    """
    img_t = tf.constant(img[None, ...], dtype=tf.float32)
    grads = np.zeros((NUM_CLASSES, VEC_SIZE), dtype=np.float32)

    for i in range(VEC_SIZE):
        vp = vec.copy(); vp[i] += epsilon
        vm = vec.copy(); vm[i] -= epsilon

        pp = model([img_t, tf.constant(vp[None, ...], dtype=tf.float32)],
                   training=False).numpy()[0]
        pm = model([img_t, tf.constant(vm[None, ...], dtype=tf.float32)],
                   training=False).numpy()[0]

        grads[:, i] = (pp - pm) / (2 * epsilon)

    return grads  # (NUM_CLASSES, VEC_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD C — Integrated Gradients (zero baseline)
# ─────────────────────────────────────────────────────────────────────────────
def method_c_integrated_grads(model, img: np.ndarray, vec: np.ndarray,
                               n_steps: int = 50) -> np.ndarray:
    """
    Integrated Gradients from baseline=zeros → vec.
    attribution[c, i] = vec[i] × ∫ ∂P(c)/∂vec_i dt   (t ∈ [0,1])
    Satisfies completeness — attributions sum to the prediction
    difference from the zero baseline.
    """
    img_t     = tf.constant(img[None, ...], dtype=tf.float32)
    alphas    = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
    sum_grads = np.zeros((NUM_CLASSES, VEC_SIZE), dtype=np.float32)

    for alpha in alphas:
        interp = alpha * vec
        vec_v  = tf.Variable(interp[None, ...], dtype=tf.float32)
        for c in range(NUM_CLASSES):
            with tf.GradientTape() as tape:
                preds = model([img_t, vec_v], training=False)
                score = preds[:, c]
            g = tape.gradient(score, vec_v)
            if g is not None:
                sum_grads[c] += g.numpy()[0]

    avg_grads    = sum_grads / (n_steps + 1)
    attributions = avg_grads * vec[None, :]
    return attributions  # (NUM_CLASSES, VEC_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISE  →  column-wise so each feature's contributions sum to 1
# ─────────────────────────────────────────────────────────────────────────────
def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    mat = np.abs(mat)
    return mat / (mat.sum(axis=0, keepdims=True) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY & SAVE
# ─────────────────────────────────────────────────────────────────────────────
def print_table(matrix: np.ndarray, method_name: str):
    print(f"\n{'='*70}")
    print(f"  Feature vs Class Contribution Matrix  [{method_name}]")
    print(f"  Rows = defect classes  |  Columns = features")
    print(f"  Values = relative importance of that feature for each class")
    print(f"{'='*70}")
    header = "Feature".ljust(22)
    for c in DEFECT_CLASSES:
        header += f"{c:>12}"
    print(header)
    print("─" * 82)
    for i, f in enumerate(FEATURE_NAMES):
        row = f.ljust(22)
        for j in range(NUM_CLASSES):
            row += f"{matrix[j, i]:12.4f}"
        print(row)
    print()


def save_primary_csv(matrix: np.ndarray, filename: str):
    """Rows = features, cols = classes — matches the original format."""
    df = pd.DataFrame(matrix.T, columns=DEFECT_CLASSES, index=FEATURE_NAMES)
    df.to_csv(filename)
    print(f"✅  Saved primary CSV → {filename}")


def save_all_methods_csv(matrices: dict, filename: str):
    """All three methods side-by-side in one CSV."""
    rows = []
    for i, fname in enumerate(FEATURE_NAMES):
        row = {'feature': fname}
        for mkey, mat in matrices.items():
            for j, cls in enumerate(DEFECT_CLASSES):
                row[f"{cls}_{mkey}"] = round(float(mat[j, i]), 6)
        rows.append(row)
    pd.DataFrame(rows).set_index('feature').to_csv(filename)
    print(f"✅  Saved all-methods CSV → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE USAGE SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
def feature_usage_test(model, img: np.ndarray, vec: np.ndarray):
    img_t  = tf.expand_dims(tf.constant(img, dtype=tf.float32), 0)
    p_real = model([img_t, tf.expand_dims(tf.constant(vec, dtype=tf.float32), 0)],
                   training=False).numpy()
    p_zero = model([img_t, tf.zeros([1, VEC_SIZE])],
                   training=False).numpy()
    diff   = np.abs(p_real - p_zero)

    print("\n─── FEATURE USAGE TEST ───────────────────────────────────────────")
    print(f"  With real features : {np.round(p_real[0], 4)}")
    print(f"  With zero features : {np.round(p_zero[0], 4)}")
    print(f"  Abs difference     : {np.round(diff[0], 4)}")
    print(f"  Total feature impact (Σ|Δ|): {diff.sum():.6f}")
    if diff.sum() < 1e-4:
        print("  ⚠  Features have near-zero impact. Classical branch weights")
        print("     may be very small. Finite-diff values will still be computed")
        print("     but expect small absolute magnitudes.")
    print("──────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    print(f"\n📂 Loading model  : {args.model}")
    model = tf.keras.models.load_model(args.model)

    print(f"🖼  Loading image  : {args.image}")
    img = load_and_preprocess(args.image)    # ← from new.py
    vec = extract_classical_vector(img)      # ← from new.py

    print(f"\n📐 Image shape    : {img.shape}")
    print(f"📊 Feature vector : {np.round(vec, 4)}")

    feature_usage_test(model, img, vec)

    print("⚙  Method A — GradientTape (tf.Variable) …")
    mat_a = normalize_matrix(method_a_gradient(model, img, vec))
    print_table(mat_a, "A — GradientTape")

    print("⚙  Method B — Finite-Difference (primary) …")
    mat_b = normalize_matrix(method_b_finite_diff(model, img, vec))
    print_table(mat_b, "B — Finite-Difference  ← PRIMARY")

    print("⚙  Method C — Integrated Gradients …")
    mat_c = normalize_matrix(method_c_integrated_grads(model, img, vec))
    print_table(mat_c, "C — Integrated Gradients")

    save_primary_csv(mat_b, args.output)
    save_all_methods_csv(
        {'gradtape': mat_a, 'findiff': mat_b, 'integrads': mat_c},
        args.output.replace('.csv', '_all_methods.csv')
    )

    print("\n📌 INTERPRETATION GUIDE")
    print("   Each cell = share of that class's gradient signal from that feature.")
    print("   Column values sum to 1 across all classes.")
    print("   High value → feature strongly drives that class prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature vs Class Contribution Matrix (imports from new.py)")
    parser.add_argument("--model",  required=True,
                        help="Path to .keras / .h5 model")
    parser.add_argument("--image",  required=True,
                        help="Path to input image")
    parser.add_argument("--output", default="feature_analysis_crack2.csv",
                        help="Output CSV filename (default: feature_analysis.csv)")
    main(parser.parse_args())