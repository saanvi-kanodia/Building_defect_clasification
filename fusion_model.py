import random, argparse, warnings
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.fftpack import dct as scipy_dct
from skimage.morphology import skeletonize
from skimage.filters import threshold_multiotsu
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

warnings.filterwarnings('ignore')

# ── Global config ─────────────────────────────────────────────────────────────
DEFECT_CLASSES = ['damp', 'mold', 'crack', 'spalling', 'peeling']
NUM_CLASSES    = len(DEFECT_CLASSES)
IMG_SIZE       = 256
BATCH_SIZE     = 16
EPOCHS_P1      = 12
EPOCHS_P2      = 18
VEC_SIZE       = 16
SEED           = 42
EXTENSIONS     = {'.jpg', '.jpeg', '.png', '.bmp'}

random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# 2.  PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════
def preprocess(image: np.ndarray) -> np.ndarray:
    """
    CLAHE on V channel (HSV) + bilateral smoothing → resize + normalize.
    Separates luminance from chroma before enhancement so colour-based
    defects (mold, damp stains) are not washed out.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0,
                                     tileGridSize=(8, 8)).apply(hsv[:, :, 2])
    out = cv2.bilateralFilter(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 9, 75, 75)
    return cv2.resize(out, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

def load_and_preprocess(path) -> np.ndarray:
    if isinstance(path, tf.Tensor):
        path = path.numpy()
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return preprocess(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# 3.  CLASSICAL FEATURE VECTOR  (16 features)
# ════════════════════════════════════════════════════════════════════════════════
def _dct_energy(gray: np.ndarray, bs: int = 32) -> float:
    """
    High-frequency DCT energy ratio.
    FIX: skip blocks that are too small; return 0 gracefully if image < bs.
    """
    h, w = gray.shape
    if h < bs or w < bs:
        return 0.0
    tot = hf = 0.0
    for r in range(0, h - bs + 1, bs):
        for c in range(0, w - bs + 1, bs):
            D   = scipy_dct(scipy_dct(
                      gray[r:r+bs, c:c+bs].astype(np.float32),
                      axis=0, norm='ortho'), axis=1, norm='ortho')
            tot += float((D ** 2).sum())
            hf  += float((D[bs // 2:, bs // 2:] ** 2).sum())
    return hf / (tot + 1e-8)


def _lbp_entropy(gray: np.ndarray, P: int = 8, R: float = 1.0) -> float:
    """LBP histogram entropy — high for mold / rough spalling."""
    lbp  = local_binary_pattern(gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=P + 2,
                           range=(0, P + 2), density=True)
    hist += 1e-8
    return float(-np.sum(hist * np.log2(hist)))


def _glcm_features(gray: np.ndarray):
    """4 GLCM statistics: contrast, correlation, energy, homogeneity."""
    g8   = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(g8, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2],
                        levels=8, symmetric=True, normed=True)
    return (float(graycoprops(glcm, 'contrast').mean()),
            float(graycoprops(glcm, 'correlation').mean()),
            float(graycoprops(glcm, 'energy').mean()),
            float(graycoprops(glcm, 'homogeneity').mean()))

def extract_classical_vector(img_f32: np.ndarray) -> np.ndarray:

    u8   = (img_f32 * 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    lab  = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)

    dct_e        = _dct_energy(gray)
    tophat       = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    edge_density = float(np.sum(cv2.Canny(tophat, 30, 90) > 0)) / IMG_SIZE ** 2
    sat_mean     = float(hsv[:, :, 1].mean()) / 255.0
    g            = gray.astype(np.float32)
    var_mean     = float(np.maximum(
                       cv2.boxFilter(g*g, -1, (15,15)) -
                       cv2.boxFilter(g,   -1, (15,15))**2, 0).mean()) / (255**2)
    intensity    = float(gray.mean()) / 255.0
    green_ratio  = float(img_f32[:,:,1].mean()) / (float(img_f32[:,:,0].mean()) + 1e-8)
    dark_ratio   = float((gray < 80).sum()) / IMG_SIZE**2
    hue_std      = float(hsv[:,:,0].astype(np.float32).std()) / 180.0
    lbp_ent      = _lbp_entropy(gray)
    gc, gcorr, ge, gh = _glcm_features(gray)
    sobel_mean   = float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)).mean()) / 255.0
    lab_a        = float(lab[:,:,1].mean()) / 128.0
    lab_b        = float(lab[:,:,2].mean()) / 128.0

    return np.array([dct_e, edge_density, sat_mean, var_mean,
                     intensity, green_ratio, dark_ratio, hue_std,
                     lbp_ent, gc, gcorr, ge, gh,
                     sobel_mean, lab_a, lab_b], dtype=np.float32)

# 4.  tf.data PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

def make_tf_dataset(split_dir: Path, training: bool):
    paths, labels = [], []
    for idx, cls in enumerate(DEFECT_CLASSES):
        d = split_dir / cls
        if not d.exists():
            print(f"  ⚠  {split_dir.name}/{cls} missing — skipped")
            continue
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            for p in d.glob(ext):
                paths.append(str(p)); labels.append(idx)
    if not paths:
        raise FileNotFoundError(f"No images found in {split_dir}")
    print(f"  {split_dir.name}: {len(paths)} images "
          f"({dict(zip(*np.unique(labels, return_counts=True)))})")

    def _load(path, label):
        def _py(p):
            img = load_and_preprocess(p)
            vec = extract_classical_vector(img)
            return img, vec
        img, vec = tf.numpy_function(_py, [path], [tf.float32, tf.float32])
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        vec.set_shape([VEC_SIZE])
        return (img, vec), label

    def augment(inputs, label):
        img, vec = inputs
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.25)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_hue(img, 0.05)
        shape  = tf.shape(img)
        crop_h = tf.cast(tf.cast(shape[0], tf.float32)*tf.random.uniform([],0.90,1.0), tf.int32)
        crop_w = tf.cast(tf.cast(shape[1], tf.float32)*tf.random.uniform([],0.90,1.0), tf.int32)
        img    = tf.image.random_crop(img, [crop_h, crop_w, 3])
        img    = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return (tf.clip_by_value(img, 0.0, 1.0), vec), label

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(paths),
        tf.data.Dataset.from_tensor_slices(tf.one_hot(labels, NUM_CLASSES))
    )).map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000, seed=SEED).map(augment,num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), labels


# 5.  CLASSICAL FEATURE EXTRACTOR — per-class masks & severity inputs
# ════════════════════════════════════════════════════════════════════════════════
class ClassicalFeatureExtractor:

    @staticmethod
    def _gray(img): return cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    @staticmethod
    def _u8(img):   return (img*255).astype(np.uint8)

    # ── crack ────────────────────────────────────────────────────────────────
    def crack_features(self, img: np.ndarray) -> dict:

        gray = self._gray(img)

        # ── Stage 1: top-hat on two kernel sizes ─────────────────────────────
        th_small = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        th_large = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)))
        tophat   = cv2.addWeighted(th_small, 0.5, th_large, 0.5, 0)

        # ── Stage 2: multi-scale Canny ────────────────────────────────────────
        edges_tight = cv2.Canny(tophat, 20, 60)
        edges_loose = cv2.Canny(tophat, 10, 40)
        edges       = cv2.bitwise_or(edges_tight, edges_loose)

        # ── Stage 3: adaptive threshold fallback on CLAHE-enhanced gray ──────
        clahe_gray  = cv2.createCLAHE(clipLimit=3.0,
                           tileGridSize=(4,4)).apply(gray)
        adapt       = cv2.adaptiveThreshold(
                          cv2.GaussianBlur(clahe_gray, (5,5), 0),
                          255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv2.THRESH_BINARY_INV, 11, 4)
        edges       = cv2.bitwise_or(edges, adapt)

        # ── Stage 4: morphological cleanup ───────────────────────────────────
        k5   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        k3   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(
                   cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k5),
                   cv2.MORPH_OPEN,  k3)

        # ── Stage 5: elongation filter ────────────────────────────────────────
        min_area = max(150, int(0.001 * IMG_SIZE**2))   # lowered from 300
        n, labeled, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        for i in range(1, n):
            area   = stats[i, cv2.CC_STAT_AREA]
            w_     = stats[i, cv2.CC_STAT_WIDTH]
            h_     = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(w_, h_) / (min(w_, h_) + 1e-8)
            if area > min_area and aspect > 2.5:
                filtered[labeled == i] = 255
        mask = filtered

        skel = skeletonize((mask > 0).astype(np.uint8)).astype(np.uint8)
        diag = (IMG_SIZE ** 2 + IMG_SIZE ** 2) ** 0.5
        return {
            'edges':           edges,
            'crack_mask':      mask,
            'skeleton':        skel,
            'crack_length_px': int(skel.sum()),
            'crack_density':   round(float(skel.sum()) / diag, 4),
            'affected_pct':    round(float((mask>0).sum()) / IMG_SIZE**2 * 100, 2),
        }

    # ── mold ─────────────────────────────────────────────────────────────────
    def mold_features(self, img: np.ndarray) -> dict:
        hsv   = cv2.cvtColor(self._u8(img), cv2.COLOR_RGB2HSV)
        # Green-brown mold range (HSV)
        cmask = cv2.inRange(hsv, np.array([20,  30,  0]),
                                  np.array([90, 255, 180]))
        ke    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ks    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(
                    cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, ke),
                    cv2.MORPH_OPEN, ks)
        bnd   = clean - cv2.erode(clean,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        n, *_ = cv2.connectedComponentsWithStats(clean)
        return {
            'color_mask':   cmask,
            'mold_mask':    clean,
            'boundary':     bnd,
            'colony_count': n - 1,
            'affected_pct': round(float((clean>0).sum()) / IMG_SIZE**2 * 100, 2),
        }

    # ── damp ─────────────────────────────────────────────────────────────────
    def damp_features(self, img: np.ndarray) -> dict:

        gray = self._gray(img)
        try:
            t = threshold_multiotsu(gray, classes=3)
        except Exception:
            t = [gray.mean() - 15, gray.mean()]

        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dm = cv2.morphologyEx(
                 cv2.morphologyEx(
                     (gray < t[0]).astype(np.uint8) * 255,
                     cv2.MORPH_OPEN, k),
                 cv2.MORPH_CLOSE, k)

        # (structured surfaces like brickwork, tile grout patterns)
        g      = gray.astype(np.float32)
        vmap   = np.maximum(cv2.boxFilter(g*g, -1, (9,9)) -
                             cv2.boxFilter(g,   -1, (9,9))**2, 0)
        # pixels with very high local variance → likely structural texture, not damp
        high_texture = (vmap > np.percentile(vmap, 85)).astype(np.uint8) * 255
        dm    = cv2.bitwise_and(dm, cv2.bitwise_not(high_texture))

        return {
            'damp_mask':    dm,
            'thresholds':   t,
            'affected_pct': round(float((dm>0).sum()) / IMG_SIZE**2 * 100, 2),
        }

    # ── spalling ─────────────────────────────────────────────────────────────
    def spalling_features(self, img: np.ndarray) -> dict:

        gray = self._gray(img)
        u8   = self._u8(img)
        hsv  = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
        g    = gray.astype(np.float32)

        vmap  = np.maximum(cv2.boxFilter(g*g, -1, (15,15)) -
                            cv2.boxFilter(g,   -1, (15,15))**2, 0)
        k     = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        vmask = cv2.morphologyEx(
                    (vmap > np.percentile(vmap, 70)).astype(np.uint8)*255,
                    cv2.MORPH_OPEN, k)

        # Colour guard: spalling areas are low-saturation (exposed concrete/brick)
        low_sat = (hsv[:,:,1] < 60).astype(np.uint8) * 255
        rmask   = cv2.bitwise_and(vmask, low_sat)
        rmask   = cv2.morphologyEx(rmask, cv2.MORPH_CLOSE, k)

        return {
            'variance_map':     vmap,
            'rough_mask':       rmask,
            'dct_energy_ratio': round(_dct_energy(gray), 4),
            'affected_pct':     round(float((rmask>0).sum()) / IMG_SIZE**2 * 100, 2),
        }

    # ── peeling ──────────────────────────────────────────────────────────────
    def peeling_features(self, img: np.ndarray) -> dict:
        """
        FIX v3: Added saturation guard (opposite of spalling) — peeling paint
        retains some colour saturation at lifted edges.
        Horizontal Sobel remains for delamination layer detection.
        """
        gray  = self._gray(img)
        u8    = self._u8(img)
        hsv   = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
        g     = gray.astype(np.float32)

        vmap  = np.maximum(cv2.boxFilter(g*g, -1, (15,15)) -
                            cv2.boxFilter(g,   -1, (15,15))**2, 0)
        k     = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        vmask = cv2.morphologyEx(
                    (vmap > np.percentile(vmap, 65)).astype(np.uint8)*255,
                    cv2.MORPH_OPEN, k)

        sob_h      = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        layer_mask = ((np.abs(sob_h) > np.percentile(np.abs(sob_h), 80))
                      .astype(np.uint8)*255)

        # Saturation guard: peeling paint has slightly more colour than raw concrete
        has_colour = (hsv[:,:,1] > 25).astype(np.uint8) * 255
        peel_mask  = cv2.bitwise_and(vmask, layer_mask)
        peel_mask  = cv2.bitwise_and(peel_mask, has_colour)
        peel_mask  = cv2.morphologyEx(peel_mask, cv2.MORPH_CLOSE, k)

        return {
            'variance_map':     vmap,
            'rough_mask':       vmask,
            'layer_mask':       layer_mask,
            'peel_mask':        peel_mask,
            'dct_energy_ratio': round(_dct_energy(gray), 4),
            'affected_pct':     round(float((peel_mask>0).sum()) / IMG_SIZE**2 * 100, 2),
        }


# ════════════════════════════════════════════════════════════════════════════════
# 6.  LOSS — Focal loss
# ════════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def _loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.exp(-ce)
        fl = alpha * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    return _loss


# ════════════════════════════════════════════════════════════════════════════════
# 7.  MODEL — MobileNetV2 + classical branch fusion
# ════════════════════════════════════════════════════════════════════════════════

def build_model(freeze_backbone: bool = True) -> Model:
    img_inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')
    base    = tf.keras.applications.MobileNetV2(
                  input_shape=(IMG_SIZE, IMG_SIZE, 3),
                  include_top=False, weights='imagenet')

    if freeze_backbone:
        base.trainable = False
    else:
        for layer in base.layers:      layer.trainable = False
        for layer in base.layers[-30:]: layer.trainable = True

    cnn_feat = layers.GlobalAveragePooling2D()(base(img_inp, training=False))
    cnn_feat = layers.Dropout(0.3)(cnn_feat)

    # Classical branch (16-D → 64 → 32)
    vec_inp  = layers.Input(shape=(VEC_SIZE,), name='classical')
    vec_feat = layers.Dense(64, activation='relu')(vec_inp)
    vec_feat = layers.BatchNormalization()(vec_feat)
    vec_feat = layers.Dense(32, activation='relu')(vec_feat)

    fused = layers.Concatenate()([cnn_feat, vec_feat])
    fused = layers.Dropout(0.5)(fused)
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dropout(0.3)(fused)
    out   = layers.Dense(NUM_CLASSES, activation='softmax')(fused)

    return Model(inputs=[img_inp, vec_inp], outputs=out)


def compile_model(model: Model, lr: float, label_smooth: float = 0.1) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smooth),
        metrics=['accuracy'])
    return model


# ════════════════════════════════════════════════════════════════════════════════
# 7b.  GRAD-CAM
# ════════════════════════════════════════════════════════════════════════════════

def compute_gradcam(model: Model, img_f32: np.ndarray,
                    vec: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Grad-CAM heatmap from the last Conv2D in MobileNetV2.
    Returns (H, W) float32 normalized to [0,1].
    """
    try:
        base_model = model.get_layer('mobilenetv2_1.00_256')
        last_conv  = next(l.name for l in reversed(base_model.layers)
                          if isinstance(l, layers.Conv2D))
        grad_model = Model(
            inputs  = model.inputs,
            outputs = [base_model.get_layer(last_conv).output, model.output])

        inp = {'image':     tf.expand_dims(img_f32, 0),
               'classical': tf.expand_dims(vec, 0)}
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(inp, training=False)
            loss = preds[:, class_idx]

        grads  = tape.gradient(loss, conv_out)[0]
        pooled = tf.reduce_mean(grads, axis=(0, 1))
        cam    = tf.reduce_sum(conv_out[0] * pooled, axis=-1).numpy()
        cam    = np.maximum(cam, 0)
        cam    = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return (cam / (cam.max() + 1e-8)).astype(np.float32)
    except Exception as e:
        print(f"  ⚠  Grad-CAM failed: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


# 8.  SEVERITY
# ════════════════════════════════════════════════════════════════════════════════

def compute_severity(cls: str, cr: dict) -> dict:
    """
    Severity formula per defect type.
    crack_density is normalised to image diagonal → resolution-independent.
    """
    aff = cr.get('affected_pct', 0.0)

    def out(sev, **extra):
        s = round(min(100.0, max(0.0, sev)), 1)
        return {'affected_area_pct': aff,
                'severity_score':    s,
                'wall_health_index': round(100.0 - s, 1),
                **extra}

    if cls == 'crack':
        cd = cr.get('crack_density', 0.0)
        cl = cr.get('crack_length_px', 0)
        return out(aff * 2 + cd * 200, crack_length_px=cl, crack_density=cd)
    if cls == 'mold':
        cc = cr.get('colony_count', 0)
        return out(aff * 1.5 + cc * 2, colony_count=cc)
    if cls == 'damp':
        return out(aff * 1.5)
    if cls == 'spalling':
        de = cr.get('dct_energy_ratio', 0.0)
        return out(aff + de * 50, dct_energy_ratio=de)
    if cls == 'peeling':
        de = cr.get('dct_energy_ratio', 0.0)
        return out(aff * 1.2 + de * 40, dct_energy_ratio=de)
    return out(aff)


def health_label(whi: float):
    if whi >= 75: return 'GOOD',     '#2ecc71'
    if whi >= 50: return 'MODERATE', '#f39c12'
    if whi >= 25: return 'POOR',     '#e67e22'
    return               'CRITICAL', '#e74c3c'

# 9.  MAIN ANALYZER
# ════════════════════════════════════════════════════════════════════════════════
class WallDefectAnalyzer:

    def __init__(self, data_dir: str = 'dataset'):
        self.data_dir  = Path(data_dir)
        self.extractor = ClassicalFeatureExtractor()
        self.model     = None

    # ── training ─────────────────────────────────────────────────────────────
    def train(self):
        print("\n📂 Building datasets …")
        train_ds, train_labels = make_tf_dataset(self.data_dir / 'train', True)
        val_ds,   _            = make_tf_dataset(self.data_dir / 'val',   False)

        weights      = compute_class_weight('balanced',
                           classes=np.arange(NUM_CLASSES), y=train_labels)
        class_weight = dict(enumerate(weights))
        print(f"  Class weights: "
              f"{ {DEFECT_CLASSES[i]: round(w,2) for i,w in class_weight.items()} }")

        print(f"\n🔒 Phase-1  (frozen backbone, {EPOCHS_P1} epochs, LR=1e-3) …")
        self.model = compile_model(build_model(freeze_backbone=True), lr=1e-3)
        h1 = self.model.fit(train_ds, validation_data=val_ds,
                            epochs=EPOCHS_P1, callbacks=self._callbacks('phase1'),
                            class_weight=class_weight, verbose=1)

        print(f"\n🔓 Phase-2  (top-30 unfrozen, {EPOCHS_P2} epochs, LR=1e-5) …")
        p2 = build_model(freeze_backbone=False)
        p2.set_weights(self.model.get_weights())
        self.model = compile_model(p2, lr=1e-5, label_smooth=0.1)
        h2 = self.model.fit(train_ds, validation_data=val_ds,
                            epochs=EPOCHS_P2, callbacks=self._callbacks('phase2'),
                            class_weight=class_weight, verbose=1)

        self._plot_history(h1, h2)

        print("\n📊 Evaluating on TEST set …")
        test_ds, _ = make_tf_dataset(self.data_dir / 'test', False)
        self._evaluate(test_ds, tag='test')
        return h1, h2

    def _callbacks(self, tag: str):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=6,
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.3,
                patience=3, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                f'best_wall_{tag}.keras',
                monitor='val_accuracy', save_best_only=True, verbose=1),
        ]

    def load(self, path: str = 'best_wall_phase2.keras'):
        self.model = tf.keras.models.load_model(path)
        print(f"  Loaded model from {path}")

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> dict:
        if self.model is None:
            raise RuntimeError("Call train() or load() first.")

        img_f32  = load_and_preprocess(image_path)
        vec      = extract_classical_vector(img_f32)
        probs    = self.model.predict(
                       {'image':     np.expand_dims(img_f32, 0),
                        'classical': np.expand_dims(vec,     0)},
                       verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_cls = DEFECT_CLASSES[pred_idx]

        gradcam  = compute_gradcam(self.model, img_f32, vec, pred_idx)

        dispatch = {
            'crack':    self.extractor.crack_features,
            'mold':     self.extractor.mold_features,
            'damp':     self.extractor.damp_features,
            'spalling': self.extractor.spalling_features,
            'peeling':  self.extractor.peeling_features,
        }
        classical = dispatch[pred_cls](img_f32)

        feat_names = ['dct_energy','edge_density','sat_mean','var_mean',
                      'intensity','green_ratio','dark_ratio','hue_std',
                      'lbp_entropy','glcm_contrast','glcm_corr',
                      'glcm_energy','glcm_homo','sobel_mean',
                      'lab_a_mean','lab_b_mean']
        results = {
            'predicted_defect':  pred_cls,
            'confidence':        round(float(probs.max()), 4),
            'all_probabilities': dict(zip(DEFECT_CLASSES, probs.tolist())),
            'classical_vector':  dict(zip(feat_names, vec.tolist())),
            'classical':         classical,
            'severity':          compute_severity(pred_cls, classical),
            'gradcam':           gradcam,
        }
        self._visualise(image_path, img_f32, vec, results)
        return results

    # ── evaluation ────────────────────────────────────────────────────────────
    def _evaluate(self, ds, tag: str = 'val'):
        preds, labs = [], []
        for (imgs, vecs), lbls in ds:
            p = self.model.predict({'image': imgs, 'classical': vecs}, verbose=0)
            preds.extend(np.argmax(p, axis=1))
            labs.extend(np.argmax(lbls.numpy(), axis=1))
        print(classification_report(labs, preds,
                                    target_names=DEFECT_CLASSES, zero_division=0))
        cm = confusion_matrix(labs, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=DEFECT_CLASSES, yticklabels=DEFECT_CLASSES, ax=ax)
        ax.set_title(f'Confusion Matrix ({tag})')
        ax.set_ylabel('True'); ax.set_xlabel('Predicted')
        plt.tight_layout()
        fname = f'confusion_matrix_{tag}.png'
        plt.savefig(fname, dpi=150); plt.close()
        print(f"  Saved {fname}")

    @staticmethod
    def _plot_history(h1, h2):
        def _m(k): return h1.history.get(k,[]) + h2.history.get(k,[])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        p1_end = len(h1.history['loss'])
        ep     = list(range(1, len(_m('loss'))+1))
        for ax, key, title in zip(axes, ['loss','accuracy'], ['Loss','Accuracy']):
            ax.plot(ep, _m(key), label='train')
            ax.plot(ep, _m(f'val_{key}'), label='val', linestyle='--')
            ax.axvline(p1_end+0.5, color='orange', linestyle=':', label='phase boundary')
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
        plt.suptitle('Training History (Phase 1 + 2)', fontsize=13)
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150); plt.close()
        print("  Saved training_history.png")

    # ── visualisation  — 3×3 panel ───────────────────────────────────────────
    def _visualise(self, image_path: str, img_f32: np.ndarray,
                   vec: np.ndarray, results: dict):
        original = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        cls      = results['predicted_defect']
        sev      = results['severity']
        cl       = results['classical']
        cam      = results['gradcam']
        probs    = results['all_probabilities']
        fvec     = results['classical_vector']

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes.flat:
            ax.set_facecolor('#16213e'); ax.axis('off')

        def show(ax, img, title, cmap=None):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, color='white', fontsize=10, pad=6)
            ax.axis('off')

        # ── Row 0: original | preprocessed | Grad-CAM ────────────────────────
        show(axes[0,0], original,  'Original Image')
        show(axes[0,1], img_f32,   'Preprocessed\n(HSV·CLAHE·Bilateral)')

        heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        overlay = np.clip(0.55*img_f32 + 0.45*heat, 0, 1)
        show(axes[0,2], overlay, f'Grad-CAM\n({cls.title()} activation)')

        # ── Row 1: defect mask | class-specific | probability bar ─────────────
        mask_cfg = {
            'crack':    ('crack_mask', 'Crack Mask\n(Multi-scale Canny)', 'hot'),
            'mold':     ('mold_mask',  'Mold Mask\n(HSV colour range)',   'Greens'),
            'damp':     ('damp_mask',  'Damp Mask\n(Multi-Otsu)',         'Blues'),
            'spalling': ('rough_mask', 'Spalling Mask\n(Var+Sat guard)',  'YlOrRd'),
            'peeling':  ('peel_mask',  'Peeling Mask\n(Var+Sobel-H)',     'copper'),
        }
        mkey, mtitle, mcmap = mask_cfg[cls]
        show(axes[1,0], cl[mkey], mtitle, mcmap)

        if cls == 'crack':
            ov = np.stack([img_f32[:,:,0]]*3, -1)
            ov[cl['skeleton']>0] = [1,0,0]
            show(axes[1,1], ov, f'Crack Skeleton\n{cl["crack_length_px"]} px  '
                                f'| density {cl["crack_density"]:.3f}')
        elif cls == 'mold':
            show(axes[1,1], cl['boundary'], 'Colony Boundaries', 'plasma')
        elif cls == 'damp':
            show(axes[1,1], cl['damp_mask'], 'Damp Region', 'Blues')
        elif cls == 'spalling':
            show(axes[1,1], cl['variance_map'], 'Local Variance Map', 'inferno')
        else:
            show(axes[1,1], cl['layer_mask'], 'Layer-Edge Map\n(Sobel-H)', 'magma')

        ax_bar = axes[1,2]
        ax_bar.axis('on'); ax_bar.set_facecolor('#0f3460')
        colors = ['#e74c3c' if k==cls else '#3498db' for k in probs]
        ax_bar.bar(probs.keys(), probs.values(), color=colors, alpha=0.85)
        ax_bar.set_title(
            f'CNN+Classical Probabilities\n{cls.title()} ({results["confidence"]:.1%})',
            color='white', fontsize=10)
        ax_bar.set_ylim(0,1)
        ax_bar.tick_params(colors='white', labelsize=8)
        for sp in ax_bar.spines.values(): sp.set_edgecolor('#444')

        # ── Row 2: feature bar | LBP map | severity report ────────────────────
        ax_feat = axes[2,0]
        ax_feat.axis('on'); ax_feat.set_facecolor('#0f3460')
        fnames = list(fvec.keys())
        fvals  = np.clip(list(fvec.values()), 0, 1)
        ax_feat.bar(range(len(fnames)), fvals, color='#3498db', alpha=0.85)
        ax_feat.set_xticks(range(len(fnames)))
        ax_feat.set_xticklabels(fnames, rotation=65, ha='right',
                                 fontsize=6, color='white')
        ax_feat.set_title('Classical Feature Vector (16-D)',
                          color='white', fontsize=10)
        ax_feat.tick_params(colors='white', labelsize=7)
        for sp in ax_feat.spines.values(): sp.set_edgecolor('#444')

        gray_u8 = cv2.cvtColor((img_f32*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lbp     = local_binary_pattern(gray_u8, 8, 1, method='uniform')
        show(axes[2,1], lbp, 'LBP Texture Map\n(P=8 R=1 uniform)', 'gray')

        whi = sev['wall_health_index']
        hlabel, hcolor = health_label(whi)
        lines = [
            '  SEVERITY REPORT  ',
            f"{'─'*26}",
            f"  Defect Type   : {cls.upper()}",
            f"  Affected Area : {sev['affected_area_pct']:.1f}%",
        ]
        if cls == 'crack':
            lines.append(f"  Crack Length  : {sev['crack_length_px']} px")
            lines.append(f"  Crack Density : {sev.get('crack_density','—')}")
        elif cls == 'mold':
            lines.append(f"  Colonies      : {sev.get('colony_count','—')}")
        elif cls in ('spalling','peeling'):
            lines.append(f"  DCT Energy    : {sev.get('dct_energy_ratio','—')}")
        lines += [
            f"{'─'*26}",
            f"  Severity Score: {sev['severity_score']}/100",
            f"  Wall Health   : {whi}/100",
            f"  Status        : {hlabel}",
            f"{'─'*26}",
        ]
        axes[2,2].axis('off')
        axes[2,2].text(
            0.05, 0.95, '\n'.join(lines),
            transform=axes[2,2].transAxes,
            fontsize=11, va='top', fontfamily='monospace', color=hcolor,
            bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.9))

        plt.suptitle(f'Wall Defect Analysis  —  {Path(image_path).name}',
                     color='white', fontsize=14, y=1.01)
        plt.tight_layout()
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = f'result_{Path(image_path).stem}_{ts}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved {out}")


# ════════════════════════════════════════════════════════════════════════════════
# 10.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description='Wall Defect Analyzer v3')
    p.add_argument('--dst',     default='dataset',
                   help='Dataset root with train/ val/ test/ sub-folders')
    p.add_argument('--train',   action='store_true',
                   help='Run full two-phase training')
    p.add_argument('--predict', nargs='+', metavar='IMAGE',
                   help='Run inference on image(s)')
    p.add_argument('--model',   default='best_wall_phase2.keras',
                   help='Saved model file to load for --predict')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.train:
        analyzer = WallDefectAnalyzer(data_dir=args.dst)
        analyzer.train()
        print("\nTraining complete.")

    if args.predict:
        analyzer = WallDefectAnalyzer(data_dir=args.dst)
        analyzer.load(args.model)
        for img_path in args.predict:
            print(f"\n── {img_path} ──")
            r   = analyzer.predict(img_path)
            sev = r['severity']
            lbl, _ = health_label(sev['wall_health_index'])
            print(f"   Prediction : {r['predicted_defect']}  "
                  f"({r['confidence']:.1%})")
            print(f"   Severity   : {sev['severity_score']}/100  "
                  f"│  Wall Health : {sev['wall_health_index']}/100  "
                  f"│  Status: {lbl}")

    if not (args.train or args.predict):
        print("No action given. Use --train or --predict <image.jpg>")
        print("Example:")
        print("python new.py --train --dst dataset")
        print("python new.py --predict img2.jpg --model best_wall_phase2.keras")

