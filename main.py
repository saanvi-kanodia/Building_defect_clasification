import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from skimage import color, filters, measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, blob_log
from scipy import fftpack

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 256
CLASSES = ["crack", "mold", "damp", "peeling", "spalling"]

FEATURE_NAMES = [
    "linearity","aspect","largest_area","area_ratio","num_comp",
    "boundary_grad","boundary_std","irregularity",
    "contrast","homogeneity","lbp_entropy",
    "low_freq","broadband","region_contrast",
    "blob_density","void_count"
]

# -----------------------------
# UTIL
# -----------------------------
def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def fft_features(gray):
    f = fftpack.fft2(gray)
    fshift = fftpack.fftshift(f)
    mag = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)

    max_r = np.sqrt(cx**2 + cy**2)
    low = dist <= 0.2 * max_r
    high = dist > 0.2 * max_r

    total = mag.sum() + 1e-8
    return mag[low].sum()/total, mag[high].sum()/total

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = normalize(gray)
    gray32 = gray.astype(np.float32)
    lab = color.rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    edges = cv2.Canny((gray*255).astype(np.uint8), 50, 150)
    thresh = filters.threshold_otsu(gray)
    mask = (gray < thresh).astype(bool)

    # STRUCTURE
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    linearity = 0
    if lines is not None:
        lengths = [np.hypot(x1-x2, y1-y2) for x1,y1,x2,y2 in lines[:,0]]
        linearity = max(lengths)/(np.sum(edges>0)+1e-8)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aspect = 0
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        aspect = max(w,h)/(min(w,h)+1e-8)

    labels = measure.label(mask)
    props = measure.regionprops(labels)
    num_comp = len(props)
    largest_area = max([p.area for p in props], default=0)
    area_ratio = largest_area/(IMG_SIZE*IMG_SIZE)

    # BOUNDARY
    grad = cv2.Sobel(gray32, cv2.CV_32F, 1, 1, ksize=3)

    if np.sum(edges) > 0:
        boundary_grad = grad[edges > 0].mean()
        boundary_std = grad[edges > 0].std()
    else:
        boundary_grad = 0
        boundary_std = 0

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c) + 1e-8
        peri = cv2.arcLength(c, True)
        irregularity = (peri**2) / area
    else:
        irregularity = 0

    # TEXTURE
    gray_u8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_u8, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]

    lbp = local_binary_pattern(gray_u8, 8, 1, 'uniform')
    hist,_ = np.histogram(lbp.ravel(), bins=10, range=(0,10), density=True)
    lbp_entropy = -np.sum(hist*np.log(hist+1e-8))

    # FREQUENCY
    low, high = fft_features(gray)

    # COLOR
    inside = lab[mask]
    outside = lab[~mask]

    region_contrast = (
        np.linalg.norm(inside.mean(axis=0) - outside.mean(axis=0))
        if len(inside) > 0 and len(outside) > 0 else 0
    )

    # NEW FEATURES

    # Blob density (mold)
    blobs = blob_log(gray, max_sigma=5, num_sigma=5, threshold=0.02)
    blob_density = len(blobs) / (IMG_SIZE * IMG_SIZE)

    # Void count (spalling)
    inv = (gray < gray.mean()).astype(np.uint8)
    v_contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    void_count = sum(1 for c in v_contours if cv2.contourArea(c) < 50)

    return [
        linearity, aspect, largest_area, area_ratio, num_comp,
        boundary_grad, boundary_std, irregularity,
        contrast, homogeneity, lbp_entropy,
        low, high,
        region_contrast,
        blob_density, void_count
    ]

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(folder):
    X, y = [], []
    for label in CLASSES:
        path = os.path.join(folder, label)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            feats = extract_features(img)
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

# -----------------------------
# FEATURE vs CLASS TABLE
# -----------------------------
def feature_class_table(X, y, le):
    print("\n=== Feature vs Class Contribution ===\n")

    X = np.array(X)
    y = np.array(y)

    print(f"{'Feature':20}", end="")
    for c in le.classes_:
        print(f"{c:12}", end="")
    print()

    for i, fname in enumerate(FEATURE_NAMES):
        overall_mean = X[:, i].mean()

        print(f"{fname:20}", end="")
        for cls_idx, cls_name in enumerate(le.classes_):
            cls_vals = X[y == cls_idx, i]
            cls_mean = cls_vals.mean()
            score = abs(cls_mean - overall_mean)
            print(f"{score:12.4f}", end="")
        print()

# -----------------------------
# TRAIN + EVAL
# -----------------------------
def train_and_evaluate():
    print("Loading data...")
    X_train,y_train = load_data("dataset/train")
    X_val,y_val = load_data("dataset/val")
    X_test,y_test = load_data("dataset/test")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='mlogloss', random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
        pred = model.predict(X_val)
        print(f"\n{name} VAL F1:", f1_score(y_val, pred, average='macro'))
        print(classification_report(y_val, pred, target_names=le.classes_))

        cm = confusion_matrix(y_val, pred)
        sns.heatmap(cm, annot=True,
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                    fmt='d')
        plt.title(f"{name} Confusion Matrix")
        plt.show()

    best = rf

    pred = best.predict(X_test)
    print("\nTEST REPORT")
    print(classification_report(y_test, pred, target_names=le.classes_))

    joblib.dump(best, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")

    # Feature-Class Table
    feature_class_table(X_train, y_train, le)

# -----------------------------
# PREDICT SINGLE IMAGE
# -----------------------------
def predict_image(img_path):
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")

    img = cv2.imread(img_path)
    feats = extract_features(img)
    feats = scaler.transform([feats])

    pred = model.predict(feats)[0]
    label = le.inverse_transform([pred])[0]

    print("Prediction:", label)

# -----------------------------
# MAIN
# -----------------------------
#if __name__ == "__main__":
#     train_and_evaluate()

predict_image("img6.jpg")