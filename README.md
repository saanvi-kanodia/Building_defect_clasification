# IVP Research: Wall Defect Classification

## Aim

This project focuses on automatic classification of common wall defects from images.

The goal is to identify defect type using:

- Deep image features from a CNN branch
- Hand-crafted (classical) visual features
- A fusion model that combines both

Target use case: support faster and more consistent visual inspection of building walls.

## Data Classes

The dataset is organized into train/validation/test splits. Each split contains five defect classes:

- `crack`
- `damp`
- `mold`
- `peeling`
- `spalling`

### Class Examples

| Class    | Example Image                                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| crack    | <img src="dataset/train/crack/cls01_001.jpg" alt="crack sample" width="220">                                                   |
| damp     | <img src="dataset/train/damp/dacl10k_v2_train_1126_jpg.rf.ddd594f23db165a12b444b0863a325d2.jpg" alt="damp sample" width="220"> |
| mold     | <img src="dataset/train/mold/cls00_102.jpg" alt="mold sample" width="220">                                                     |
| peeling  | <img src="dataset/train/peeling/cls03_001.jpg" alt="peeling sample" width="220">                                               |
| spalling | <img src="dataset/train/spalling/cls05_002.jpg" alt="spalling sample" width="220">                                             |

## Method

The project uses a hybrid (fusion) approach:

1. Image preprocessing

- Resize to 256x256
- Contrast enhancement (e.g., CLAHE on luminance)
- Noise-aware smoothing and normalization

2. Classical feature extraction

- Texture features: LBP entropy, GLCM properties
- Frequency features: DCT/FFT energy-based descriptors
- Color/statistical features in HSV/LAB/gray domains
- Defect-specific morphology cues (edge density, blob/void cues, roughness proxies)

3. Deep learning branch

- CNN-based image branch learns high-level visual patterns from raw images

4. Feature fusion and classification

- Classical vector and deep embedding are fused
- Final softmax layer predicts one of the five classes

5. Evaluation

- Validation and test reports
- Confusion matrix and class-wise precision/recall/F1

## Models Used

The repository contains and uses multiple model strategies:

- Fusion deep model (TensorFlow/Keras)
  - Script: `fusion_model.py`
  - Saved checkpoints: `best_wall_phase1.keras`, `best_wall_phase2.keras`

- Classical ML baselines
  - Script: `feature_vector_classical.py`
  - Models: Random Forest, XGBoost (and voting/ensemble variants in code)

- Feature contribution analysis
  - Script: `feature_analysis.py`
  - Produces class-vs-feature contribution tables (CSV outputs)

## Main Files

- `main.py` - project entry / utilities
- `fusion_model.py` - fusion training + inference pipeline
- `feature_vector_classical.py` - classical ML feature pipeline
- `feature_analysis.py` - feature attribution/contribution analysis

## Notes

- Large datasets and model artifacts are ignored by `.gitignore` for cleaner Git history.
- Keep directory names and class folders unchanged to avoid label mismatch.
