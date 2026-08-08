# E-Waste Image Classification

Classifies electronic-waste images into 10 categories, comparing a **custom CNN
baseline** against two **transfer-learning** backbones — **InceptionV3** and
**ResNet152** — on the same held-out test set with the same metrics.

Originally an internship group project (2024); this is a clean, fully
documented re-implementation.

---

## Dataset

- **Source:** Kaggle — `akshat103/e-waste-image-dataset` (`modified-dataset/` split).
- **Classes (10):** Battery, Keyboard, Microwave, Mobile, Mouse, PCB, Player,
  Printer, Television, Washing Machine.
- **Balance:** perfectly balanced — identical image count per class
  (imbalance ratio = 1.00), ~2240 training images.
- **Split:** author-provided `train / val / test` (test = 300 images, 30/class).
  Using the dataset's own split means the hold-out is fixed and standard, not a
  random cut we made.

---

## Results (300-image test set)

| Model        | Test Accuracy | Parameters |
|--------------|--------------:|-----------:|
| Custom CNN   |     ~0.69–0.73 |      1.2 M |
| InceptionV3  |          0.937 |     21.8 M |
| ResNet152    |          0.950 |     58.4 M |

**Reading of the result:** ResNet152 is the most accurate, but beats
InceptionV3 by only ~1.3 points for ~2.7× the parameters (and higher inference
cost). **Default ship recommendation: InceptionV3** — best accuracy-per-cost.
ResNet152 only if the hardest classes (e.g. Battery) are mission-critical.

> The custom CNN's accuracy varies more between runs (~0.69–0.73) than the
> fine-tuned models — smaller from-scratch models have higher variance, which is
> itself an argument for transfer learning's reliability.

---

## Method

1. **EDA** — confirmed perfect class balance, so plain accuracy is an honest
   headline metric and per-class F1 answers "which classes get confused" rather
   than "is accuracy lying".
2. **Input pipeline** (`tf.data`) — modern, GPU-friendly (cache + prefetch),
   with **per-model preprocessing**: `[0,1]` scaling for the custom CNN,
   `[-1,1]` for InceptionV3, Caffe-style mean subtraction for ResNet152.
   Wrong scaling silently breaks pretrained weights.
3. **Augmentation** (train only) — in-model Keras layers: horizontal flip,
   ±10% rotation, zoom, contrast. Auto-disabled at inference; val/test stay
   untouched so evaluation reflects real images.
4. **Custom CNN** — 4 conv blocks (filters 32→64→128→256), BatchNorm,
   GlobalAveragePooling, dropout. The baseline that quantifies how much
   transfer learning actually buys.
5. **Transfer learning** — two-phase per backbone:
   (1) freeze backbone, train new head; (2) unfreeze top ~30% at a tiny LR
   (1e-5), **BatchNorm layers kept frozen** to protect ImageNet running stats.
6. **Evaluation** — per-class precision/recall/F1, confusion matrices,
   accuracy-vs-parameters comparison, inference latency, and Grad-CAM
   interpretability.

---

## Key findings

- The custom CNN confused **shape-similar box appliances**
  (Player / Printer / Microwave) — a coarse-shape limit of a shallow net.
- Transfer learning fixed **exactly those classes**, as predicted — richer
  pretrained features resolve fine part-level detail. (Hypothesis → confirmed.)
- Remaining errors are **data ambiguity, not model failure**: laptop battery
  packs look like small devices, and the "Player" class is visually incoherent
  (at least one test image appears mislabeled). Next lever is data quality, not
  more model capacity.
- Grad-CAM heatmaps land on the objects, not the background — errors are honest
  semantic confusion, not a model cheating on artefacts.

---

## Notebook structure (`E_waste.ipynb`)

| Section | Content |
|--------:|---------|
| 1–3 | Setup, dataset auto-detection, `tf.data` loaders |
| 4–5 | EDA (class balance, samples) + class weights |
| 6–7 | Per-model preprocessing pipeline + augmentation + training helpers |
| 8 | Custom CNN (build + train + evaluate) |
| 9 | Two-phase transfer-learning helpers |
| 10–11 | InceptionV3 and ResNet152 |
| 12 | Head-to-head comparison (accuracy vs params) |
| 13 | Grad-CAM interpretability |
| 14 | Interview-defence cheat sheet |
| +extras | Inference latency, per-class F1 heatmap, misclassification gallery |

---

## Team & contributions (internship, 2024)

Three-member group project.

- **[You]** — **data augmentation strategy** and **ResNet152 fine-tuning**
  (two-phase freeze→fine-tune, BatchNorm handling).
- Member 2 — custom CNN architecture.
- Member 3 — InceptionV3 transfer learning.

Comparison framework, evaluation, and interpretability were shared.

---

## How to run

1. Open `E_waste.ipynb` in a **Kaggle GPU** kernel with the dataset attached.
2. Run top to bottom — the path/layout is auto-detected, so no edits needed.
3. First run downloads ImageNet weights (~90 MB InceptionV3, ~230 MB ResNet152).
4. If ResNet152 hits an out-of-memory error, set `BATCH_SIZE = 16` and re-run
   from the config cell.

**Stack:** TensorFlow / Keras · scikit-learn · matplotlib / seaborn.
