# Beyond Pixels: Clinical Text-Supervised Multimodal Learning to Improve Unimodal Inference toward Tuberculosis Detection on Chest Radiographs

This repository contains the code for the study **“Beyond Pixels: Clinical Text-Supervised Multimodal Learning to Improve Unimodal Inference toward Tuberculosis Detection on Chest Radiographs.”** The project implements an alignment-regularized multimodal training strategy that uses clinically grounded text only during training as privileged supervision, while enabling **image-only inference** at deployment.

We combine chest radiographs (CXRs) with deterministic, structured TB reports and raw clinical notes to regularize a vision encoder. The text channel is used exclusively during training to shape the image representations; at test time, the model operates purely on CXRs, matching the realities of screening and low-resource workflows where radiology reports may be missing or delayed.

---

## Objective and main contributions

**Objective.**  
To leverage clinical text as a privileged training signal to improve the robustness and generalization of an image-only TB detector, without requiring text at inference time.

**Key contributions.**

- **Alignment-regularized multimodal training.**  
  A PyTorch implementation of a multimodal architecture that jointly optimizes image and text encoders with:
  - Cross-entropy losses on image and text logits, and
  - Alignment losses (cosine similarity + supervised NT-Xent contrastive loss) between image and text embeddings.

- **Privileged text supervision with image-only deployment.**  
  A training regime where deterministic structured reports and raw clinical notes supervise the image encoder during training, but only the image head is used at test time (e.g., in real-world CAD/TB triage).

- **Deterministic, PII-safe structured reports.**  
  A report generation pipeline that converts metadata into structured, PII-free TB descriptions (status, subtype, distribution, location, extras), producing one `.txt` file per image and a CSV with structured reports and quality flags.

- **Unimodal vs multimodal comparisons.**  
  Fully reproducible pipelines for:
  - Unimodal image-only training and evaluation,
  - Multimodal image+text training with raw and structured text.

- **Representation and performance analysis.**  
  Tools for:
  - ROC comparison of unimodal and multimodal models,
  - UMAP–based embedding visualization with clustering metrics,
  - Grad-CAM overlays to confirm lesion-focused attention.

---

## Repository organization and scripts/notebooks

> The core code is provided as Jupyter notebooks; you can run them as notebooks or convert them into `.py` scripts for automated pipelines. File names here refer to the notebook bases.

### 1. `train_val_test_data_creation`

**Purpose.**  
Create deterministic train/validation/test splits at the **patient level** for the Shenzhen CXR dataset, and store them as CSV files consumed by both unimodal and multimodal pipelines.

**What it does.**

- Reads the main metadata CSV (e.g., `shen_demo.csv`) containing filenames, patient IDs, TB labels, and quality flags.
- Filters to the target subset (e.g., frontal CXRs, TB vs normal, valid labels).
- Produces three CSVs:
  - `label_train.csv`
  - `label_valid.csv`
  - `label_test.csv`  
  each with at least:
  - `Filename` (image file name),
  - `Class` or `label` (0 = normal, 1 = TB).
- Ensures patient-level separation between splits.

**Role in the pipeline.**  
This is the **first step**. All training and evaluation notebooks read these CSVs from the dataset root.

---

### 2. `report_generation`

**Purpose.**  
Generate **deterministic structured reports** and optional **raw-style notes** for each CXR to use as the text modality during multimodal training.

**What it does.**

- Ingests the merged metadata (including TB status, subtype, distribution, and additional radiographic descriptors).
- Applies a rule-based template to build **PII-safe, deterministic** sentences such as:
  > “This pediatric chest radiograph shows active secondary tuberculosis on the right upper lobe with fibrous changes.”
- Produces:
  - A `.txt` file per image under a configurable `reports/` directory (one report per training/validation image).  
- Includes a **validator** that checks:
  - Every row in the CSV has a corresponding `.txt` file,
  - Template structure (e.g., must contain “chest radiograph”),
  - Consistency with `normal` vs `TB` labels,
  - Simple PII leakage patterns (digits, unexpected tokens).

**Role in the pipeline.**  
Run this **after** data splits. The multimodal notebook expects one report per image (for the training and validation sets) in the configured `reports_subdir`.

---

### 3. `image_text_unimodal`

**Purpose.**  
Train and evaluate **unimodal classifiers (Image/Text)**:

- Image-only models (e.g., VGG-11 based),
- Text-only models (CXR-BERT based),

using a shared `RunConfig` and metrics stack. This also includes image-only test inference and Grad-CAM visualization.

**What it does (image-only mode).**

- Defines a `RunConfig` object (modality, network backbone, batch size, paths to CSVs and images, output directory, etc.).
- Builds a PyTorch dataset + loader pipeline:
  - Reads CSVs (`label_train.csv`, `label_valid.csv`, `label_test.csv`) from `dataset_root`,
  - Uses Albumentations for resizing, CLAHE, normalization, and tensor conversion.
- Implements the **image encoder + classifier** (e.g., VGG-11 with a small MLP head).
- Implements a **text encoder** stack using CXR-BERT (from `transformers`) with a text fine-tuning policy (e.g., freeze all, unfreeze last encoder layer(s)), even for unimodal text runs.
- Defines training logic:
  - Standard cross-entropy loss.
  - Balanced sampling (via `WeightedRandomSampler` when enabled).
  - Logging of metrics per epoch: TP, FP, FN, TN, balanced accuracy, sensitivity, specificity, precision, NPV, F1, MCC, Cohen’s κ, ROC-AUC.
  - **Model selection by highest validation MCC**, saving `best_<NETWORK>_<MODALITY>_val_loss.pt` plus JSON logs and training/validation loss curves.
- Test inference:
  - Reloads the best checkpoint,
  - Evaluates on the test CSV via the same metrics stack,
  - Saves `test_metrics.csv` and per-sample `softmax_preds.csv`.

**Role in the pipeline.**  
Provides the **unimodal baseline** (image-only and text-only) against which multimodal models are compared. It also shares evaluation helpers with the multimodal pipeline.

---

### 4. `multimodal`

**Purpose.**  
Train an **image+text multimodal model** with alignment-regularized supervision and evaluate its **image-only head** for deployment. The architecture is designed so that:

- During training: both image and text branches are active, and
- During inference: only the image head is used (text is dropped).

**What it does.**

- Defines a multimodal network with:
  - Image encoder (e.g., VGG-11 up to a global pooling layer),
  - Text encoder (`microsoft/BiomedVLP-CXR-BERT-general` through `transformers`),
  - Projection heads producing `z_img` and `z_txt`,
  - Shared classifier head mapping embeddings to logits.
- Introduces a **text fine-tuning policy** for CXR-BERT (frozen vs partially unfrozen layers).
- Implements a `LossCfg` and a composite loss:
  - `L_ce_img` — cross-entropy on image logits,
  - `L_ce_txt` — cross-entropy on text logits (when available),
  - Alignment terms (enabled when `use_align_losses=True`):
    - `L_cos`: negative cosine similarity between `z_img` and `z_txt`,
    - `L_contrast`: supervised NT-Xent contrastive loss between image/text embeddings with temperature τ and weight λ.  
  - Total loss:  
    `L = L_{\text{ce,img}} + L_{\text{ce,txt}} + L_{\text{cos}} + \lambda \cdot L_{\text{contrast}}`
- Runs a **λ–τ grid search** over alignment hyperparameters, saving the best run per configuration by validation MCC into subdirectories such as `mm_grid1/`, `mm_grid2/`.
- Uses the same metrics stack as the unimodal pipeline, including saved curves and MCC-based early stopping.
- Provides image-only test inference for selected checkpoints:
  - Reloads multimodal checkpoint,
  - Calls only the image branch + classifier,
  - Saves `test_metrics.csv` and `softmax_preds.csv`.

**Role in the pipeline.**  
This is the central **multimodal + privileged information** training implementation. It is where structured vs raw text supervision is compared, and where multimodal training is shown to improve image-only inference.

---

### 5. `roc_comparison`

**Purpose.**  
Quantitatively compare **unimodal** and **multimodal** models via ROC curves and AUROC / best-MCC metrics on the **same** test set (Shenzhen, Montgomery, TBX11K, etc.).

**What it does.**

- Loads:
  - Unimodal VGG-11 checkpoint (image-only),
  - Multimodal (raw text) checkpoint,
  - Multimodal (structured text) checkpoint.
- Reconstructs:
  - The unimodal image head,
  - The multimodal image head (loading multimodal weights with `strict=False`).
- Builds two consistent loaders on the same test CSV:
  - `uni_loader`: Albumentations + OpenCV image pipeline,
  - `mm_loader`: PIL + torchvision transforms.
- For each model:
  - Applies the appropriate pipeline,
  - Computes AUROC and best MCC (with its threshold),
  - Smooths ROC curves for publication-quality plots.
- Outputs:
  - A combined ROC figure (PNG),
  - Logged AUROC values for each model.

**Role in the pipeline.**  
Used after training to illustrate the improvement in discrimination from multimodal training, especially the structured text model’s improvement in AUROC over unimodal and multimodal-raw variants.

---

### 6. `umap_visualization`

**Purpose.**  
Analyze and visualize the learned **image embeddings** from unimodal and multimodal models via UMAP, and quantify cluster quality.

**What it does.**

- Loads a selected checkpoint:
  - Unimodal or multimodal (image head only),
  - Hooks into the deepest convolutional block (pre-global-pooling feature map).
- Builds a dataset from a test CSV, using either:
  - The unimodal Albumentations pipeline (`PIPELINE="uni"`), or
  - The multimodal torchvision pipeline (`PIPELINE="mm"`).
- Extracts feature embeddings for all test images, then runs UMAP over a grid:
  - `n_neighbors ∈ {5, 10, 15, 20, 50, 100, 200}`,
  - `min_dist ∈ {0.0, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5}`,
  - `metric = "euclidean"`.
- For each grid point:
  - Generates a 2D scatter plot (Normal vs TB with distinct colors),
  - Computes:
    - Silhouette score,   
- Saves all figures into a directory derived from the checkpoint path.

**Role in the pipeline.**  
Provides evidence that multimodal training with structured text yields **better-separated image embeddings**, which is consistent with the observed generalization gains.

---

## Methods overview: multimodal training with privileged text supervision

At a high level, the method can be summarized as follows:

1. **Data preparation.**
   - Lung-cropped CXRs are preprocessed and resized (e.g., 224×224) with standardized normalization and CLAHE (only during training) via Albumentations.
   - Train/validation/test splits are created at the patient level, yielding `label_train.csv`, `label_valid.csv`, and `label_test.csv`.

2. **Text supervision.**
   - Clinical metadata is converted into:
     - **Deterministic structured reports** encoding TB status, location, pattern, and additional findings using a fixed, PII-safe template.
     - Optionally, **raw notes** that approximate free-text reports.
   - Each training/validation image is associated with one corresponding report file.

3. **Multimodal model.**
   - The image encoder (e.g., VGG-11 trunk) maps CXRs to embeddings \(z_{\text{img}}\).
   - The text encoder (CXR-BERT) maps tokenized reports to embeddings \(z_{\text{txt}}\), with a configurable fine-tuning policy controlling which layers are trainable.
   - A shared classifier maps either embedding to TB vs normal logits.

4. **Training objective.**
   - Supervised cross-entropy losses on image logits and text logits ensure each modality is discriminative for TB.
   - Alignment losses encourage **shared semantics**:
     - An explicit cosine similarity term pulls \(z_{\text{img}}\) and \(z_{\text{txt}}\) together for the same sample.
     - A supervised NT-Xent contrastive loss additionally encourages inter-class separation in the joint embedding space, scaled by λ and temperature τ.

5. **Privileged information / deployment.**
   - During training, both image and text are used.
   - At inference, only the **image head** is invoked. The text encoder and text head are not needed, making the method feasible in settings where radiology reports are unavailable at screening time.

6. **Evaluation and analysis.**
   - The same metrics (including MCC as the model selection criterion) are used for both unimodal and multimodal models.
   - ROC comparison across unimodal, multimodal-raw, and multimodal-structured models demonstrates performance gains.
   - UMAP and Grad-CAM analyses show that multimodal training reshapes the image representation: better cluster separation and more lesion-focused attention maps.

---

## Getting started

### Prerequisites

- **Python:** 3.11.11
- **GPU:** NVIDIA GPU with CUDA support (experiments were run on an A100 with CUDA 12.6)
- **OS:** Linux is recommended.
- Check **requirements.txt** file for other library requirements. 
