# Thermal Imaging and CNN-based Machine Learning for Knee Pain Assessment

## Project Title & Abstract
**Project Title:** Thermal Imaging and CNN-based Machine Learning for Knee Pain Assessment

**Abstract:** This repository implements a reproducible, robust machine learning pipeline for the binary classification of clinical thermal knee images. By augmenting thermal imagery with tabular clinical metadata, the system is designed to predict clinically significant knee pain. The project encompasses rigorous exploratory data analysis (EDA) for both tabular and image modalities, strict patient-level data splitting to prevent leakage, manifest-driven dataset construction, and a parameterized convolutional neural network (`ConfigurableCNN`) trained using label smoothing. A comprehensive hyperparameter optimization stage ensures the model architecture and training dynamics are effectively tuned to address inherent clinical data challenges such as class imbalance.

---

## Repository Structure
The following core files constitute the analytical and modeling pipeline:

- `Tabular_EDA.ipynb` — Handles the exploratory data preparation for tabular clinical data. It standardizes identifiers, reshapes visual analog scale (VAS) data, merges labels and scores, and engineers features to export a clean tabular checkpoint.
- `Image_EDA.ipynb` — Implements a strict, multi-step data pipeline for thermal image preprocessing. It manages image-metadata reconciliation, anatomical side consolidation, and leak-proof train/validation/test splits, outputting the final dataset manifests.
- `hyperparameter_search.py` — The hyperparameter optimization driver. It orchestrates automated searches (using Random Search or Optuna) over a defined grid of architectural and learning parameters, evaluating and logging the best configurations.
- `train.py` — The primary training and fine-tuning script. It instantiates the model and data loaders from the generated manifests, executes the training loop with class-weighted label smoothing loss, and records the training history and test set predictions.
- `results_analysis.ipynb` — The post-training evaluation notebook. It ingests the exported training history and predictions to generate diagnostic visualizations (such as learning curves, confusion matrices, and ROC curves) and facilitates targeted error analysis.

---

## Data Preprocessing Pipeline
The data preparation workflow is meticulously divided into tabular and image processing to ensure data integrity:

- **Tabular Data Cleaning:** Clinical metadata is parsed to resolve missing values, encode categorical variables, and filter invalid diagnostic flags. The outcome is a consolidated baseline linking patient identifiers, visits, and pain outcome labels.
- **Image & Metadata Reconciliation:** Raw thermal images are cross-referenced against the cleaned tabular data. Filenames are parsed to extract patient and visit metadata, ensuring every image maps to a valid clinical record.
- **Normalization and Resizing:** Images are uniformly resized to a standard resolution (256x256) and normalized to stabilize gradient updates during network training.
- **Label Derivation and Consolidation:** Clinical labels are engineered into a binary outcome. Knee side variables (left/right) and multi-visit data are carefully mapped to guarantee accurate label assignment per image.
- **Leak-Proof Splitting:** Dataset manifests (`train_split.csv`, `val_split.csv`, `test_split.csv`) are generated with strict patient-level grouping. This ensures zero data leakage, meaning no single patient appears across multiple subsets.

---

## Model Architecture
- **ConfigurableCNN:** The core predictive model is a custom, lightweight convolutional neural network. The architecture utilizes increasing channel depths (24 -> 48 -> 96 -> 128) and adaptive average pooling to process spatial hierarchies. It is highly parameterized, allowing hyperparameters such as spatial dropout (`dropout_spatial`), dense dropout (`dropout_classifier`), and the activation function (e.g., `ReLU` or `GELU`) to be dynamically injected during optimization.
- **Label Smoothing Loss:** To address the ambiguity and noise inherent in clinical datasets, the network optimizes a custom `LabelSmoothingLoss`. This approach softens the target distributions, preventing the network from producing overconfident predictions and serving as a strong regularization technique.

---

## Hyperparameter Optimization
- **Search Strategy:** The pipeline supports systematic hyperparameter tuning using either Random Search or Bayesian Optimization via Optuna. Random Search operates as the default mechanism to efficiently explore high-dimensional spaces.
- **Hyperparameter Grid:** The search space explores the following parameters to find the optimal model configuration:
  - `dropout_spatial`: [0.2, 0.3, 0.4]
  - `dropout_classifier`: [0.25, 0.35, 0.5]
  - `label_smoothing`: [0.0, 0.05, 0.1]
  - `activation`: ["ReLU", "GELU"]
  - `weight_decay`: [1e-4, 5e-4, 1e-3]
  - `lr` (learning rate): [1e-4, 5e-4, 1e-3]
  - `threshold`: [0.5, 0.6, 0.7] (Optimization of the decision boundary to combat class imbalance)

---

## Training & Evaluation Pipeline
- **Reproducibility:** The training environment enforces strict deterministic execution by freezing pseudorandom number generators across Python, NumPy, and PyTorch, and restricting cuDNN heuristics.
- **Data Ingestion:** A custom PyTorch `Dataset` dynamically reads image paths from the CSV manifests, applies transformations, and yields tensor batches to the model.
- **Imbalance Mitigation:** Inverse-frequency class weights are computed dynamically from the training manifest and passed to the loss function to penalize majority-class bias.
- **Model Selection Criterion:** The training loop continuously monitors both training and validation losses. The optimal model state is captured and saved at the epoch where the **combined loss** (the absolute sum of training loss and validation loss) reaches its global minimum, indicating the best balance between data fitting and generalization.

*(Note: Specific quantitative results, including accuracy, F1 scores, and loss values, are strictly withheld from this repository pending formal submission and publication in a peer-reviewed medical journal.)*

---

## Results Analysis
The evaluation module provides a comprehensive suite of diagnostic tools without exposing raw patient data:
- **Learning Curves:** Visualizes epoch-by-epoch tracking of training versus validation metrics to diagnose variance and bias.
- **Confusion Matrix:** Aggregates test set predictions to illustrate the distribution of true positives, false positives, true negatives, and false negatives.
- **ROC-AUC Metrics:** Computes the Receiver Operating Characteristic curve and the Area Under the Curve to evaluate the model's discriminative capacity across all theoretical probability thresholds.
- **Error Analysis:** Isolates highest-confidence misclassifications (false positives and false negatives) to assist domain experts in understanding model failure modes.

---

## How to Run
To reproduce the pipeline, execute the files in the following sequential order:

1. **Tabular Preprocessing:**
   - Execute `Tabular_EDA.ipynb` to clean the clinical metadata and generate the structured data checkpoint.
2. **Image Preprocessing & Manifest Generation:**
   - Execute `Image_EDA.ipynb` to reconcile images with tabular data, perform quality control, and export the strict patient-level split manifests (`train_split.csv`, `val_split.csv`, `test_split.csv`).
3. **Hyperparameter Search:**
   - Execute the search script to identify the optimal network architecture and training parameters.
   - `python hyperparameter_search.py`
4. **Model Training:**
   - Execute the main training script to fine-tune the selected model configuration and export the test set predictions.
   - `python train.py`
5. **Results Analysis:**
   - Open and execute `results_analysis.ipynb` to process the exported training history and prediction logs into publication-ready visualizations and metric reports.

---

## Credits and Acknowledgments

**Machine Learning and CNN Research:**
- Nir Lapidot
- Shiri Guniman

**Project Mentors:**
- Dr. Sharon Yalov-Handzel
- Dr. Lilach Gavish
- Dr. Oshrit Hoffer

**Clinical Study, Data Collection, and Technical Assistance:**
This study is based on thermal imaging and data collected as part of a previous clinical study on the treatment of anterior knee pain in combatants. We extend our gratitude to the following individuals and organizations for their vital contributions:
- **Dr. Barzilay** (Principal Investigator)
- **Dr. Gam** (Military Collaborator)
- The Orthopedic Team, including **Dr. Spitzer**, **Dr. Friedman**, and **Dr. Lowe**, as well as the military physicians for their contribution to participant recruitment.
- **Prof. Gertz** and **Prof. Eisenkraft** from the Institute for Military Medicine.
- **Ms. Makhervax** for her technical assistance.

This research was supported by the **Milgrom Family Foundation for Research in Military Medicine**.
