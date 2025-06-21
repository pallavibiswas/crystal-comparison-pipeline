# Crystal Comparison Pipeline


This repository contains a side-by-side implementation of two pipelines for crystal structure classification: one based on SOP & RSF descriptors, and the other using ACE descriptors. Both pipelines are benchmarked using a Multi-Layer Perceptron (MLP) classifier with consistent hyperparameter tuning and evaluation procedures.

---

## Project Structure

```
├── ace_pipeline
│   ├── data                # Preprocessed ACE descriptor data, alpha cutoffs, evaluation sets, and PCA visualizations
│   ├── features            # Raw ACE descriptor files (large)
│   ├── model               # Trained MLP model, prediction outputs, and summary logs
│   ├── evaluation          # Accuracy, confusion matrix plots, classification reports
│   └── train               # Training scripts, validation splits, and visualization tools
├── dc3_sop_rsf_pipeline
│   ├── data                # SOP+RSF features, distance cutoffs, alpha cutoffs, evaluation sets, and PCA visualizations
│   ├── features            # Raw dump files (large)
│   ├── model               # Trained MLP model, DC3 classifier, prediction outputs, and logs
│   ├── evaluation          # Accuracy, confusion matrix plots, classification reports
│   └── train               # Training scripts and visualization tools
└── README.md               # Project overview and documentation
```

---

## Methodology

This project contains two parallel pipelines for crystal structure classification: one using SOP & RSF descriptors (with DC3 refinement), and the other using ACE descriptors (with direct MLP classification). Both follow a rigorous development and evaluation process.

### 1. Feature Extraction
- **SOP & RSF Pipeline**: Extracts structural and radial symmetry features from dump files.
- **ACE Pipeline**: Uses Atomic Cluster Expansion (ACE) descriptors from external packages.

### 2. Data Preparation
- Features and labels are standardized into `X` and `y` arrays.
- Missing values are imputed using mean strategy.
- Data is split into training (80%) and testing (20%) sets for model evaluation.

### 3. Model Training (ACE Pipeline)
- A `sklearn` MLPClassifier is trained on the training set using:
  - Hyperparameter tuning (hidden layer size, learning rate)
  - Early stopping and validation score tracking
  - Standardized and imputed feature vectors
- Final model is saved and used for test set prediction.

### 4. DC3 Refinement (SOP & RSF Pipeline)
- A separate MLPClassifier makes initial lattice predictions.
- Then, **DC3 logic** refines these predictions using:
  - Alpha coherence score (with cutoff from training data)
  - Euclidean distance to perfect lattice points
- If a predicted solid structure is too far from its ideal, it is marked `-1` (unknown).
- If alpha score is below cutoff, it's labeled as `3` (liquid).

### 5. Evaluation
- Accuracy, precision, recall, and F1-score are calculated.
- Confusion matrices and classification reports are generated and visualized.
- PCA is optionally applied to visualize dimensionality and class separation.

---

## Results Summary

| Feature Type      | Accuracy (MLP) | Accuracy (Test) | F1 Score (Avg) | PCA Clustering     | Dimensionality           |
|-------------------|----------------|------------------|----------------|---------------------|---------------------------|
| SOP & RSF (DC3)   | ~88.1%         | ~74.9%           | ~0.86          | Clear clusters      | Lower variance retained   |
| ACE               | ~83.4%         | ~90.0%           | ~0.90          | Less separation     | Higher variance retained  |

---

## Citation / Credit

This repository builds upon the methodology and feature engineering proposed in the following works:

> Chung, H. W., Freitas, R., Cheon, G., & Reed, E. J. (2022). Data-centric framework for crystal structure identification in atomistic simulations using machine learning. Physical Review Materials, 6(4). https://doi.org/10.1103/physrevmaterials.6.043801 

> Freitas, R. (2022, March 2). *DC3 Github Repo*. GitHub. https://github.com/freitas-rodrigo/DC3

The SOP & RSF feature extraction + MLPClassifier has been used and adapted from the above credited github repository.

All dump file data and ACE descriptor information are credited to Dr. Ryan Sills and Dr. Qianqian Zhao. 

---

## Notes

- Some files exceed GitHub’s 100MB limit and are tracked via Git LFS.
- Ensure `read_functions.py` is available in both pipelines for data loading.

