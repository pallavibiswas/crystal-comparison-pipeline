<<<<<<< HEAD
# Crystal Comparison Pipeline
=======
# Crystal Structure Classification Pipeline
>>>>>>> 5031889 (Save local changes before pulling)

This repository contains a side-by-side implementation of two pipelines for crystal structure classification: one based on SOP & RSF descriptors, and the other using ACE descriptors. Both pipelines are benchmarked using a Multi-Layer Perceptron (MLP) classifier with consistent hyperparameter tuning and evaluation procedures.

---

## Project Structure

```
├── ace_pipeline
│   ├── data                # Preprocessed ACE descriptor data and combined datasets + PCA visualization
│   ├── features            # Raw ACE descriptor files (large)
│   ├── model               # Trained MLP model and summary logs
│   └── train               # Scripts for training, validation, and plots
├── dc3_sop_rsf_pipeline
│   ├── data                # Generated SOP+RSF features and combined datasets + PCA visualization
│   ├── features            # Raw dump files
│   ├── model               # Trained MLP model and summary logs
│   └── train               # Training scripts, validation splits, and visualization
└── README.md               # Project overview and documentation
```

---

## Methodology

Each pipeline performs the following steps:
1. Feature preparation (ACE or SOP/RSF)
2. Data formatting and cleaning
3. MLPClassifier training with:
   - Hyperparameter tuning for hidden layers and learning rates
   - Cross-validation for robustness
   - Learning curve analysis
4. PCA for dimensionality reduction and feature insight

---

## Results Summary

| Feature Type | Accuracy (MLP) | PCA Clustering | Dimensionality |
|--------------|----------------|----------------|----------------|
| SOP & RSF    | ~84.9%         | Clear clusters | Lower variance retained |
| ACE          | ~86.2%         | Less separation| Higher variance retained |

---

## Citation / Credit

<<<<<<< HEAD
This repository builds upon the methodology and feature engineering proposed in the following works:

> Chung, H. W., Freitas, R., Cheon, G., & Reed, E. J. (2022). Data-centric framework for crystal structure identification in atomistic simulations using machine learning. Physical Review Materials, 6(4). https://doi.org/10.1103/physrevmaterials.6.043801 

> Freitas, R. (2022, March 2). *DC3 Github Repo*. GitHub. https://github.com/freitas-rodrigo/DC3

The SOP & RSF feature extraction + MLPClassifier has been used and adapted from the above credited github repository.

All dump file data and ACE descriptor information are credited to Dr. Ryan Sills and Dr. Qianqian Zhao. 
=======
This repository is built upon the methodology and feature engineering from the following work:

> *[Placeholder for APA citation of Chung’s paper]*  
> *(Please add full reference here)*
>>>>>>> 5031889 (Save local changes before pulling)

---

## Notes

- Some files exceed GitHub’s 100MB limit and are tracked via Git LFS.
- Ensure `read_functions.py` is available in both pipelines for data loading.

