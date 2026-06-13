# SenoDetect
##Project Overview

Breast cancer is one of the leading causes of cancer-related mortality worldwide, and yet one of the most treatable when caught early. The challenge? Tumors in their early stages are often clinically silent — symptoms don't show up until the disease has already progressed. This is exactly where data-driven approaches can make a meaningful difference.

This project investigates the use of supervised machine learning for *binary classification* of breast tumors — distinguishing between *benign* (non-cancerous) and *malignant* (cancerous) cases. The goal is to assess whether a well-tuned ML pipeline can serve as a reliable decision-support layer alongside medical professionals.

The dataset used is the **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset**, curated by the University of Wisconsin Hospital. It contains 30 real-valued features computed from digitized Fine Needle Aspirate (FNA) biopsy images, capturing cell nuclei characteristics like texture, perimeter, smoothness, and fractal dimension.

Two supervised learning algorithms were implemented and benchmarked:
--> K-Nearest Neighbors (KNN) — a distance-based, instance-level classifier
--> Support Vector Machine (SVM) — a margin-maximizing classifier with strong theoretical guarantees



## K-Nearest Neighbors (KNN)

KNN is a lazy learning algorithm — it skips the training phase entirely and defers all computation to inference time. Classification works by:

- Computing the distance (typically Euclidean) between a query point and all training samples
- Identifying the *K* closest neighbors
- Assigning the class via majority vote

It's non-parametric, meaning it makes no assumptions about the underlying data distribution. This flexibility is both a strength and a limitation. When K is too small, the model is sensitive to noise and local outliers, leading to high variance. As K grows, the decision boundary smooths out — but too high and you risk underfitting.

One real concern with KNN is the **curse of dimensionality**: as the number of features increases, distances between points become less meaningful, which hurts classification performance. It's also slow at prediction time since it has to scan the entire training set for every new query.

---

## Support Vector Machine (SVM)

SVM is a discriminative classifier rooted in statistical learning theory. Rather than modeling class distributions, it finds the optimal separating hyperplane — the decision boundary that maximizes the margin between the two classes.

The data points that sit closest to this boundary are called support vectors, and they're the only ones that actually influence the model. This makes SVM:

- Robust to overfitting, especially in high-dimensional spaces
- Memory-efficient — most training points are irrelevant after fitting
- Theoretically grounded in structural risk minimization

For data that isn't linearly separable, SVM applies the kernel trick — implicitly mapping features into a higher-dimensional space where a clean linear boundary *can* exist, without ever computing the transformation explicitly. This is what makes it particularly well-suited for complex medical classification problems like this one.

---

## Results & Observations

Before passing features into either classifier, Principal Component Analysis (PCA) was applied as a preprocessing step — projecting the original 30 features onto a lower-dimensional subspace of uncorrelated principal components while retaining maximum variance.

Here's what the experiments revealed:

--> SVM consistently outperformed KNN in classification accuracy, particularly as training data increased — a direct outcome of its structural risk minimization principle enabling better generalization
--> KNN was highly sensitive to the choice of K — low values produced noisy, high-variance boundaries, while larger values generally stabilized accuracy
--> PCA had a much greater effect on SVM than on KNN — as more principal components were retained, SVM showed consistent accuracy gains, suggesting it benefits significantly from a clean, low-noise feature representation
--> Peak accuracy of 97.95% was achieved with PC = 1 and K = 9 — a notable result showing that even a heavily compressed single-component representation can retain strong discriminative signal

---

## Conclusion

This project shows that classical supervised learning — when paired with proper preprocessing and hyperparameter tuning — can reach clinical-grade accuracy on breast cancer classification. SVM proved to be the stronger model overall, offering more stable performance, better generalization, and superior handling of the high-dimensional feature space.

The broader takeaway is that ML-based diagnostic systems aren't here to replace clinicians — they're here to augment them. With further work on model interpretability (think SHAP values or LIME) and validation across diverse clinical datasets, pipelines like this could realistically integrate into real-world diagnostic workflows as a fast, consistent second opinion.
