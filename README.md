# 🏦 Predicting Loan Defaults via Random Forest Architecture: Theory, Mathematics, and Practical Implementation

This repository contains an end-to-end Machine Learning Engineering pipeline designed to predict credit default risk using the **Lending Club** dataset. The project bridges statistical learning theory with a modular, object-oriented software architecture.

> 📐 **Mathematician's Perspective & Author's Note:** 
> As a mathematics student, my structural coding experience is relatively new and adapting to enterprise software engineering is an ongoing learning curve. However, this project was architected with a strict focus on bridging rigorous **Statistical Learning Theory** and **Algorithmic Mathematics** (such as axis-parallel hyperplanes, entropy optimization, and variance reduction formulas) into functional production code. 
> 
> *The procedural implementation scripts inside the modules carry some artifacts of a bulit cloud environment (Google Colab/Drive trials), but the core emphasis remains on the foundational mathematics and the execution of 13 systematically tracked empirical experiments.*

---

## 📐 Theoretical & Mathematical Foundations

### 1. Axis-Parallel Hyperplane Splitting
At every decision node, the feature space $\mathbb{R}^d$ is disjointly partitioned by a cutting hyperplane defined as:
$$h(x) : w^T x + b = 0$$
In this Decision Tree implementation, the weight vector is restricted a priori to standard basis vectors $w \in \{e_1, e_2, \dots, e_d\}$, enforcing axis-parallel divisions where the multi-dimensional dot product collapses into a single feature value split equation: $x_j = -b$.

### 2. Information Gain Optimization via Bayes Reduction
To score and select the optimal candidate midpoint $v$ for continuous attributes ($X_j \le v$), the pipeline calculates the net reduction in global Dataset Entropy $H(D)$:
$$\text{Gain}(D, D_Y, D_N) = H(D) - \left[ \frac{n_Y}{n}H(D_Y) + \frac{n_N}{n}H(D_N) \right]$$
The mathematical core optimizes this split point by reducing continuous target probabilities into raw database row counts utilizing **Bayes Theorem**:
$$P(c_i | D_Y) = P(c_i | X \le v) = \frac{N_{vi}}{\sum_{j=1}^{k} N_{vj}}$$

### 3. Categorical Search Optimization (The Mirror Trick)
For a categorical feature with $d$ choices, the number of potential binary splits grows exponentially: $N_{\text{splits}} = 2^{d-1} - 1$. The system optimizes search routines via **The Mirror Trick**, terminating tree grid searches exactly at the halfway mark $\lfloor d/2 \rfloor$ since opposite subsets yield mathematically identical purity partitions.

### 4. Variance Reduction via Bagging
Individual decision trees exhibit high variance ($\sigma^2$). The Ensemble architecture constructs $B$ decorrelated decision trees on independent random rows chosen via Bootstrapping: $D_b \sim \text{Bootstrap}(D)$. The final ensemble variance converges to a rigorous limit:
$$\lim_{B \to \infty} \text{Var(Forest)} = \rho\sigma^2$$
By injecting random feature subsampling, the forest explicitly minimizes tree correlation ($\rho$) to prevent overfitting.

---

## 📁 Modular Code Implementation & Pipeline Architecture

The project moves away from legacy monolithic notebooks and encapsulates the entire machine learning lifecycle into an object-oriented, fully decoupled package structure:

* **`main.py`:** The master orchestrator script that runs the entire end-to-end pipeline.
* **`model_config.py`:** Stores all project configurations, tree hyperparameters, and platform file paths in one central place.
* **`scraper.py`:** Programmatically authenticates with the Kaggle API, handles secure data downloads, and extracts serialized zip archives.
* **`database.py`:** Handles safe relational database operations to read and write data frames into local SQLite database tables using secure transaction blocks.
* **`eda.py`:** Conducts data quality diagnostics, checks for high correlations, performs target leakage analysis, and flags highly dominant features.
* **`models.py`:** Controls cross-validation stratification rules, implements isolated robust scaling via `StandardScaler` (preserving binary dummy flags), and manages model routines.
* **`evaluation.py`:** Computes deep classification metrics (ROC-AUC, PR-AUC, F1, Recall) and plots domain-tailored Confusion Matrices and ROC curves.
* **`utils.py`:** Provides operational infrastructure automation, directory mapping, and automated experiment logging.

---

## 🔬 Critical Discussion: Categorical Imputation & Data Engineering Trade-off

A crucial data engineering decision was made during the data preparation phase regarding missing values in categorical columns:

* **The Strategy:** Instead of blindly filling empty categorical fields with the most frequent value (Mode), missing entries were imputed randomly by **preserving the natural categorical distribution frequency** of the column.
* **The Intention:** In highly imbalanced datasets, filling missing fields with the Mode biases the data further toward the majority class. Preserving the distribution frequency was intended to prevent information loss regarding minority categories (the rare cases we want to catch).
* **The Architectural Trade-off (Model Contradiction Risk):** While this prevents minority distribution decay, injecting random categorical values based on probability distribution can introduce artificial noise. Assigning different categories to rows with otherwise identical financial profiles may create logical contradictions within that specific feature. This increased noise might confuse the trees during hyperplane splitting, making it harder for the model to establish clear decision boundaries.

---
---

## 📊 Experimental Results & Performance Benchmarking

### 1. Comprehensive 13-Experiment Evolutionary Progress Matrix
The project followed a rigorous 13-stage milestone testing matrix to structurally counter severe Class Imbalance (~80% Fully Paid / ~20% Charged Off). Below is the complete evolutionary performance map showing how strategic tree pruning structurally resolved variance issues:

| Exp ID | Model Architecture & Hyperparameters | Accuracy | ROC-AUC | Precision (0) | Recall (0) | F1-Score (0) | PR-AUC |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | Default (Unweighted, `depth=None`, $n=200$) | 80.08% | 69.75% | 46.34% | 3.44% | 6.41% | ~22.00% |
| **Exp 3** | $w=12:1$ (`depth=None`, $n=200$) | 80.01% | 68.50% | 39.13% | 1.63% | 3.13% | ~19.90% |
| **Exp 6** | $w=5:1$ (`depth=None`, $n=200$) | 80.04% | 70.30% | 38.89% | 1.27% | 2.46% | ~19.90% |
| **Exp 8** | $w=5:1$ + `max_depth=17`, $n=100$ | 79.54% | 69.53% | 45.54% | 16.67% | 24.40% | ~27.80% |
| **Exp 7** | $w=5:1$ + `max_depth=15`, $n=100$ | 78.75% | 69.60% | 43.38% | 23.73% | 30.68% | ~31.50% |
| **Exp 9** | $w=5:1$ + `max_depth=13`, $n=100$ | 76.20% | 70.06% | 38.51% | 33.70% | 35.94% | ~36.50% |
| **Exp 10**| $w=5:1$ + `max_depth=11`, $n=100$ | 71.82% | 69.94% | 34.57% | 47.28% | 39.94% | ~39.50% |
| **Exp 11**| 🏆 $w=5:1$ + `max_depth=9`, $n=100$ | 66.26% | 70.79% | 32.07% | **62.86%** | 42.47% | ~43.50% |
| **Exp 12**| $w=5:1$ + `max_depth=9`, $n=50$ | 66.01% | 70.33% | 31.80% | 62.50% | 42.15% | ~43.10% |
| **Exp 13**| $w=5:1$ + `max_depth=9`, $n=25$ | 65.58% | 70.22% | 31.42% | 62.32% | 41.77% | ~42.80% |

---

### 2. Feature Selection Impact: Experiment 11 vs. The New Champion Model
Using Tree-based Feature Importance metrics, the pipeline dropped 118 noisy, high-cardinality dimensions, preserving only the **12 most prominent financial drivers**. Below is the comparative matrix demonstrating a massive data reduction with absolutely zero decay in predictive power:

| Metric / Structural Parameter | Exp 11 (130 Features Base) | New Champion Model (12 Pruned Features) | Delta / Engineering Gain |
|:---|:---:|:---:|:---|
| **Feature Dimensionality Count** | 130 | 12 | 🎯 **90.7% Data Storage Savings!** |
| **Risk Group (0) Recall** | 62.86% | 62.86% | 🤝 **Exactly Identical Target Accuracy!** |
| **Risk Group (0) F1-Score** | 0.4247 | 0.4166 | 📉 Negligible Delta (-0.0081) |
| **Good Client (1) Recall** | 62.86% | 65.67% | 📈 **+2.81% Net Improvement! (Rescued Revenue)** |
| **Global Accuracy** | 66.26% | 65.11% | ⚖️ Highly Stable Distribution |













## 📚 Academic References
* *Data Mining and Machine Learning: Fundamental Concepts and Algorithms (2nd Edition)* - Mohammed J. Zaki, Wagner Meira Jr. (Cambridge University Press, 2020).
* *An Introduction to Statistical Learning: with Applications in R (ISLR)* - Gareth James et al. (Springer, 2013).
