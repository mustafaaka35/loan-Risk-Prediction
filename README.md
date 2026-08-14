# 🏦 Predicting Loan Defaults via Random Forest Architecture: Theory, Mathematics, and Practical Implementation

This repository contains an end-to-end Machine Learning Engineering pipeline designed to predict credit default risk using the **Lending Club** dataset. The project bridges statistical learning theory with a modular, object-oriented software architecture.

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

## 📚 Academic References
* *Data Mining and Machine Learning: Fundamental Concepts and Algorithms (2nd Edition)* - Mohammed J. Zaki, Wagner Meira Jr. (Cambridge University Press, 2020).
* *An Introduction to Statistical Learning: with Applications in R (ISLR)* - Gareth James et al. (Springer, 2013).
