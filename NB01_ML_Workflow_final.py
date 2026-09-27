#!/usr/bin/env python
# coding: utf-8

# # The Machine Learning Workflow
# ## Machine Learning — BS Computer Science 3(3-0)
# 
# 
# 
# | | |
# |---|---|
# | **Notebook** | `NB01_ML_Workflow.ipynb` |
# | **Lecture** | Lectures 1–2 (Week 1) |
# | **CLO** | CLO-1 |
# | **Dataset** | Iris (bundled with scikit-learn) |
# 
# **Objective.** Establish the vocabulary, the notation and the split discipline that every later notebook depends on.
# 
# **By the end of this notebook you will be able to:**
# 
# 1. Load a dataset and read its shape aloud as 'n samples, m features'.
# 2. Distinguish supervised, unsupervised and reinforcement learning for a described problem.
# 3. Produce a stratified train/test split with a fixed seed and verify the class proportions.
# 4. State the five stages of the ML workflow.
# 
# ---

# ## 0. Setup and reproducibility

# In[1]:


# ---------------------------------------------------------------------
# Reproducibility header — required in EVERY notebook you submit
# ---------------------------------------------------------------------
import sys, numpy as np, pandas as pd, sklearn, matplotlib
import matplotlib.pyplot as plt

print('python      ', sys.version.split()[0])
print('numpy       ', np.__version__)
print('pandas      ', pd.__version__)
print('scikit-learn', sklearn.__version__)
print('matplotlib  ', matplotlib.__version__)

RANDOM_STATE = 1
rng = np.random.default_rng(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

plt.rcParams['figure.figsize'] = (8, 4.5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# ## 1. Loading a dataset
# 
# The Iris dataset contains 150 flowers of three species, each described by four measurements in centimetres. It is the textbook's running example (Raschka Ch. 1).
# 
# The universal convention is **rows are samples, columns are features**, so `X.shape` must always read `(n_samples, n_features)`.

# In[2]:


from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

print('X shape :', X.shape)          # (150, 4) -> 150 samples, 4 features
print('y shape :', y.shape)
print('features:', data.feature_names)
print('classes :', data.target_names)
print()
print('x_1^(150) — sepal length of flower 150 =', X[149, 0])


# > **Read the shape out loud in words**, not as numbers: *'150 samples, 4 features.'* If it ever prints `(4, 150)` your matrix is transposed and scikit-learn will train on nonsense.

# ## 2. A first look at the data
# 
# These six lines are the standard first look at any dataset in this course.

# In[3]:


df = pd.DataFrame(X, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(y, data.target_names)

print(df.shape)
display(df.head())
df.info()
display(df.describe().round(2))
print()
print('class balance:')
print(df['species'].value_counts(normalize=True))
print()
print('missing values per column:')
print(df.isnull().sum())


# **Interpretation (write two sentences).**
# 
# One row represents one iris flower, described by four measurements: sepal length, sepal width, petal length, and petal width. The features are measured in centimetres and are on similar numerical scales, but scaling can still matter for models that are sensitive to feature magnitude.
# 

# ## 3. Seeing whether the classes are separable
# 
# Before fitting anything, plot the data. A class-coloured scatter answers the question *are these classes separable at all?*

# In[4]:


fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

ax[0].hist(X[:, 2], bins=20, edgecolor='black')
ax[0].set_xlabel('petal length [cm]')
ax[0].set_ylabel('count')
ax[0].set_title('Distribution of one feature')

for label, marker in zip(np.unique(y), ('o', 's', '^')):
    ax[1].scatter(X[y == label, 2], X[y == label, 3],
                  marker=marker, alpha=0.8, edgecolor='k',
                  label=data.target_names[label])
ax[1].set_xlabel('petal length [cm]')
ax[1].set_ylabel('petal width [cm]')
ax[1].set_title('Are the classes separable?')
ax[1].legend()

plt.tight_layout()
plt.show()


# ## 4. The split — the experiment that measures generalisation
# 
# Generalisation is the only result that counts, so you must deliberately hide data from yourself. `random_state` makes the split reproducible; `stratify` preserves the class proportions on both sides.

# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

print('train', X_train.shape, ' test', X_test.shape)
print()
print('train class proportions:', np.round(np.bincount(y_train) / len(y_train), 3))
print('test  class proportions:', np.round(np.bincount(y_test) / len(y_test), 3))


# ### Exercise 4.1 — what stratification prevents
# 
# Build a deliberately imbalanced target (say 95% class 0), split it **without** `stratify`, and print the class counts in the test set. Repeat with `stratify`.

# In[6]:


# Create an imbalanced target with approximately 95% class 0
y_imb = np.array([0] * 95 + [1] * 5)

# Split without stratification and print the test class counts
_, _, _, y_test_unstrat = train_test_split(
    np.zeros((100, 1)), y_imb, test_size=0.3, random_state=RANDOM_STATE
)
print('Without stratify:', np.bincount(y_test_unstrat))

# Split with stratification and print the test class counts
_, _, _, y_test_strat = train_test_split(
    np.zeros((100, 1)), y_imb, test_size=0.3, random_state=RANDOM_STATE, stratify=y_imb
)
print('With stratify   :', np.bincount(y_test_strat))

# Explain the result
print('Without stratify, the minority class is not guaranteed to keep its original proportion in the test set.')


# ## 5. Classify the problem type
# 
# Complete the table below. For each problem give the learning type, the specific task, the target variable if one exists, and one risk of getting the prediction wrong.
# 
# | # | Problem | Learning type | Task | Target | Risk if wrong |
# |---|---|---|---|---|---|
# | 1 | Predict which students will fail from attendance and quiz history | Supervised | Classification | Fail / pass | A student who needs support may be missed. |
# | 2 | Group 50,000 supermarket customers for a campaign | Unsupervised | Clustering | None | Customers may be placed in unsuitable groups. |
# | 3 | Estimate apartment rent from area, location and rooms | Supervised | Regression | Rent | The estimated rent may be inaccurate. |
# | 4 | Teach a drone to reach a rooftop on minimum battery | Reinforcement learning | Control / sequential decision-making | None | The drone may waste battery or fail to reach the rooftop. |
# | 5 | Detect intrusions given 900 labelled attacks and 4M unlabelled sessions | Semi-supervised | Classification | Attack / normal | An intrusion may be missed. |
# | 6 | Compress 784-pixel digit images to two numbers for plotting | Unsupervised | Dimensionality reduction | None | Important information may be lost during compression. |
# 

# ## 6. The workflow, stated
# 
# Every project in this course — including your semester project — follows these five stages (Raschka Ch. 1, 'A roadmap for building machine learning systems').
# 
# 1. **Preprocessing** — missing values, encoding, scaling, splitting. 60–80% of real effort.
# 2. **Training and model selection** — fit candidates, tune on validation data.
# 3. **Evaluation** — the untouched test set, opened once.
# 4. **Interpretation** — which cases were wrong, and is there a pattern?
# 5. **Deployment and monitoring** — data drifts; an unmonitored model fails silently.

# ---
# ## Deliverable
# 
# Completed problem-classification table (section 5), the stratification exercise (4.1) with its written explanation, and the two-sentence interpretation in section 2.

# ---
# ## Submission checklist
# 
# Tick every box before you submit. Items 1–6 are the course reproducibility
# rules from Lecture 3 and are marked in the rubric.
# 
# - [ ] The random seed is fixed **and printed**
# - [ ] Library versions are printed in the first cell
# - [ ] The raw data file was **not** modified
# - [ ] Preprocessing happens **after** the split, inside a `Pipeline`
# - [ ] The dataset source and licence are stated in a markdown cell
# - [ ] **Kernel → Restart & Run All** completes without error
# - [ ] Every experiment is followed by a written interpretation
# - [ ] At least one wrong prediction or failure case is explained
# - [ ] Results are reported as **mean ± standard deviation** where cross-validated
# - [ ] Every plot has axis labels with units and a legend
# 
