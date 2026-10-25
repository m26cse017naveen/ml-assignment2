# Machine Learning Assignment 2: 
**Student:** Naveen Pratap\
**Roll Number:** M26CSE017\
**Random Seed:** `26017` (derived from the digits in the roll number)\
**Dataset:** AT&T Cambridge Olivetti Faces (`K = 40` identities,
`D = 4096` pixel features)\
**Repository:**
[m26cse017naveen/ml-assignment2](https://github.com/m26cse017naveen/ml-assignment2)

This repository contains the notebook, experiment outputs, figures, and
environment configuration for Machine Learning Assignment 2. The
experiments study decision-tree baselines, bagging, multi-class SAMME
boosting, robustness to image occlusion, and empirical bias--variance
tradeoffs.

> **Reproducibility note:** The commands and paths below reflect the
> files currently described for this repository. If you later split
> notebook code into separate scripts or add files, update this README
> to match the final repository structure.

## 1. Repository Contents

``` text
ml-assignment2/
├── data/                   # Dataset or saved experiment data, if included
├── figures/                # Generated figures
├── logs/                   # Experiment outputs and logs
├── Assignment2.ipynb       # Interactive assignment notebook
├── environment.yml         # Conda environment definition
└── README.md               # Project overview and reproduction guide
```

Some data may be downloaded or generated while running the notebook.
Check the notebook cells for the exact data-loading and output paths.

## 2. Environment Setup

### Option A: Conda (recommended)

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
Anaconda, then run:

``` bash
git clone https://github.com/m26cse017naveen/ml-assignment2.git
cd ml-assignment2
conda env create -f environment.yml
conda activate ml-assignment2
jupyter notebook
```

Open `Assignment2.ipynb` in Jupyter and run the cells in order.

If the environment is already created, update it with:

``` bash
conda env update -f environment.yml --prune
```

### Option B: Python virtual environment

If you prefer `venv`, create and activate an environment, then install
the packages required by the notebook. Use `environment.yml` as the
source of truth for package requirements.

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install jupyter numpy scipy scikit-learn matplotlib pandas
jupyter notebook
```

**macOS/Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install jupyter numpy scipy scikit-learn matplotlib pandas
jupyter notebook
```

## 3. Dataset, Split & Occlusion Protocol

The Olivetti Faces dataset contains **400 grayscale face images** from
40 people, with 10 images per identity. Each image is `64 × 64` pixels
and is represented as 4096 raw pixel features.

The experiment split is identity-wise and reproducible using seed
`26017`:

-   **Training:** 6 images per identity (240 images total)
-   **Validation:** 2 images per identity (80 images total)
-   **Test:** 2 images per identity (80 images total)

The same saved split should be reused across experiments to ensure fair
comparisons and avoid data leakage. For the occlusion robustness
evaluation, a centered `16 × 16` region (rows `24:40`, columns `24:40`)
is set to zero in each test image.

The assignment uses raw flattened pixel values; do not apply PCA or
another dimensionality-reduction method if following the stated
assignment protocol.

## 4. Experiments

### Question 1 --- Decision Trees and Bagging

-   Train a decision-tree baseline using a grid of maximum depths
    `{2, 4, 8, 16, None}` and minimum leaf sizes `{1, 2, 4}`.
-   Select the configuration using **validation accuracy only**, then
    keep the selected setting fixed for subsequent comparisons.
-   Evaluate bagging across ensemble sizes
    `B ∈ {1, 5, 10, 25, 50, 100, 200}`.
-   Compare clean-test and center-occluded-test performance.
-   Examine bootstrap sample diversity, including the expected number of
    distinct examples in a bootstrap sample.

### Question 2 --- Multi-Class SAMME Boosting

-   Implement and evaluate SAMME boosting for the 40-class
    face-recognition task.
-   Explore tree depths `{1, 2, 3, 4}` and boosting rounds
    `{10, 25, 50, 100}`.
-   Inspect training dynamics, sample weights, prediction margins, and
    difficult training examples.
-   Compare boosting against bagging using test accuracy and training
    time where available.

### Question 3 --- Empirical Bias--Variance Decomposition

-   Repeat training over independent, stratified resamples.
-   Estimate empirical bias-squared and variance using multi-class
    squared loss, following the assignment's prescribed definition.
-   Compare a single decision tree, bagging, and the selected boosting
    configuration.
-   Study how tree depth affects the bias--variance tradeoff.

## 5. Running the Notebook

From the repository root:

``` bash
jupyter notebook Assignment2.ipynb
```

Run the notebook cells from top to bottom so that imports, data loading,
reproducible splits, model training, evaluation, and figures are created
in the intended order.

For reproducible results:

1.  Keep the random seed fixed at `26017`.
2.  Reuse the same train/validation/test split for the main experiments.
3.  Use validation results for model selection; reserve the test set for
    final evaluation.
4.  Do not change the preprocessing or occlusion protocol between
    models.
5.  Record package versions and execution settings if results differ
    across environments.

## 6. Results and Figures

The notebook is the primary place to inspect experiment outputs.
Generated files, when present, are organized in `logs/` and `figures/`.

When reporting results, include at least:

-   Decision-tree train, validation, and test accuracy, along with leaf
    statistics.
-   Bagging performance across ensemble sizes on clean and occluded test
    images.
-   SAMME performance across depths and rounds, plus the requested
    diagnostics.
-   Empirical bias-squared and variance estimates for the compared
    models.
-   A clear explanation of the main findings and limitations.

**Important:** Insert only values produced by your own completed run
into the final report. Results can vary if the split, seed, code, or
environment differs.

## 7. Report Submission

Prepare the final report in the required **CVPR two-column format**.
Verify that:

-   The report describes the methods and experiments actually
    implemented.
-   Figures and tables are readable and referenced in the text.
-   Metrics match the saved experiment outputs.
-   The PDF compiles successfully and meets the assignment's page and
    formatting requirements.
-   The repository link and student details are correct.
