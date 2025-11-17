# Machine Learning Component (Phase 4)

**Status:** Prompts 1-4 Complete (Dataset Preparation + Models)
**Last Updated:** 11-17-2025
**Test Coverage:** 28 tests, all passing

---

## Overview

This module implements machine learning models to predict which vertices make good TSP tour starting points (anchors) based on structural graph features extracted in Phase 3.

The ML component transforms algorithmic insight (anchors matter) into practical speedup (predict good anchors without exhaustive search). Focus is on **interpretability over accuracy** - understanding WHY certain vertices make good anchors is as important as prediction performance.

---

## Architecture

### Core Design Pattern: Modular ML Pipeline

```
DatasetPreparator (Prompt 1)
    - Missing value handling
    - Outlier handling
    - Constant feature removal
    - Feature validation

TrainTestSplitter (Prompt 2)
    - Random split
    - Graph-based split
    - Stratified graph split
    - Graph-type holdout
    - Size-based holdout

LinearRegressionModel (Prompt 3)
    - OLS, Ridge, Lasso, ElasticNet
    - Coefficient extraction
    - Feature importance
    - Model diagnostics

TreeBasedModel (Prompt 4)
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - Feature importance
```

**Key Principles:**
1. **Interpretability**: Linear models preferred, coefficients must be extractable
2. **Generalization**: Multiple splitting strategies to test robustness
3. **Modularity**: Each component (prep, split, model) is independent
4. **Validation**: Comprehensive evaluation metrics (R², MAE, RMSE)

---

## Implemented Components (Prompts 1-4)

### 1. ML Problem Formulation and Dataset Preparation (Prompt 1)

**File:** `dataset.py`

**Classes:**
- `MLProblemType` - Enum for problem types (REGRESSION, BINARY, MULTICLASS, RANKING)
- `DatasetPreparator` - Handles data cleaning and preparation

**Problem Formulations Supported:**

**Regression** (PRIMARY):
- Target: Continuous anchor quality score (e.g., percentile rank, normalized tour weight)
- Evaluation: R², MAE, RMSE
- Advantage: Captures full range of anchor quality
- Use case: When you need to rank all vertices by quality

**Binary Classification**:
- Target: 1 if vertex in top-k%, 0 otherwise
- Evaluation: Accuracy, precision, recall, F1, ROC-AUC
- Advantage: Simpler problem, clearer decision boundary
- Use case: When you just need to avoid bad anchors

**Multi-class Classification**:
- Target: "excellent", "good", "mediocre", "poor"
- Evaluation: Accuracy, confusion matrix
- Advantage: Balances granularity and simplicity
- Use case: When you want quality tiers

**Ranking** (FUTURE):
- Target: Relative ordering of vertices
- Evaluation: Pairwise accuracy, ranking correlation
- Use case: When you need THE best anchor

**Data Preparation Features:**

```python
DatasetPreparator(
    problem_type=MLProblemType.REGRESSION,
    remove_constant_features=True,
    constant_threshold=1e-6,
    handle_outliers='clip',  # or 'remove', 'none'
    outlier_percentiles=(1.0, 99.0),
    handle_missing='mean',  # or 'median', 'remove', 'none'
    random_seed=42
)
```

**Missing Value Handling:**
- `mean`: Impute with feature mean
- `median`: Impute with feature median
- `remove`: Remove rows with missing values
- `none`: Leave missing values as-is

**Outlier Handling:**
- `clip`: Clip to percentile bounds (default: 1st-99th percentile)
- `remove`: Remove rows with outliers in any feature
- `none`: Keep all values

**Constant Feature Removal:**
- Removes features with variance < threshold (default 1e-6)
- Prevents numerical issues in linear models
- Reduces dimensionality

**Output:**
```python
X_clean, y_clean, metadata = preparator.prepare(X, y)

# metadata contains:
# - original_shape, final_shape
# - removed_features (constant features)
# - missing_info (imputation values, rows removed)
# - outlier_info (bounds, values clipped/removed)
```

---

### 2. Train-Test Split Strategy (Prompt 2)

**File:** `dataset.py`

**Classes:**
- `SplitStrategy` - Enum for split strategies
- `TrainTestSplitter` - Implements various splitting strategies
- `DatasetSplit` - Result container for splits

**Splitting Strategies:**

**1. Random Split (Baseline):**
```python
splitter = TrainTestSplitter(
    strategy=SplitStrategy.RANDOM,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
split = splitter.split(X, y)
```
- Pro: Simple, standard approach
- Con: Vertices from same graph can appear in train and test (overfitting risk)
- Use case: Baseline comparison

**2. Graph-Based Split:**
```python
splitter = TrainTestSplitter(strategy=SplitStrategy.GRAPH_BASED)
split = splitter.split(X, y, graph_ids=graph_ids)
```
- Pro: No graph appears in multiple splits (prevents information leakage)
- Con: Smaller effective sample size if graphs vary in vertex count
- Use case: Standard evaluation strategy

**3. Stratified Graph Split (RECOMMENDED):**
```python
splitter = TrainTestSplitter(strategy=SplitStrategy.STRATIFIED_GRAPH)
split = splitter.split(X, y, graph_ids=graph_ids, graph_types=graph_types)
```
- Pro: Ensures proportional representation of each graph type in train/val/test
- Pro: Prevents graph-level information leakage
- Con: Requires careful bookkeeping
- Use case: Primary evaluation strategy for diverse graph collections

**4. Graph-Type Holdout:**
```python
splitter = TrainTestSplitter(strategy=SplitStrategy.GRAPH_TYPE_HOLDOUT)
split = splitter.split(
    X, y,
    graph_ids=graph_ids,
    graph_types=graph_types,
    holdout_graph_type='random'
)
```
- Pro: Strongest generalization test - can model handle unseen distributions?
- Con: Model might fail if holdout type is too different
- Use case: Test generalization to new graph types

**5. Size-Based Holdout:**
```python
splitter = TrainTestSplitter(strategy=SplitStrategy.SIZE_HOLDOUT)
split = splitter.split(
    X, y,
    graph_ids=graph_ids,
    graph_sizes=graph_sizes,
    size_threshold=100
)
```
- Pro: Tests whether patterns scale to larger graphs
- Con: Large graphs might have different structural properties
- Use case: Test scalability

**Output:**
```python
split.X_train, split.y_train  # Training set
split.X_val, split.y_val      # Validation set
split.X_test, split.y_test    # Test set
split.train_graphs            # Graph IDs in each split
split.metadata                # Split statistics
```

---

### 3. Linear Regression Baseline (Prompt 3)

**File:** `models.py`

**Classes:**
- `ModelType` - Enum for model types
- `LinearRegressionModel` - Linear models (OLS, Ridge, Lasso, ElasticNet)
- `ModelResult` - Result container with predictions and metrics

**Supported Linear Models:**

**1. OLS (Ordinary Least Squares):**
```python
model = LinearRegressionModel(
    model_type=ModelType.LINEAR_OLS,
    standardize_features=False  # No regularization
)
```
- Formula: `y = β₀ + β₁×x₁ + ... + βₙ×xₙ`
- Training: Ordinary least squares
- Pro: Fast, interpretable, no hyperparameters
- Con: Can overfit with many features

**2. Ridge Regression (L2 Regularization):**
```python
model = LinearRegressionModel(
    model_type=ModelType.LINEAR_RIDGE,
    alpha=1.0,  # Regularization strength
    standardize_features=True  # Required for regularization
)
```
- Penalty: `α × Σ(βᵢ²)`
- Pro: Reduces overfitting, handles multicollinearity
- Con: All features retained (coefficients shrunk but not zeroed)
- Use case: When all features potentially relevant

**3. Lasso Regression (L1 Regularization):**
```python
model = LinearRegressionModel(
    model_type=ModelType.LINEAR_LASSO,
    alpha=0.1,
    standardize_features=True
)
```
- Penalty: `α × Σ|βᵢ|`
- Pro: Performs feature selection (drives some coefficients to zero)
- Con: Arbitrarily selects one from correlated features
- Use case: When feature subset sufficient, interpretability critical

**4. ElasticNet (L1 + L2):**
```python
model = LinearRegressionModel(
    model_type=ModelType.LINEAR_ELASTICNET,
    alpha=0.1,
    l1_ratio=0.5,  # 1.0 = Lasso, 0.0 = Ridge
    standardize_features=True
)
```
- Penalty: `α × (l1_ratio × Σ|βᵢ| + (1 - l1_ratio) × Σ(βᵢ²))`
- Pro: Combines benefits of Ridge and Lasso
- Use case: When feature groups are correlated

**Training and Evaluation:**

```python
# Fit model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
result = model.evaluate(X_test, y_test)

# result.metrics contains:
# - r2: Coefficient of determination (0-1, higher is better)
# - mae: Mean absolute error (lower is better)
# - rmse: Root mean squared error (lower is better)
# - mse: Mean squared error (lower is better)
```

**Feature Importance:**

```python
# Get coefficients
coefs = model.get_coefficients()
# {'feat_0': 2.34, 'feat_1': 1.12, ..., 'intercept': 0.5}

# Get feature importance (absolute coefficients)
importance = model.get_feature_importance()
# {'feat_0': 2.34, 'feat_1': 1.12, ...}

# Standardized coefficients (for comparison across features)
# Automatically computed if standardize_features=True
```

**Model Diagnostics:**

```python
diagnostics = model.get_diagnostics(X_test, y_test)
# {
#   'residuals': array of (y_true - y_pred),
#   'residuals_mean': should be ~0 for unbiased model,
#   'residuals_std': smaller = better fit,
#   'residuals_skew': ~0 = symmetric errors,
#   'residuals_kurtosis': ~0 = normally distributed errors
# }
```

**Interpretation Guidelines:**

1. **Coefficient Signs**: Match intuition?
   - Positive coefficient: Higher feature → higher anchor quality
   - Negative coefficient: Higher feature → lower anchor quality

2. **Coefficient Magnitudes**: Which features matter most?
   - Compare standardized coefficients (if features were standardized)
   - Or compare `|coef| × feature_std` for raw coefficients

3. **Statistical Significance**: (Not implemented in Prompts 1-4)
   - Future: Confidence intervals, p-values

4. **Model Diagnostics**:
   - Residuals centered at 0: Unbiased predictions
   - Residuals normally distributed: Linear model appropriate
   - Patterned residuals: Non-linearity or missing features

---

### 4. Tree-Based Models (Prompt 4)

**File:** `models.py`

**Classes:**
- `TreeBasedModel` - Tree models (Decision Tree, Random Forest, Gradient Boosting)

**Supported Tree Models:**

**1. Decision Tree:**
```python
model = TreeBasedModel(
    model_type=ModelType.DECISION_TREE,
    max_depth=5,
    min_samples_leaf=5,
    min_samples_split=10,
    random_seed=42
)
```
- Pro: Easy to visualize and interpret, handles non-linearities
- Con: High variance, overfits easily
- Use case: Quick exploration, visualizing decision rules

**2. Random Forest (RECOMMENDED):**
```python
model = TreeBasedModel(
    model_type=ModelType.RANDOM_FOREST,
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',  # or 'log2', int, float
    min_samples_leaf=5,
    random_seed=42
)
```
- Pro: Robust, handles non-linearities, good feature importance
- Pro: Bagging reduces variance
- Con: Less interpretable than single tree or linear model
- Use case: Comparison baseline, non-linear relationships

**3. Gradient Boosting:**
```python
model = TreeBasedModel(
    model_type=ModelType.GRADIENT_BOOSTING,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=5,
    random_seed=42
)
```
- Pro: Often highest accuracy, good feature importance
- Con: Slower training, easier to overfit, many hyperparameters
- Use case: Maximum performance (if interpretability not critical)

**Training and Evaluation:**

```python
# Same API as linear models
model.fit(X_train, y_train)
predictions = model.predict(X_test)
result = model.evaluate(X_test, y_test)

# result.metrics: same as linear models (r2, mae, rmse)
# result.coefficients: None (tree models don't have coefficients)
```

**Feature Importance:**

```python
importance = model.get_feature_importance()
# {
#   'feat_0': 0.32,  # Based on impurity reduction
#   'feat_1': 0.18,
#   ...
# }

# Importances sum to 1.0
# Higher = more important for prediction
```

**Tree vs Linear Models:**

When to prefer **Linear Models**:
- Interpretability is critical (research goal)
- Linear relationships expected
- Small feature set (< 50 features)
- Coefficient analysis needed

When to prefer **Tree Models**:
- Non-linear relationships suspected
- Feature interactions important
- Black-box prediction acceptable
- Higher accuracy needed

**Recommendation for Phase 4:**
1. Start with **Linear Ridge** (interpretable baseline)
2. Compare to **Random Forest** (non-linear baseline)
3. If RF much better → investigate non-linearities
4. If similar → prefer Linear for interpretability

---

## Usage Examples

### Complete ML Pipeline

```python
import pandas as pd
from ml import (
    DatasetPreparator,
    MLProblemType,
    TrainTestSplitter,
    SplitStrategy,
    LinearRegressionModel,
    TreeBasedModel,
    ModelType
)

# 1. Load features and labels (from Phase 3)
X = pd.read_csv('features.csv')
y = pd.read_csv('labels.csv')['anchor_quality']
graph_ids = pd.read_csv('labels.csv')['graph_id']
graph_types = pd.read_csv('labels.csv')['graph_type']

# 2. Prepare dataset
preparator = DatasetPreparator(
    problem_type=MLProblemType.REGRESSION,
    remove_constant_features=True,
    handle_outliers='clip',
    handle_missing='mean'
)
X_clean, y_clean, prep_metadata = preparator.prepare(X, y)

# 3. Split data
splitter = TrainTestSplitter(
    strategy=SplitStrategy.STRATIFIED_GRAPH,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
split = splitter.split(X_clean, y_clean, graph_ids=graph_ids, graph_types=graph_types)

# 4. Train linear model
linear_model = LinearRegressionModel(
    model_type=ModelType.LINEAR_RIDGE,
    alpha=1.0,
    standardize_features=True
)
linear_model.fit(split.X_train, split.y_train)

# Evaluate on validation set
val_result = linear_model.evaluate(split.X_val, split.y_val)
print(f"Ridge R²: {val_result.metrics['r2']:.3f}")
print(f"Ridge MAE: {val_result.metrics['mae']:.3f}")

# 5. Train tree model for comparison
tree_model = TreeBasedModel(
    model_type=ModelType.RANDOM_FOREST,
    n_estimators=100,
    max_depth=10,
    random_seed=42
)
tree_model.fit(split.X_train, split.y_train)

val_result_tree = tree_model.evaluate(split.X_val, split.y_val)
print(f"RF R²: {val_result_tree.metrics['r2']:.3f}")
print(f"RF MAE: {val_result_tree.metrics['mae']:.3f}")

# 6. Extract feature importance
linear_importance = linear_model.get_feature_importance()
tree_importance = tree_model.get_feature_importance()

# Top 5 features from each model
linear_top5 = sorted(linear_importance.items(), key=lambda x: x[1], reverse=True)[:5]
tree_top5 = sorted(tree_importance.items(), key=lambda x: x[1], reverse=True)[:5]

print("\\nLinear model top features:", [f for f, _ in linear_top5])
print("Tree model top features:", [f for f, _ in tree_top5])

# 7. Final evaluation on test set
test_result = linear_model.evaluate(split.X_test, split.y_test)
print(f"\\nTest R²: {test_result.metrics['r2']:.3f}")
```

---

## Test Coverage (28 Tests)

**Test File:** `src/tests/test_phase4_ml.py`

### Test Classes:

1. **TestDatasetPreparator** (7 tests)
   - Initialization
   - Missing value handling (mean, median, remove)
   - Constant feature removal
   - Outlier handling (clip, remove)
   - Metadata extraction

2. **TestTrainTestSplitter** (6 tests)
   - Random split
   - Graph-based split
   - Stratified graph split
   - Graph-type holdout
   - Size-based holdout
   - Split summary

3. **TestLinearRegressionModel** (9 tests)
   - OLS, Ridge, Lasso, ElasticNet fitting
   - Coefficient extraction
   - Feature importance
   - Model evaluation
   - Diagnostics
   - Error handling

4. **TestTreeBasedModel** (5 tests)
   - Decision tree, Random forest, Gradient boosting
   - Feature importance
   - Non-linearity handling vs linear models

5. **TestModelIntegration** (1 test)
   - Complete pipeline end-to-end

**Run tests:**
```bash
python3 -m unittest src.tests.test_phase4_ml -v
```

**Requirements:**
- pandas
- scikit-learn

---

## Future Work (Prompts 5-12)

### Prompt 5: Model Evaluation and Comparison (Not Implemented)
- Comprehensive evaluation framework
- Algorithm performance metrics (predicted vs best anchor)
- Paired statistical tests
- Per-graph-type analysis
- Effect size computation

### Prompt 6: Cross-Validation Strategy (Not Implemented)
- K-fold cross-validation
- Stratified k-fold
- Group k-fold (for graph-level splits)
- Nested cross-validation (for hyperparameter tuning)
- Leave-one-graph-out

### Prompt 7: Hyperparameter Tuning (Not Implemented)
- Grid search
- Random search
- Bayesian optimization
- Hyperparameter search spaces
- Validation set strategies

### Prompt 8: Feature Engineering for ML (Not Implemented)
- Feature scaling strategies
- Non-linear transformations
- Feature interactions
- Dimensionality reduction (PCA)
- Feature selection integration

### Prompt 9: Model Interpretation (Not Implemented)
- Coefficient analysis with confidence intervals
- Partial dependence plots
- SHAP values
- Case studies
- Feature insight synthesis

### Prompt 10: Prediction-to-Algorithm Pipeline (Not Implemented)
- Graph → features → prediction → algorithm execution
- Batch prediction
- Comparison to baselines
- Error analysis

### Prompt 11: Model Generalization Testing (Not Implemented)
- Cross-graph-type generalization
- Cross-size generalization
- Cross-distribution generalization
- Adversarial graph testing

### Prompt 12: Online Learning and Model Updates (Not Implemented)
- Incremental data collection
- Model versioning
- Performance tracking
- Active learning
- Model ensembling

---

## Design Decisions and Rationale

### Why Linear Regression as Primary Model?
- **Interpretability**: Coefficients directly show feature importance and direction
- **Simplicity**: Fewer hyperparameters, easier to understand and debug
- **Research Goal**: Understanding WHY vertices are good anchors matters
- **Baseline**: Strong baseline that complex models must beat

### Why Standardize Features?
- **Regularization**: Ridge/Lasso require standardized features for fair penalization
- **Coefficient Comparison**: Makes coefficients comparable across features
- **Numerical Stability**: Prevents dominance by large-scale features

### Why Multiple Splitting Strategies?
- **Robustness**: Single split might be lucky/unlucky
- **Generalization**: Different splits test different aspects of generalization
- **Graph Structure**: Standard random split leaks information via graph membership

### Why Both Linear and Tree Models?
- **Complementary**: Linear assumes linearity, trees capture non-linearities
- **Comparison**: If tree >> linear, suggests non-linear relationships
- **Interpretability vs Accuracy**: Linear for understanding, trees for performance

---

## Critical Principles

### Principle 1: Interpretability Over Accuracy
- Research goal is **understanding**, not just prediction
- Linear models preferred over black boxes
- Feature importance must be extractable
- Coefficients must make intuitive sense

### Principle 2: Prevent Information Leakage
- **Never** use test data during training
- **Never** put vertices from same graph in train and test (use graph-based splits)
- **Never** fit data transformations on test set (fit on train, apply to test)
- **Always** validate splitting strategy prevents leakage

### Principle 3: Test Generalization Rigorously
- Single train/test split insufficient
- Test on held-out graph types (transfer learning)
- Test on different graph sizes (scalability)
- Report performance degradation clearly

### Principle 4: Feature Scaling for Regularization
- **Always** standardize features for Ridge/Lasso/ElasticNet
- **Never** standardize for Decision Trees/Random Forests
- Fit scaler on training set, transform train/val/test
- Save scaler with model for deployment

---

## Integration with Other Phases

### Phase 3 (Features) → Phase 4 (ML)
- Feature extraction produces pandas DataFrames
- Column names preserved for interpretability
- Feature metadata (names, descriptions) used for reporting

### Phase 4 (ML) → Phase 6 (Analysis)
- Model coefficients enable publication-quality interpretations
- Feature importance drives research insights
- Predicted vs actual anchor quality for case studies

### Phase 4 (ML) → Phase 5 (Pipeline)
- Trained models saved for batch prediction
- Hyperparameter configurations stored
- Model versioning for reproducibility

---

## Common Pitfalls and Solutions

### Pitfall 1: Information Leakage via Graph Membership
**Problem:** Random split puts vertices from same graph in train and test
**Solution:** Use graph-based or stratified graph splitting

### Pitfall 2: Fitting Transformations on Test Data
**Problem:** Standardizing features using statistics from entire dataset
**Solution:** Fit StandardScaler only on training set, transform all sets

### Pitfall 3: Comparing Models with Different Splits
**Problem:** Model A on split 1, Model B on split 2 → unfair comparison
**Solution:** Use same splits for all models (set random_seed consistently)

### Pitfall 4: Overfitting to Validation Set
**Problem:** Repeatedly tuning hyperparameters on validation set
**Solution:** Use nested cross-validation or hold out separate test set

### Pitfall 5: Ignoring Feature Scaling
**Problem:** Ridge/Lasso penalize features unequally due to scale differences
**Solution:** Always standardize features for regularized linear models

---

## Performance Expectations (Success Criteria)

From guide/04_machine_learning_component.md:

**Minimum Success:**
- Linear regression R² > 0.5 on test set (explains >50% of variance)
- Predicted anchor produces tours within 10-15% of best-anchor on average
- Predicted anchor beats random anchor >70% of the time

**Good Success:**
- Linear regression R² > 0.7
- Predicted anchor within 5-10% of best-anchor
- Predicted anchor beats random >80% of time
- Top 5 features make intuitive sense

**Excellent Success:**
- Linear regression R² > 0.8
- Predicted anchor within 5% of best-anchor
- Model generalizes: performance on held-out types within 20% of training
- Clear, publishable insights about anchor quality predictors

---

## Version History

**v1.0 - 11-17-2025 (Prompts 1-4 Complete)**
- Dataset preparation: missing values, outliers, constant features
- Train/test splitting: 5 strategies (random, graph-based, stratified, holdout×2)
- Linear models: OLS, Ridge, Lasso, ElasticNet
- Tree models: Decision Tree, Random Forest, Gradient Boosting
- 28 comprehensive tests, all passing
- Requires pandas, scikit-learn

**Next Version (Prompts 5-12):**
- Model evaluation and comparison framework
- Cross-validation strategies
- Hyperparameter tuning utilities
- Feature engineering pipelines
- Model interpretation tools
- Prediction-to-algorithm pipeline
- Generalization testing
- Online learning and model updates

---

## Maintainer Notes

**Code Quality:**
- Consistent API across models (fit, predict, evaluate)
- Type hints throughout
- Docstrings for all public methods
- No warnings on typical inputs

**Documentation:**
- Inline comments explain WHY not WHAT
- Complex algorithms reference sklearn docs
- Edge cases documented explicitly

**Testing:**
- Unit tests for each model type
- Integration test for complete pipeline
- Real computation (no mocking)

**Dependencies:**
- pandas: DataFrame handling
- scikit-learn: Models, metrics, preprocessing
- numpy: Array operations

**Performance:**
- Linear models: < 1 second for 1000 samples, 50 features
- Random Forest: ~5 seconds for 1000 samples, 50 features, 100 trees
- Gradient Boosting: ~10 seconds for 1000 samples, 50 features, 100 trees

---

**Document Maintained By:** Builder Agent
**Last Review:** 11-17-2025
**Status:** Phase 4 (Prompts 1-4) production-ready
