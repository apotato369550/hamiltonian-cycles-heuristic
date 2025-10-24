# Metaprompt 4: Machine Learning Component

## Context
You have a dataset: vertices from diverse graphs, each with structural features and an anchor quality label. Now comes the ML phase: training models to predict which vertices make good anchors WITHOUT exhaustive search. This is the payoff - turning your algorithmic insights into fast, predictive heuristics.

This isn't about achieving 99% accuracy. It's about understanding WHICH features matter, WHY they matter, and whether predictions are good enough to improve algorithm performance in practice. Interpretability matters as much as accuracy.

---

## Prompt 1: ML Problem Formulation and Dataset Preparation

Clearly define the machine learning problem and prepare the dataset for training.

**Problem formulation options:**

**Regression:** Predict continuous anchor quality score
- Target: normalized tour weight, percentile rank, or optimality ratio
- Evaluation metrics: R², MAE, RMSE
- Advantage: captures full range of anchor quality
- Challenge: regression is harder than classification

**Binary classification:** Predict whether a vertex is a "good anchor"
- Target: 1 if vertex is in top k% of anchors, 0 otherwise
- Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC
- Advantage: simpler problem, clearer decision boundary
- Challenge: loses information about quality gradations

**Multi-class classification:** Predict anchor quality tier
- Target: "excellent", "good", "mediocre", "poor"
- Evaluation metrics: accuracy, confusion matrix, per-class metrics
- Advantage: balances granularity and simplicity
- Challenge: class imbalance possible

**Ranking:** Predict relative ordering of vertices within a graph
- Target: for any vertex pair, predict which is better
- Evaluation metrics: pairwise accuracy, ranking correlation, top-k accuracy
- Advantage: directly matches your use case (finding best anchor)
- Challenge: requires specialized ranking algorithms (RankNet, LambdaMART)

**Dataset preparation steps:**
1. Load feature matrix and labels
2. Handle missing values (imputation or removal)
3. Remove constant features (zero variance)
4. Handle outliers (clip, remove, or leave as-is)
5. Split data: training, validation, test sets
6. Consider stratification: ensure each split has representation of graph types

Think about: Which problem formulation best matches your research question? If you just need to avoid bad anchors, classification is fine. If you need THE best anchor, ranking might be better.

---

## Prompt 2: Train-Test Split Strategy

Design a splitting strategy that tests generalization properly. Don't just randomly shuffle!

**Random split (baseline):**
- Randomly assign vertex examples to train/val/test (e.g., 70/15/15 split)
- Pro: simple, standard approach
- Con: vertices from same graph can appear in training and test - overfitting risk

**Graph-based split:**
- Split by graph: some graphs in training, different graphs in test
- Pro: tests generalization to new graph instances
- Con: if graph types differ, test set might be too different

**Stratified graph split:**
- Ensure each split has proportional representation of graph types and sizes
- Pro: balanced evaluation across graph varieties
- Con: requires careful bookkeeping

**Graph-type holdout:**
- Train on some graph types (e.g., Euclidean and metric)
- Test on held-out type (e.g., random graphs)
- Pro: strongest generalization test - can your model handle unseen distributions?
- Con: model might fail if held-out type is too different

**Size-based holdout:**
- Train on small/medium graphs (20-100 vertices)
- Test on large graphs (200+ vertices)
- Pro: tests whether patterns scale
- Con: large graphs might have different structural properties

**Recommendation:** Use multiple strategies
- Primary evaluation: stratified graph split
- Generalization test 1: graph-type holdout
- Generalization test 2: size holdout

Think about: How do you ensure test set is representative? What if your graph collection is imbalanced (many Euclidean, few random)?

---

## Prompt 3: Linear Regression Baseline

Implement linear regression as your primary model. This should be your strongest model due to interpretability.

**Basic linear regression:**
- Model: anchor_quality = β₀ + β₁×feature₁ + β₂×feature₂ + ... + βₙ×featureₙ
- Training: ordinary least squares (OLS)
- Advantages: fast, interpretable, mathematically well-understood

**Regularized variants:**
- Ridge regression (L2 regularization): penalizes large coefficients, reduces overfitting
- Lasso regression (L1 regularization): drives some coefficients to zero, performs feature selection
- ElasticNet: combines L1 and L2

**Training considerations:**
- Feature scaling: standardize features to mean=0, std=1 (essential for regularized models)
- Hyperparameter tuning: cross-validate to find optimal regularization strength
- Multicollinearity: check condition number, variance inflation factor (VIF)

**Model interpretation:**
- Coefficient magnitudes: which features have strongest effects?
- Coefficient signs: do they match intuition? (e.g., higher MST degree → better anchor?)
- Feature importance: rank features by |coefficient| × feature_std

**Diagnostics:**
- Residual plots: are residuals random or patterned?
- Q-Q plots: are residuals normally distributed?
- Leverage and influence: are outliers driving the model?

Output:
- Trained model coefficients with confidence intervals
- Feature importance ranking
- Model performance metrics: R², MAE, RMSE
- Diagnostic plots

Think about: Do you need polynomial features? Should you include interaction terms? Start simple, add complexity only if needed.

---

## Prompt 4: Tree-Based Models

Implement tree-based models as comparison baselines. These handle non-linearities automatically.

**Decision Tree:**
- Single tree: easy to visualize and interpret
- Hyperparameters: max depth, min samples per leaf, min samples for split
- Pro: handles non-linearities, requires no feature scaling
- Con: high variance, overfits easily

**Random Forest:**
- Ensemble of decision trees with bagging
- Hyperparameters: number of trees, max depth, max features per split
- Pro: robust, handles non-linearities, provides feature importance
- Con: less interpretable than single tree

**Gradient Boosting (XGBoost, LightGBM, CatBoost):**
- Sequential ensemble of trees
- Hyperparameters: number of trees, learning rate, max depth, subsample ratio
- Pro: often highest accuracy, good feature importance
- Con: slower training, easier to overfit, many hyperparameters

**Training approach:**
- Use cross-validation for hyperparameter tuning
- Grid search or random search over hyperparameter space
- Early stopping for boosting: stop when validation performance plateaus

**Feature importance extraction:**
- For trees: count how often feature is used for splitting, weighted by improvement
- Plot feature importance bar charts
- Compare to linear regression coefficients: do models agree on important features?

Think about: Are tree models substantially better than linear regression? If not, prefer linear for interpretability. If yes, investigate what non-linearities they're capturing.

---

## Prompt 5: Model Evaluation and Comparison

Build comprehensive evaluation framework to compare models fairly.

**Performance metrics:**

For regression:
- R² (coefficient of determination): proportion of variance explained
- Mean Absolute Error (MAE): average absolute prediction error
- Root Mean Squared Error (RMSE): penalizes large errors more
- Median Absolute Error: robust to outliers

For classification:
- Accuracy: overall correctness
- Precision, Recall, F1: especially important if classes are imbalanced
- ROC-AUC: area under receiver operating characteristic curve
- Precision-Recall curve: for imbalanced datasets

**Algorithm performance metrics (the real test):**
- Tour quality from predicted anchor: use predicted best vertex, run single-anchor heuristic
- Compare to baselines:
  - Random anchor (average over multiple random selections)
  - Best anchor (exhaustive search)
  - Median anchor
  - Worst anchor
- Success rate: what percentage of time does predicted anchor beat random?
- Optimality gap: how close is predicted anchor to best anchor?

**Comparative analysis:**
- Paired comparisons: model A vs. model B on same test graphs
- Statistical significance: paired t-test or Wilcoxon signed-rank
- Effect size: not just "is it significant?" but "how big is the difference?"

**Per-graph-type analysis:**
- Break down performance by graph type: Euclidean, metric, random
- Identify where each model excels or struggles
- Create performance matrix: rows = models, columns = graph types, cells = R² or MAE

Think about: What's your primary metric? R² measures statistical fit, but tour quality measures practical utility. They might not align!

---

## Prompt 6: Cross-Validation Strategy

Implement rigorous cross-validation to avoid overfitting and get reliable performance estimates.

**K-fold cross-validation:**
- Split training data into K folds (e.g., K=5 or K=10)
- Train on K-1 folds, validate on remaining fold
- Repeat K times, average results
- Pro: uses all training data for both training and validation
- Con: standard k-fold might leak information if vertices from same graph are in different folds

**Stratified k-fold:**
- Ensure each fold has proportional representation of graph types
- Important for imbalanced datasets

**Group k-fold:**
- Critical for your problem: ensure all vertices from same graph stay together
- Prevents information leakage between training and validation
- Harder to get balanced folds if graphs vary in size

**Nested cross-validation:**
- Outer loop: k-fold for performance estimation
- Inner loop: k-fold for hyperparameter tuning
- Prevents hyperparameter overfitting
- Expensive: K_outer × K_inner training runs

**Leave-one-graph-out:**
- Train on all graphs except one, test on held-out graph
- Repeat for each graph
- Strongest generalization test
- Expensive for large graph collections

Implementation:
- Use sklearn's cross-validation utilities with custom splitters
- Track performance metrics across all folds
- Report mean and standard deviation of metrics
- Identify folds with unusually high/low performance (outlier graphs?)

Think about: How much do performance estimates vary across folds? High variance suggests overfitting or insufficient training data.

---

## Prompt 7: Hyperparameter Tuning

Design systematic hyperparameter optimization for each model type.

**Hyperparameters to tune:**

Linear models:
- Regularization strength (alpha for Ridge/Lasso)
- L1 ratio (for ElasticNet)

Tree models:
- Max depth
- Min samples per leaf
- Min samples for split
- Max features per split (for Random Forest)

Boosting models:
- Number of estimators
- Learning rate
- Max depth
- Subsample ratio
- Column subsample ratio

**Tuning strategies:**

**Grid search:**
- Exhaustively try all combinations in a predefined grid
- Pro: thorough, reproducible
- Con: exponentially expensive with many hyperparameters

**Random search:**
- Sample random combinations from hyperparameter distributions
- Pro: more efficient than grid search for high-dimensional spaces
- Con: might miss optimal combination

**Bayesian optimization:**
- Use previous trials to inform next hyperparameter choices
- Pro: efficient, finds good configurations quickly
- Con: more complex to implement

**Implementation approach:**
1. Define hyperparameter search space
2. Use cross-validation to evaluate each configuration
3. Track all trials: hyperparameters → performance
4. Select best configuration based on validation performance
5. Retrain on full training set with best hyperparameters
6. Evaluate on test set

Think about: Are you overfitting to the validation set through excessive tuning? Should you use a separate hyperparameter tuning set?

---

## Prompt 8: Feature Engineering for ML

Apply ML-specific feature transformations based on model requirements and exploratory analysis.

**Scaling and normalization:**
- Standardization: required for linear models with regularization
- Min-max scaling: useful if you want features in [0,1] range
- Robust scaling: use median and IQR, robust to outliers

**Handling skewed features:**
- Log transform: for right-skewed features (many small values, few large)
- Square root: milder transformation than log
- Box-Cox: automated power transform

**Handling outliers:**
- Clipping: cap values at percentiles (e.g., 1st and 99th)
- Winsorization: replace outliers with nearest non-outlier value
- Removal: drop vertices with outlier feature values (use cautiously)

**Feature interactions:**
- For linear models: manually create interaction terms if EDA suggests they matter
- For tree models: not necessary, trees capture interactions automatically

**Dimensionality reduction:**
- PCA: project features into principal components
- Pro: reduces correlation, might improve performance
- Con: loses interpretability
- Use only if you have strong evidence it helps

**Feature selection:**
- Remove low-variance features
- Remove highly correlated features (keep one from each correlated pair)
- Use L1 regularization for automatic selection
- Keep only top-k features by importance

Think about: Should transformations be fit on training data and applied to test data? Yes! Never fit transformations on test data (data leakage).

---

## Prompt 9: Model Interpretation and Explanation

Build tools to understand WHAT your models learned and WHY they make predictions.

**For linear models:**
- Coefficient table: feature name, coefficient value, confidence interval, p-value
- Standardized coefficients: multiply coefficient by feature std to get comparable effect sizes
- Feature contribution plots: for a specific prediction, show contribution of each feature

**For tree models:**
- Feature importance plots: bar chart of importance scores
- Partial dependence plots: how does predicted quality change as a single feature varies?
- Individual tree visualization: for decision trees, plot the tree structure
- SHAP values: modern explanation technique, shows feature contributions for each prediction

**Model comparison:**
- Do linear and tree models agree on important features?
- If they disagree, why? Non-linearities? Interactions?

**Case studies:**
- Pick interesting test graphs
- Show: predicted best vertex, actual best vertex, feature values of both
- Explain: why did the model choose this vertex? Which features drove the decision?

**Feature insights:**
- Which features are universally important across all models?
- Which features are important only for specific graph types?
- Are there surprising features? Features you expected to matter but didn't?

Think about: Can you form a simple rule from model coefficients? E.g., "good anchors have high MST degree and low mean edge weight." If so, you've achieved interpretable insight.

---

## Prompt 10: Prediction-to-Algorithm Pipeline

Build the complete pipeline: from new graph → feature extraction → prediction → algorithm execution → result.

**The pipeline:**
1. Input: a new graph instance (never seen during training)
2. Extract features for all vertices
3. Load trained model
4. Predict anchor quality score for each vertex
5. Select vertex with highest predicted score
6. Run single-anchor heuristic from that vertex
7. Return tour and performance metrics

**Comparison and validation:**
- Also run: random anchor, nearest neighbor baseline, best anchor (if computationally feasible)
- Compare tour qualities: does predicted anchor beat random? How close to best anchor?
- Track metadata: which vertex was predicted best, what were its features, what was its predicted score

**Batch prediction:**
- Apply to a collection of test graphs
- Aggregate results: average tour quality, success rate, runtime
- Identify failure cases: graphs where predicted anchor performed poorly

**Error analysis:**
- For each graph, compute prediction error: predicted score - actual score
- Identify patterns: does the model systematically overestimate or underestimate?
- Graph properties correlated with high error: large graphs? Non-metric graphs?

Think about: Is the model's top-1 prediction good enough, or should you use top-k? Maybe try top-3 predicted anchors and use the best tour?

---

## Prompt 11: Model Generalization Testing

Rigorously test how well models generalize to different graph types and sizes.

**Test 1: Cross-graph-type generalization**
- Train on Euclidean + metric graphs
- Test on random graphs
- Question: do learned features transfer to different graph structures?

**Test 2: Cross-size generalization**
- Train on graphs with 20-50 vertices
- Test on graphs with 100-200 vertices
- Question: do anchor quality patterns scale?

**Test 3: Cross-weight-distribution generalization**
- Train on graphs with uniform weight distributions
- Test on graphs with skewed distributions
- Question: is the model sensitive to weight distribution shape?

**Test 4: Adversarial graphs**
- Generate graphs designed to break heuristics: pathological cases
- Test whether model avoids predicting bad anchors on adversarial instances

**Generalization metrics:**
- Performance degradation: how much does test performance drop relative to validation?
- Consistency: does model rank deteriorate or just absolute scores?
- Failure modes: on what types of graphs does the model fail?

Think about: If generalization is poor, should you train separate models per graph type? Or add graph-type-specific features?

---

## Prompt 12: Online Learning and Model Updates

Design a system for updating models as you collect more data.

**Incremental data collection:**
- As you generate new graphs and benchmark them, add to training set
- Periodically retrain models with expanded dataset
- Track model version: model_v1, model_v2, etc.

**Performance tracking over versions:**
- Does performance improve with more data?
- At what dataset size does performance plateau?
- Learning curves: plot test performance vs. training set size

**Active learning:**
- Identify graphs where current model is uncertain or performs poorly
- Prioritize collecting more data (generating more graphs) of those types
- Iteratively improve model on weak points

**Model ensembling:**
- Train multiple models on different subsets or with different hyperparameters
- Ensemble predictions: average, voting, or weighted combination
- Often improves robustness

Think about: How do you decide when to update the model? After every X new graphs? When performance drops? On a schedule?

---

## Success Criteria

You've succeeded when:
- Linear regression achieves R² > 0.5 on test set (explains >50% of variance)
- Predicted anchor produces tours within 10-15% of best-anchor on average
- Predicted anchor beats random anchor >70% of the time
- You can explain the top 5 most important features and why they matter
- Model generalizes: performance on held-out graph types within 20% of training performance

## What NOT to Do

- Don't use deep learning or neural networks - they're overkill and uninterpretable
- Don't chase marginal accuracy improvements at the cost of interpretability
- Don't skip model interpretation - understanding WHY matters for research
- Don't trust a single train/test split - use cross-validation
- Don't ignore failure cases - they reveal model limitations

## Next Steps After Completion

With trained models, you can:
1. Write the paper: "Predicting Optimal TSP Heuristic Starting Points via Structural Graph Features"
2. Publish the dataset: graph instances + features + anchor quality labels
3. Release the model: trained coefficients + feature extraction code
4. Extend to other heuristics: can you predict good parameters for other TSP algorithms?
5. Apply to real-world instances: test on TSPLIB benchmarks

The ML component transforms your algorithmic insight (anchors matter) into practical speedup (predict good anchors without exhaustive search). This is the research payoff.
