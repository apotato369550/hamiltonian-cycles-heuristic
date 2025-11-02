# Metaprompt 6: Analysis, Visualization, and Research Insights

## Context
You've generated data, benchmarked algorithms, engineered features, and trained models. Now comes the most important research phase: extracting insights, understanding patterns, and answering scientific questions. This isn't just about making pretty plots - it's about turning data into knowledge and knowledge into publishable contributions.

This phase is where you discover which graph types favor which algorithms, what makes vertices good anchors, and whether your hypotheses hold up to statistical scrutiny. It's the difference between "I ran some experiments" and "I discovered fundamental patterns in TSP heuristics."

---

## Prompt 1: Exploratory Data Analysis Framework

Build a systematic EDA framework to understand your experimental data before jumping to conclusions.

**Dataset overview:**
- How many graphs in total? Breakdown by type and size?
- How many algorithm runs? Success rate?
- How many vertices in feature dataset? Any missing values?
- Data quality checks: outliers, invalid values, unexpected patterns

**Distribution analysis:**
- Tour weight distributions by algorithm
- Runtime distributions by algorithm and graph size
- Feature distributions: are they normal, skewed, bimodal?
- Anchor quality score distributions: balanced or skewed?

**Relationship exploration:**
- Algorithm performance correlations: if algorithm A beats B on graph G, does it beat B on similar graphs?
- Feature correlations: which features move together?
- Performance vs. graph properties: does tour quality correlate with metricity? Size? Weight range?

**Visual EDA tools:**
- Pair plots: scatter plot matrix of key features vs. anchor quality
- Correlation heatmaps: feature correlations, algorithm correlations
- Distribution plots: histograms, KDE plots, violin plots
- Time series plots: if experiments ran over time, any temporal patterns?

**Anomaly detection:**
- Outlier graphs: which graphs are unusually easy or hard?
- Outlier algorithms: runs with unusually long runtime or bad performance
- Outlier features: vertices with extreme feature values

**EDA outputs:**
- Summary statistics tables
- Distribution plots for all key variables
- Correlation matrices
- List of outliers/anomalies with explanations
- Initial insights to guide deeper analysis

Think about: EDA is open-ended exploration. Don't go in with fixed expectations. Be curious - let the data surprise you.

---

## Prompt 2: Algorithm Performance Comparison Analysis

Design rigorous comparative analysis to determine which algorithms perform best under which conditions.

**Overall performance ranking:**
- Aggregate tour quality across all graphs: mean, median, std dev
- Rank algorithms by average performance
- Statistical significance: is algorithm A's advantage over B significant?
- Effect size: not just significant, but how big is the difference?

**Win/loss/tie analysis:**
- For each graph, determine which algorithm produced best tour
- Compute win rates: "Best-anchor wins 60% of graphs, Nearest-neighbor 25%, others 15%"
- Pairwise win rates: "Best-anchor beats Nearest-neighbor 75% of the time"
- Visualize as heatmap or tournament bracket

**Performance by graph type:**
- Break down results by graph type (Euclidean, metric, random)
- Which algorithms excel on which types?
- Statistical tests: is performance difference between types significant?
- Interaction effects: does algorithm ranking change with graph type?

**Performance by graph size:**
- Plot tour quality vs. graph size for each algorithm
- Does performance degrade with size? At what rate?
- Are some algorithms more scalable than others?
- Scaling analysis: fit power laws or other models

**Performance distributions:**
- Box plots: show full distribution of tour qualities per algorithm
- Identify variance: some algorithms consistent, others high-variance
- Worst-case analysis: what's each algorithm's worst performance?
- Risk assessment: which algorithm is safest bet?

**Optimality gap analysis:**
- For small graphs with known optima
- Compute optimality gaps: (heuristic - optimal) / optimal
- Which algorithms get closest to optimal?
- Does optimality gap correlate with graph properties?

Think about: Don't just report averages. Show distributions, worst cases, variance. A consistent algorithm that's sometimes not the best might be more valuable than a high-variance algorithm that occasionally wins big.

---

## Prompt 3: Graph Property and Algorithm Performance Relationship Analysis

Investigate which graph properties predict algorithm performance.

**Property extraction:**
- For each graph, compute properties:
  - Size
  - Metricity score
  - Symmetry
  - Weight distribution statistics (mean, std, skewness, kurtosis)
  - Density
  - Diameter
  - Graph type (categorical)

**Univariate relationships:**
- For each property, correlate with algorithm performance
- Scatter plots: property vs. tour quality, colored by algorithm
- Examples:
  - Does metricity score predict Christofides advantage?
  - Do sparse graphs favor certain heuristics?
  - Does weight range affect performance variance?

**Multivariate relationships:**
- Multiple regression: predict algorithm performance from multiple graph properties
- Decision trees: "If metricity > 0.8 AND size < 50, use algorithm X"
- Clustering: group graphs by properties, analyze algorithm performance per cluster

**Algorithm selection rules:**
- Can you create simple rules for algorithm selection?
- Example: "Use Christofides on metric graphs with size < 100, otherwise use best-anchor"
- Validate rules: what's the success rate of rule-based selection?

**Failure mode analysis:**
- Identify graphs where all algorithms perform poorly (hard instances)
- What properties do hard instances share?
- Identify graphs where specific algorithms fail catastrophically
- What triggers these failures?

**Visualization approaches:**
- Scatter plot matrix: graph properties vs. algorithm performance
- 3D plots: two properties + performance
- Heatmaps: property bins × algorithms, cells show average performance
- Decision boundary plots: regions where each algorithm is best

Think about: This analysis reveals fundamental insights: "Metric graphs favor X algorithm because triangle inequality enables Y property." These become your paper contributions.

---

## Prompt 4: Feature Importance and Anchor Quality Analysis

Deep dive into which features predict anchor quality and why.

**Univariate feature analysis:**
- For each feature, compute correlation with anchor quality
- Create scatter plots: feature vs. anchor quality
- Identify top-k most correlated features
- Test statistical significance of correlations

**Feature importance from models:**
- Extract feature importance from trained models
- For linear models: absolute coefficient values (standardized)
- For tree models: Gini importance or permutation importance
- Compare importance across models: do they agree?

**Feature contribution analysis:**
- For specific vertices (case studies):
  - Show feature values
  - Show model prediction
  - Show actual anchor quality
  - Explain: which features drove the prediction?
- Use SHAP values for detailed explanations

**Feature interactions:**
- Are there important feature interactions?
- Example: "MST degree matters, but only when mean edge weight is low"
- Interaction plots: show how one feature's effect depends on another
- Fit models with explicit interaction terms

**Feature groups:**
- Compare importance of feature groups:
  - Weight-based features
  - Topological features
  - MST-based features
- Which group is most predictive overall?
- Does this vary by graph type?

**Feature redundancy:**
- Identify highly correlated features
- Can you remove redundant features without performance loss?
- Build minimal feature set: smallest set achieving 95% of full model performance

**Visualization:**
- Feature importance bar charts
- Partial dependence plots: show how prediction changes with one feature
- 2D feature interaction plots
- Feature correlation network graphs

Think about: Feature importance is central to your research story. If you can say "MST degree is the #1 predictor of anchor quality because...", that's a publication-worthy insight.

---

## Prompt 5: Model Performance and Generalization Analysis

Analyze ML model performance in depth to understand strengths, weaknesses, and reliability.

**Prediction accuracy analysis:**
- Scatter plots: predicted vs. actual anchor quality
- Residual plots: prediction errors vs. actual values
- Error distribution: are errors symmetric or skewed?
- Outlier analysis: which vertices have large prediction errors?

**Model comparison:**
- Compare multiple models (linear, tree-based, etc.)
- Performance metrics table: R², MAE, RMSE per model
- Statistical tests: are differences significant?
- Visualization: overlaid predicted vs. actual plots for each model

**Generalization analysis:**
- Compare validation and test performance
- If test performance much worse, investigate overfitting
- Per-graph-type performance: does model generalize across types?
- Per-size performance: does model generalize to larger graphs?

**Error pattern analysis:**
- When does the model make large errors?
- Do errors correlate with graph properties?
- Systematic bias: does model over-predict or under-predict in certain regimes?
- Confusion matrix (for classification): which classes are confused?

**Top-k prediction performance:**
- Even if exact predictions aren't perfect, how often is the true best anchor in top-k predictions?
- Plot: accuracy vs. k (k=1, 5, 10, 20)
- Compare to random baseline: is top-k better than random selection?

**Practical performance:**
- Use predicted best anchor to run single-anchor heuristic
- Compare tour quality to:
  - Random anchor baseline
  - Best anchor (exhaustive search)
  - Other algorithms
- This is the real test: does ML improve practical algorithm performance?

Think about: Perfect prediction isn't the goal. If your model gets you 90% of the way to best-anchor with 1/n the computation, that's a huge win.

---

## Prompt 6: Case Study Deep Dives

Select interesting specific cases for detailed narrative analysis.

**Successful prediction case:**
- Pick a graph where model predicted the best anchor correctly
- Show:
  - Graph visualization
  - Feature values for predicted best vertex
  - Why model chose it (feature contributions)
  - Comparison of predicted-best tour vs. other tours
- Narrative: "The model identified vertex 12 as the best anchor because it has high MST degree (5) and low mean edge weight (23.4)..."

**Failed prediction case:**
- Pick a graph where model prediction was poor
- Analyze:
  - What went wrong?
  - Were features misleading?
  - Is this graph an outlier?
  - What could improve prediction?
- Narrative: "The model failed on this graph because... This reveals a limitation..."

**Graph type comparison case:**
- Pick similar-sized graphs of different types
- Show how anchor quality patterns differ
- Explain which features matter for each type
- Narrative: "On Euclidean graphs, geometric centrality matters. On random graphs, MST structure matters more..."

**Counterintuitive case:**
- Find a vertex that looks like it should be a good anchor but isn't (or vice versa)
- Explain the counterintuitive finding
- Narrative: "Vertex 5 has the lowest total edge weight, suggesting centrality, but it's a poor anchor because its two cheapest edges point in the same direction, creating a dead-end..."

**Algorithm comparison case:**
- Pick a graph where best-anchor beats nearest-neighbor by large margin
- Show the tours visually, explain why anchor heuristic wins
- Narrative: "On this metric graph, pre-committing to edges (3,7) and (3,11) from anchor vertex 3 prevented the greedy heuristic from getting trapped in expensive edges later..."

Think about: Case studies make your research tangible. They're the stories you'll tell in talks and the examples that go in papers.

---

## Prompt 7: Visualization Suite for Publication

Create publication-quality visualizations that communicate findings clearly.

**Algorithm comparison visualizations:**
- Box plots: tour quality distributions per algorithm
  - Clean axes, readable fonts, legend
  - Color scheme: colorblind-friendly
- Line plots: performance vs. graph size
  - Multiple lines (one per algorithm), confidence intervals
- Bar charts: win rates per algorithm
  - Sorted by performance

**Feature importance visualizations:**
- Horizontal bar chart: top-10 features by importance
  - Sort by importance, include error bars
- Heatmap: feature correlation matrix
  - Diverging colormap (red-white-blue)
- Partial dependence plots: show how predictions change with feature values

**Model performance visualizations:**
- Scatter plot: predicted vs. actual anchor quality
  - Diagonal line showing perfect prediction
  - Color points by graph type
- Residual plot: errors vs. predicted values
  - Horizontal line at zero
- ROC curve (for classification): true positive rate vs. false positive rate

**Graph structure visualizations:**
- For small example graphs:
  - Graph layout with edge weights
  - Highlight tours from different algorithms in different colors
  - Highlight predicted best anchor vertex

**Summary visualizations:**
- Table: algorithm performance comparison
  - LaTeX formatted for papers
- Heatmap: algorithm vs. graph type, cells show win rate or average performance

**Style guidelines:**
- High resolution (300 DPI for papers)
- Consistent color schemes across all plots
- Readable fonts (at least 10pt when printed)
- Clear axis labels with units
- Legends positioned clearly
- Minimal chartjunk: no 3D effects, no unnecessary gridlines
- Accessible: colorblind-friendly palettes, sufficient contrast

Think about: Reviewers often look at figures before reading text. Your visualizations should tell the story standalone.

---

## Prompt 8: Statistical Rigor and Hypothesis Testing

Apply proper statistical methods to ensure findings are robust.

**Hypothesis formulation:**
- Null hypothesis: "Algorithm A and B have equal performance"
- Alternative hypothesis: "Algorithm A outperforms B"
- Set significance level: α = 0.05 (or 0.01 for stronger claims)

**Test selection:**
- Paired tests: each graph tested with both algorithms, so pairs are correlated
- Parametric (t-test) vs. non-parametric (Wilcoxon signed-rank):
  - Use parametric if differences are normally distributed
  - Use non-parametric if skewed or with outliers
- Multiple comparisons: if testing many algorithm pairs, apply correction (Bonferroni, Holm-Bonferroni)

**Effect size measurement:**
- Cohen's d: standardized mean difference
  - Small: d ≈ 0.2, Medium: d ≈ 0.5, Large: d ≈ 0.8
- Practical significance vs. statistical significance:
  - A 1% improvement might be statistically significant but practically unimportant
  - Report both: "Algorithm A beats B by 12% on average (p<0.001, d=0.7)"

**Confidence intervals:**
- Don't just report point estimates (mean performance)
- Report 95% confidence intervals: "Algorithm A: 245.3 ± 8.2"
- If confidence intervals overlap, difference may not be significant

**Multiple testing correction:**
- If testing 10 algorithm pairs, chance of false positive increases
- Apply Bonferroni correction: divide α by number of tests
- Or use False Discovery Rate (FDR) control

**Power analysis:**
- Did you have enough data to detect real differences?
- Compute statistical power: probability of detecting a true effect
- If power is low (<0.8), might need more graphs

**Assumption checking:**
- Normality: Q-Q plots, Shapiro-Wilk test
- Homogeneity of variance: Levene's test
- Independence: are algorithm runs truly independent?

**Reporting:**
- Report test used, test statistic, p-value, effect size
- Example: "Paired t-test: t(99)=4.23, p<0.001, d=0.61"
- Include confidence intervals and descriptive statistics

Think about: Statistical significance doesn't mean practical importance. Always report effect sizes and confidence intervals, not just p-values.

---

## Prompt 9: Research Question Investigation Framework

Structure your analysis around specific research questions rather than aimless exploration.

**Question 1: Do anchor-based heuristics outperform classical baselines?**
- Hypothesis: Best-anchor beats nearest-neighbor on average
- Analysis:
  - Compare tour qualities across all graphs
  - Statistical test: paired t-test or Wilcoxon
  - Effect size: mean improvement percentage
  - Break down by graph type: where does anchor-based win?
- Conclusion: Accept/reject hypothesis with evidence

**Question 2: What vertex properties predict anchor quality?**
- Hypothesis: MST degree and mean edge weight are top predictors
- Analysis:
  - Univariate correlations for all features
  - Feature importance from trained models
  - Validate: do top features generalize across graph types?
- Conclusion: "MST degree (r=0.62) and mean edge weight (r=-0.51) are strongest predictors..."

**Question 3: Can we predict good anchors without exhaustive search?**
- Hypothesis: ML model produces tours within 10% of best-anchor
- Analysis:
  - Compare predicted-anchor tour quality to best-anchor
  - Success rate: how often does predicted anchor beat random?
  - Computational savings: exhaustive search is O(n), prediction is O(1)
- Conclusion: "Model achieves 92% of best-anchor quality with 50× speedup..."

**Question 4: Do anchor heuristics work on non-metric graphs?**
- Hypothesis: Anchor advantage disappears on non-metric graphs
- Analysis:
  - Compare performance on metric vs. non-metric graphs
  - Test interaction effect: algorithm × metricity
  - Identify threshold: below what metricity score does advantage vanish?
- Conclusion: "Anchor heuristics require metricity >0.6 to outperform baselines..."

**Question 5: How does performance scale with graph size?**
- Hypothesis: All heuristics' optimality gaps increase with size
- Analysis:
  - Plot optimality gap vs. size (for graphs with known optima)
  - Fit scaling curves: linear, logarithmic, power law?
  - Compare scaling rates across algorithms
- Conclusion: "Nearest-neighbor gap grows as O(log n), anchor-based as O(√n)..."

**Question template for each:**
1. State hypothesis clearly
2. Describe analysis method
3. Present results (data, plots, statistics)
4. Interpret findings
5. State conclusion (accept/reject hypothesis, qualifications)

Think about: Frame your paper around these questions. Each question becomes a subsection. Clear research questions make clear narratives.

---

## Prompt 10: Insight Synthesis and Theory Building

Move beyond data description to build explanatory theories for observed patterns.

**Pattern identification:**
- List all major findings from analysis
- Group related findings
- Identify surprising findings that challenge assumptions

**Mechanism explanation:**
- For each pattern, ask WHY
- Example: "Why does high MST degree predict good anchors?"
  - Possible explanation: High MST degree indicates structural centrality
  - Central vertices have many short edges available
  - Pre-committing to short edges constrains later choices productively
- Validate: does explanation hold across graph types?

**Theory construction:**
- Generalize from specific findings to broader principles
- Example theory: "Anchoring effectiveness depends on edge diversity. Vertices with high variance in edge weights make poor anchors because anchor edges might not be representative. Vertices with uniform edge weights (low variance) but overall low weights make best anchors."
- Test theory: does it predict new phenomena? Does it contradict anything?

**Comparative theory:**
- Why does algorithm A beat B on graph type X but lose on type Y?
- Develop theory explaining interaction between algorithm properties and graph properties
- Example: "Greedy algorithms excel on graphs with clear local optima (metric graphs) but struggle on graphs with misleading local cues (random graphs)"

**Boundary conditions:**
- When does your approach work? When does it fail?
- Example: "Anchor heuristics require triangle inequality satisfaction >60%. Below this threshold, pre-committed edges are as likely to be suboptimal as randomly chosen edges."

**Novel insights:**
- What did you discover that wasn't in literature before?
- Unexpected correlations, counterintuitive findings, new algorithmic principles
- These become your paper contributions

**Future predictions:**
- Based on your theories, what predictions can you make?
- "We predict anchor-based approaches will excel on TSP variants with additional constraints (time windows, capacity) because pre-commitment is complementary to constraint satisfaction"

Think about: Data + analysis = findings. Findings + explanation = insights. Insights + generalization = theory. Theory is what gets published.

---

## Prompt 11: Limitations and Threats to Validity Analysis

Rigorously assess limitations, biases, and threats to validity in your research.

**Internal validity threats:**
- Confounding variables: are other factors explaining observed differences?
- Implementation bugs: could bugs in algorithms or feature extraction bias results?
- Measurement error: how accurate are tour weights, feature values?
- Statistical assumptions: did you violate test assumptions?

**External validity threats:**
- Sample bias: does your graph collection represent all TSP instances?
- Overfitting to test set: did you tune too much on validation data?
- Synthetic graphs only: do findings generalize to real-world TSP instances?
- Size limitations: you tested up to 200 vertices - what about 10,000 vertices?

**Construct validity threats:**
- Anchor quality score: is your label definition the right one?
- Feature relevance: did you miss important features?
- Algorithm implementations: are your implementations fair and optimal?

**Statistical conclusion validity:**
- Sample size: do you have enough graphs for reliable estimates?
- Power: could you have missed real effects due to insufficient data?
- Multiple testing: did you adequately correct for multiple comparisons?
- Outliers: do extreme values drive results?

**Reproducibility threats:**
- Random seed sensitivity: do results hold across different random seeds?
- Hyperparameter sensitivity: do conclusions depend on specific hyperparameters?
- Code bugs: can others reproduce your results independently?

**Mitigation strategies:**
- Acknowledge limitations explicitly
- Quantify uncertainty: report confidence intervals, sensitivity analyses
- Validate with held-out data: test on graph types not used in training
- Robustness checks: re-run key analyses with different assumptions
- Transparency: share code and data for independent verification

Think about: Acknowledging limitations strengthens your work. Reviewers will find limitations anyway - better to discuss them yourself and explain why they don't invalidate your conclusions.

---

## Prompt 12: Research Narrative and Contribution Framing

Craft the narrative arc that connects your findings into a coherent research story.

**Story structure:**

**Act 1: Motivation and Gap**
- TSP is important and hard
- Existing heuristics have limitations
- Observation: starting point matters for constructive heuristics
- Gap: no systematic way to choose good starting points
- Your approach: anchor-based heuristics + ML for anchor prediction

**Act 2: Methodology**
- Generated diverse graph collection (X graphs, Y types)
- Benchmarked Z algorithms
- Extracted W features per vertex
- Trained models to predict anchor quality

**Act 3: Findings**
- Finding 1: Anchor-based heuristics beat nearest-neighbor by X% on metric graphs
- Finding 2: MST degree and mean edge weight are top predictors of anchor quality
- Finding 3: ML models predict good anchors, achieving Y% of exhaustive search quality with Z× speedup
- Finding 4: Performance depends on graph metricity (threshold analysis)

**Act 4: Insights**
- Why anchoring works: explanation of mechanism
- When anchoring works: boundary conditions
- Generalization: do findings transfer?

**Act 5: Implications**
- Practical: fast, effective TSP heuristic with learned anchor selection
- Theoretical: insights about graph structure and optimization
- Methodological: lightweight ML for algorithm configuration

**Contribution framing:**

**Algorithmic contribution:**
- Novel family of anchor-based heuristics
- Systematic evaluation on diverse graphs

**ML contribution:**
- Predictive model for anchor quality
- Feature engineering for TSP graph structure
- Demonstration that simple models (linear regression) outperform complex ones

**Empirical contribution:**
- Comprehensive benchmark dataset
- Analysis of algorithm-graph interactions
- Identification of structural features predicting heuristic success

**Theoretical contribution:**
- Explanation of why anchoring works
- Conditions under which anchoring is effective
- Connections between graph properties and optimization difficulty

Think about: Your paper should have a clear "so what?" - why should readers care? Frame contributions in terms of problems solved, questions answered, and knowledge advanced.

---

## Success Criteria

You've succeeded when:
- You can state 3-5 clear, statistically-supported findings
- You've identified and explained the top predictive features for anchor quality
- You've quantified the practical benefit: "Model achieves X% of best-anchor performance with Y× speedup"
- You have publication-quality figures telling the story
- You've written a clear narrative connecting motivation → methods → findings → insights
- Limitations are acknowledged and addressed

## What NOT to Do

- Don't cherry-pick results: report all findings, including negative results
- Don't confuse correlation with causation
- Don't over-interpret: let data constrain claims
- Don't ignore statistics: p-values and effect sizes matter
- Don't skip visualization: plots communicate better than tables
- Don't forget the narrative: data without story is just numbers

## Next Steps After Completion

With analysis complete:
1. Write the paper: Introduction → Related Work → Methods → Results → Discussion → Conclusion
2. Prepare conference presentation: 10-minute talk summarizing key findings
3. Create poster: visual summary of research
4. Submit to venue: TSP workshop, combinatorial optimization conference, or journal
5. Release artifacts: code, data, models on GitHub for reproducibility
6. Plan follow-up research: extensions, applications, new questions

Analysis is where data becomes science. Do it rigorously, visualize it clearly, explain it compellingly.
