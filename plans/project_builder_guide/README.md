# TSP Research Pipeline - Metaprompt Collection

## Overview

This directory contains a comprehensive series of metaprompts for designing and implementing a research pipeline for Traveling Salesman Problem (TSP) heuristic algorithm research. These prompts are specifically crafted to guide implementation WITHOUT providing code - only directions, ideas, and architectural guidance.

## Purpose

The goal is to investigate **anchor-based heuristics** for TSP: algorithms that pre-commit to two low-cost edges from a starting vertex (the "anchor") before greedily constructing the rest of the tour. The research pipeline will:

1. Generate diverse graph instances with controlled properties
2. Benchmark anchor-based heuristics against classical algorithms
3. Extract structural features from graphs and vertices
4. Train ML models to predict which vertices make good anchors
5. Analyze results to understand when and why anchoring works

## Metaprompt Structure

Each metaprompt file is designed to be **copy-pasted to another LLM** (like Grok, GPT, etc.) for implementation guidance. They follow a consistent structure:

- **Context Block**: Explains the problem and architectural approach
- **Multiple Numbered Prompts**: Each prompt is a standalone, actionable directive
- **Success Criteria**: How to know when the implementation is complete
- **What NOT to Do**: Common pitfalls to avoid
- **Next Steps**: What comes after completing this component

## Metaprompt Files

### [01_graph_generation_system.md](01_graph_generation_system.md)
**Focus**: Building the foundation - generating diverse TSP graph instances

Covers:
- Core graph data structures
- Euclidean, metric, and random graph generation
- Property verification (metricity, symmetry)
- Batch generation pipelines
- Storage and retrieval systems
- Visualization utilities
- Comprehensive testing

**Start here** - this is your data foundation. Everything else depends on having quality, diverse graph instances.

---

### [02_algorithm_benchmarking_pipeline.md](02_algorithm_benchmarking_pipeline.md)
**Focus**: Systematic algorithm testing and comparison

Covers:
- Unified algorithm interfaces
- Baseline implementations (nearest neighbor, greedy, Christofides)
- Anchor-based heuristic implementations
- Tour validation and quality metrics
- Batch benchmarking systems
- Results storage and retrieval
- Statistical analysis tools
- Visualization and reporting
- Optimal solution computation for small graphs

**Dependencies**: Requires graph generation system (01)

---

### [03_feature_engineering_system.md](03_feature_engineering_system.md)
**Focus**: The ML bridge - extracting predictive features from graphs

Covers:
- Feature extraction architecture
- Weight-based vertex features
- Topological features (centrality measures)
- MST-based features
- Neighborhood and regional features
- Heuristic-specific features
- Graph-level context features
- Feature validation and analysis
- Anchor quality labeling strategies
- Feature engineering pipelines
- Feature selection utilities
- Transformations and derived features

**Dependencies**: Requires graph generation (01) and benchmarking (02)

---

### [04_machine_learning_component.md](04_machine_learning_component.md)
**Focus**: Training models to predict anchor quality

Covers:
- ML problem formulation (regression, classification, ranking)
- Train-test split strategies
- Linear regression baseline (primary model for interpretability)
- Tree-based models (comparison baselines)
- Model evaluation and comparison
- Cross-validation strategies
- Hyperparameter tuning
- Feature engineering for ML
- Model interpretation (SHAP, coefficients, importance)
- Prediction-to-algorithm pipeline
- Generalization testing
- Online learning and model updates

**Dependencies**: Requires feature engineering (03)

---

### [05_pipeline_integration_workflow.md](05_pipeline_integration_workflow.md)
**Focus**: Connecting all components into a reproducible research system

Covers:
- Pipeline architecture design
- Configuration management
- Experiment tracking and metadata
- Reproducibility infrastructure
- Automated testing and validation
- Performance monitoring and profiling
- Parallel execution and scaling
- Error handling and fault tolerance
- Results analysis and reporting
- Interactive exploration tools
- Documentation systems
- Version control and collaboration

**Dependencies**: All previous components (01-04)

---

### [06_analysis_visualization_insights.md](06_analysis_visualization_insights.md)
**Focus**: Extracting scientific insights and preparing for publication

Covers:
- Exploratory data analysis framework
- Algorithm performance comparison analysis
- Graph property and performance relationship analysis
- Feature importance and anchor quality analysis
- Model performance and generalization analysis
- Case study deep dives
- Publication-quality visualization suite
- Statistical rigor and hypothesis testing
- Research question investigation framework
- Insight synthesis and theory building
- Limitations and threats to validity
- Research narrative and contribution framing

**Dependencies**: Requires complete pipeline with experimental results (01-05)

---

## How to Use These Metaprompts

### For Sequential Implementation

1. **Read the full collection first** to understand the big picture
2. **Start with 01** (graph generation) and work sequentially
3. **For each file**:
   - Copy the entire file content
   - Paste into your LLM of choice (Grok, GPT, Claude, etc.)
   - Ask: "Implement this component following these guidelines"
   - The LLM will provide directions and architectural guidance
4. **Test each component** thoroughly before moving to the next
5. **Iterate**: You may need to revisit earlier components as you learn

### For Parallel Development

If you have multiple developers or want to work on multiple fronts:

- **Team A**: Work on 01 (graph generation) and 02 (benchmarking)
- **Team B**: Work on 03 (features) and 04 (ML) - can start once 01-02 produce sample data
- **Team C**: Work on 05 (integration) - design infrastructure while others build components
- **Everyone**: Contribute to 06 (analysis) once data is flowing

### For Exploratory Use

If you want to understand the approach before building:

1. Read 01, 02, 03 for the technical pipeline
2. Read 06 for the research questions and analytical approach
3. Read 05 for the integration philosophy
4. Decide which parts are most relevant to your specific needs

## Key Principles

### 1. NO CODE in Metaprompts
These metaprompts provide **directions and ideas only**. They explain WHAT to build and WHY, but not the exact implementation. This gives you flexibility to choose:
- Programming language (Python recommended, but not required)
- Libraries and frameworks
- Implementation details

### 2. Modularity
Each component is independent. You can:
- Swap implementations (different graph generator, different algorithms)
- Skip components (maybe you don't need ML, just benchmarking)
- Extend components (add new features, new algorithms)

### 3. Reproducibility First
Every metaprompt emphasizes reproducibility:
- Random seeds for all stochastic processes
- Configuration files for all experiments
- Comprehensive logging
- Version control

### 4. Research Quality
This isn't a quick hack - it's a foundation for publishable research:
- Statistical rigor
- Comprehensive testing
- Clear documentation
- Publication-quality outputs

### 5. Interpretability Over Complexity
The pipeline favors:
- Simple models (linear regression) over complex ones (deep learning)
- Explicit features over learned representations
- Clear explanations over black boxes

## Research Philosophy

This pipeline embodies a specific research philosophy:

**Structure-first thinking**: Rather than learning patterns from millions of solved instances (neural approaches), analyze inherent graph properties to guide algorithm behavior. This is meta-learning about the optimization landscape.

**Lightweight ML**: Use classical ML (linear regression, random forests) on carefully engineered features rather than end-to-end deep learning. This maintains interpretability and requires less data.

**Systematic experimentation**: Generate controlled synthetic data to isolate variables, then test on real-world instances for validation.

**Theory-driven feature engineering**: Design features based on graph theory and TSP intuition, not blind feature generation.

## Expected Timeline

Rough estimates for full implementation:

- **Week 1**: Graph generation system (01) - foundation takes time
- **Week 2**: Algorithm benchmarking (02) - standardize existing code
- **Week 3**: Feature engineering (03) - creative but time-consuming
- **Week 4**: ML component (04) - training is fast, evaluation is thoughtful
- **Ongoing**: Integration (05) - build incrementally as components complete
- **Ongoing**: Analysis (06) - continuous as data accumulates

Total: **4-6 weeks** for a solid v1 implementation, then iterative improvement.

## Success Metrics

You've succeeded when:

✅ You can generate 100 diverse graphs in under a minute (01)
✅ You can benchmark 5 algorithms on 100 graphs in under an hour (02)
✅ You've identified 5-10 features that clearly correlate with anchor quality (03)
✅ Your ML model predicts good anchors, achieving >90% of best-anchor quality (04)
✅ You can run a complete experiment from scratch with a single command (05)
✅ You have publication-quality figures and clear, statistically-supported findings (06)

## Common Pitfalls

❌ **Skipping graph verification**: You'll waste weeks debugging with corrupted graphs
❌ **Premature optimization**: Build it clean first, optimize later
❌ **Ignoring reproducibility**: Future you will curse present you
❌ **Analysis without statistics**: Eyeballing results leads to false conclusions
❌ **Over-engineering**: Simple beats complex for research code

## Extensions and Future Work

Once the core pipeline is complete, consider:

1. **Real-world instances**: Test on TSPLIB benchmark problems
2. **Sparse graphs**: Extend to incomplete graphs
3. **Asymmetric TSP**: Add directional cost differences
4. **Dynamic TSP**: Handle changing edge weights
5. **Multi-objective TSP**: Optimize multiple criteria simultaneously
6. **Other combinatorial problems**: Apply the approach to VRP, SAT, scheduling

## Resources and Context

These metaprompts are based on:
- Classical TSP literature (Christofides, nearest neighbor, greedy heuristics)
- Modern ML for combinatorial optimization
- Algorithm portfolio selection research
- Graph theory and structural graph properties
- Statistical experimental design

They're designed for:
- Researchers investigating TSP heuristics
- Students learning about algorithm design and ML
- Practitioners needing interpretable TSP solvers
- Anyone interested in data-driven algorithm configuration

## Contributing and Iteration

These metaprompts are living documents. As you implement:

- **Document what works**: Implementation choices that paid off
- **Document what doesn't**: Dead ends and pitfalls you encountered
- **Share insights**: Novel findings or approaches
- **Suggest improvements**: Ways to make metaprompts clearer

The goal is continuous improvement of both the pipeline and the metaprompts themselves.

## Final Notes

Remember: **You're not just building a TSP solver. You're building a research platform for understanding when and why heuristics work.**

The graph generation system enables controlled experiments. The benchmarking system reveals performance patterns. The feature engineering bridges algorithms to ML. The ML component automates insight. The integration makes it reproducible. The analysis makes it science.

Build thoughtfully. Test thoroughly. Document clearly. Analyze rigorously.

Good luck, and may your tours be short and your insights be deep!

---

## Quick Start

**Absolute beginner?** Start here:

1. Read this README fully
2. Skim all six metaprompt files to get the big picture
3. Start with [01_graph_generation_system.md](01_graph_generation_system.md)
4. Copy Prompt 1 from file 01, paste into an LLM, ask for implementation guidance
5. Implement, test, repeat for subsequent prompts
6. Move to file 02 when file 01 is solid

**Already have some components?**
- Graph generation working? Skip to 02
- Algorithms working? Skip to 03
- Features extracted? Skip to 04
- Models trained? Skip to 05 and 06

**Just want to understand the approach?**
- Read 01 for data generation philosophy
- Read 02 for algorithm comparison methodology
- Read 03 for feature engineering approach
- Read 04 for ML strategy
- Read 06 for research insights framework

**Ready to publish?**
- Focus on 06 for analysis and narrative framing
- Ensure 05 is solid for reproducibility
- Prepare artifacts: code repository, dataset, trained models

---

*These metaprompts are designed to be explored, adapted, and extended. Use them as a guide, not a rigid prescription. The best research happens when you engage critically with the methods and make them your own.*
