# Changelog

All notable changes to the TSP Anchor-Based Heuristic Research Platform.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Phase 2 Steps 5-8: Benchmarking runners, batch processing, statistical analysis, visualization
- Phase 3: Feature engineering system
- Phase 4: Machine learning component
- Phase 5: Pipeline integration
- Phase 6: Analysis and insights

---

## [2.0.0] - 2025-11-05

### Added - Phase 2 Steps 1-4 Complete
- **Algorithm Interface & Registry System**
  - `src/algorithms/base.py`: Core algorithm interfaces (TourResult, AlgorithmMetadata, TSPAlgorithm)
  - `src/algorithms/registry.py`: Centralized algorithm registration with @register_algorithm decorator
  - Support for filtering algorithms by tags, graph type, and graph size
  - Random seed support for reproducible algorithm execution

- **Tour Validation & Quality Metrics**
  - `src/algorithms/validation.py`: Comprehensive tour validation (structure, edges, constraints)
  - `src/algorithms/metrics.py`: Quality metrics (weight, statistics, optimality gap, approximation ratio)
  - TourValidator class with optional caching for batch operations
  - MetricsCalculator class with caching layer for expensive computations

- **Baseline TSP Algorithms** (3 algorithms, 5 variants)
  - `src/algorithms/nearest_neighbor.py`: Nearest Neighbor (random start, best-of-all starts)
  - `src/algorithms/greedy.py`: Greedy edge-picking with Union-Find cycle detection
  - `src/algorithms/exact.py`: Held-Karp dynamic programming exact solver (n ≤ 20)

- **Anchor-Based Heuristics** (3 algorithms, 5 variants)
  - `src/algorithms/single_anchor.py`: Single anchor with two pre-committed edges
  - `src/algorithms/best_anchor.py`: Exhaustive anchor search across all vertices
  - `src/algorithms/multi_anchor.py`: Multi-anchor variants (random, distributed selection)

- **Test Suite** (89 new tests)
  - `src/tests/test_algorithms.py`: Core interface tests (59 tests)
  - `src/tests/test_baseline_algorithms.py`: Baseline algorithm tests (16 tests)
  - `src/tests/test_anchor_algorithms.py`: Anchor algorithm tests (14 tests)

- **Documentation**
  - `PHASE2_COMPLETE.md`: Detailed Phase 2 implementation summary
  - `src/algorithms/CLAUDE.md`: Algorithm package documentation
  - `src/tests/CLAUDE.md`: Test suite documentation
  - `docs/CHANGELOG.md`: This unified changelog

### Changed
- Updated root `CLAUDE.md`:
  - Project status: Phase 2 Steps 1-4 Complete (50% of Phase 2)
  - Last updated date: 11-05-2025
  - Test suite status: 123 tests total (34 Phase 1 + 89 Phase 2)
  - Document version: 3.0
  - Current priorities: Complete Phase 2 Steps 5-8

### Validated
- All 123 tests passing (100% pass rate)
- All 8 algorithms registered and functional
- Edge case testing (3-vertex graphs, asymmetric graphs, various sizes)
- Code quality review: No bugs found, production-ready
- Performance testing: All algorithms meet expected complexity bounds

### Technical Details
- **Lines of Code**: ~2,200 algorithm code + ~1,350 test code
- **Algorithms**: 8 total (3 baseline variants, 5 anchor variants)
- **Test Coverage**: 89 tests covering interfaces, validation, metrics, and algorithms
- **Performance**: All algorithms tested on graphs up to n=100

---

## [1.0.0] - 2025-10-30

### Added - Phase 1 Complete
- **Graph Generation System**
  - `src/graph_generation/euclidean_generator.py`: Euclidean graphs from 2D coordinates
  - `src/graph_generation/metric_generator.py`: Metric graphs (MST and completion strategies)
  - `src/graph_generation/quasi_metric_generator.py`: Quasi-metric asymmetric graphs
  - `src/graph_generation/random_generator.py`: Random graphs with various distributions
  - `src/graph_generation/graph_utils.py`: Verification, storage, visualization utilities
  - `src/graph_generation/batch_generator.py`: YAML-based batch generation

- **Test Suite** (34 tests)
  - `src/tests/test_graph_generators.py`: Comprehensive graph generation tests
  - 10 Euclidean tests
  - 6 Metric tests
  - 3 Quasi-metric tests
  - 7 Random tests
  - 3 Edge case tests
  - 2 Consistency tests
  - 3 Performance benchmarks

- **Documentation**
  - `src/graph_generation/CLAUDE.md`: Graph generation package documentation
  - `docs/10-29-2025_change.md`: Phase 1 bug fixes
  - `docs/10-30-2025_change.md`: Phase 1 improvements

### Changed
- Initial project structure established
- Root `CLAUDE.md` created with multi-phase architecture
- Directory structure organized for 6 phases

### Technical Principles Established
- **Principle 1**: Euclidean property preservation (scale coordinates, not weights)
- **Principle 2**: Quasi-metric directional triangle inequality (forward only)
- **Principle 3**: MST vs Completion strategies for metric graphs

---

## [0.1.0] - 2025-10-25

### Added - Project Initialization
- Initial repository structure
- `CLAUDE.md`: Project vision and architectural principles
- `/guides/`: 6 metaprompt specifications for each phase
  - `01_graph_generation_system.md`
  - `02_algorithm_benchmarking_pipeline.md`
  - `03_feature_engineering_system.md`
  - `04_machine_learning_component.md`
  - `05_pipeline_integration_workflow.md`
  - `06_analysis_visualization_insights.md`
- `README.md`: Project overview
- `LICENSE.md`: Project license
- `requirements.txt`: Python dependencies
- `.gitignore`: Git exclusions

### Established
- Multi-agent development workflow (Foreman, Planner, Builder, Validator, Debugger)
- 7 architectural principles
- 6-phase research pipeline design
- Testing-first development philosophy

---

## Version History Summary

| Version | Date | Phase | Status | Tests | Lines of Code |
|---------|------|-------|--------|-------|---------------|
| 2.0.0 | 2025-11-05 | Phase 2 (Steps 1-4) | Complete | 123 | ~3,550 |
| 1.0.0 | 2025-10-30 | Phase 1 | Complete | 34 | ~1,200 |
| 0.1.0 | 2025-10-25 | Initialization | Complete | 0 | 0 |

---

## Notes

### Changelog Maintenance
- This changelog is maintained in `/docs/CHANGELOG.md`
- Format: Major.Minor.Patch semantic versioning
- Major: Phase completions
- Minor: Significant feature additions within phase
- Patch: Bug fixes and minor improvements

### Update Frequency
- Updated at major milestones (phase completions, significant features)
- Validated changes marked with validation date
- Each entry includes: Added, Changed, Fixed, Deprecated, Removed, Security (as applicable)

### Related Documents
- `/CLAUDE.md`: Overall project context and status
- `/PHASE2_COMPLETE.md`: Detailed Phase 2 implementation notes
- `/docs/*.md`: Specific change documentation
- `/guides/*.md`: Implementation specifications

---

**Changelog Version**: 1.0
**Last Updated**: 2025-11-05
**Maintained By**: All agents (consolidated by Foreman)
