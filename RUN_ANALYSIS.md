# How to Run the Anchor Statistics Analysis

## Quick Start

From the project root:

```bash
python3 scripts/run_full_analysis.py
```

This runs all 8 phases sequentially.

---

## Important Notes

1. **Use python3** not python (python2 may be default)

2. **From project root** - The scripts use relative imports:
   ```bash
   cd /home/jay/Desktop/Coding\ Stuff/hamiltonian-cycles-heuristic
   python3 scripts/run_full_analysis.py
   ```

3. **Check your venv is active** (optional but recommended):
   ```bash
   source venv/bin/activate
   python3 scripts/run_full_analysis.py
   ```

---

## What Happens

The master script will:
1. ✅ Generate 100 test graphs
2. ✅ Compute anchor quality for each vertex
3. ✅ Extract edge statistics
4. ✅ Run correlation analysis
5. ✅ Train regression models
6. ✅ Analyze decision tree
7. ✅ Test hypotheses
8. ✅ Validate predictions

All results save to `results/anchor_analysis/`

---

## Individual Phases (if needed)

```bash
# Run specific phase
python3 scripts/01_generate_test_graphs.py
python3 scripts/02_compute_anchor_quality.py
python3 scripts/03_extract_edge_statistics.py
python3 scripts/04_correlation_analysis.py
python3 scripts/05_simple_regression.py
python3 scripts/06_decision_tree_analysis.py
python3 scripts/07_hypothesis_validation.py
python3 scripts/08_practical_validation.py
```

---

## Expected Runtime

- Total: 15-20 minutes for full pipeline
- Phase 2 (anchor quality): ~8 minutes (most expensive)
- Other phases: ~1-2 minutes each

---

## Troubleshooting

**ImportError: cannot import X from src.Y**
- Make sure you're running from project root
- Check venv is activated if using one

**FileNotFoundError: data/anchor_analysis/**
- Directories are created automatically
- If not, run from project root

**Out of memory**
- Phase 2 tests many graph combinations
- If you run out of memory, reduce vertices or graph count in Phase 1 config

---

## After Running

Check these files:
- `results/anchor_analysis/correlations.csv` - Feature correlations
- `results/anchor_analysis/hypothesis_test_results.csv` - Test results
- `results/anchor_analysis/practical_validation_results.csv` - Validation results

Look at these plots:
- `results/anchor_analysis/correlations_plot.png`
- `results/anchor_analysis/hypothesis_validation.png`
- `results/anchor_analysis/model_comparison.png`

---

**Ready? Run:** `python3 scripts/run_full_analysis.py`
