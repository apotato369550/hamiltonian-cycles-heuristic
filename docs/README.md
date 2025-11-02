# Documentation Index

This folder contains change logs and technical documentation for the Hamiltonian Cycles Heuristic project.

## Files

### [10-29-2025_change.md](10-29-2025_change.md)
**Comprehensive bug fix documentation from October 29, 2025**

Contains detailed analysis of 5 bugs fixed:
- Bug #1: Euclidean graphs failing distance verification
- Bug #2: Quasi-metric graphs violating triangle inequality
- Bug #3: Narrow weight range test failing
- Bug #4: Weight scaling test expectations
- Bug #5: Quasi-metric test expecting perfect metricity

**Key Results:** All 34 tests now passing ✅

**Read this for:**
- Understanding the bugs that were fixed
- Technical details of solutions implemented
- Known issues for future work
- Summary of test results

---

## Key Documentation Files (Root Directory)

### [../CLAUDE.md](../CLAUDE.md)
**Critical directives for all Claude instances working on this project**

Contains:
- Critical technical principles (3 must-know principles)
- Known issues and workarounds
- Test suite status
- Common tasks and approaches
- Environment setup guide
- Performance considerations
- Debugging guide
- Questions to ask yourself

**Read this before working on:**
- Any bug fixes
- Test modifications
- Generator implementations
- Verifier improvements

---

## Quick Reference

### Running Tests
```bash
python src/tests/test_graph_generators.py
```
Expected: All 34 tests pass in ~0.1 seconds

### Running Demo
```bash
python src/main.py
```
Shows 5 comprehensive demonstrations of all generators

### Checking Specific Property
```python
from graph_generation.verification import GraphVerifier

verifier = GraphVerifier(fast_mode=False)
result = verifier.verify_metricity(matrix)
print(f"Metricity Score: {result.details['metricity_score']}")
```

---

## Critical Principles (Remember These!)

1. **Euclidean Property:** Scale COORDINATES, not weights
2. **Quasi-Metrics:** Only check forward-path constraints, not all permutations
3. **MST vs Completion:** Choose strategy based on desired weight distribution

See CLAUDE.md for detailed explanations.

---

## Last Updated
October 29, 2025

**Status:** All tests passing ✅ | Code stable | Documentation complete

---

For more information, see:
- CLAUDE.md - Directives and principles
- 10-29-2025_change.md - Detailed bug analysis
- README files in src/ and config/ folders
