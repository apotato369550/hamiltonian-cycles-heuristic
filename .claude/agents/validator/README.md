# Validator Agent Directory

**Agent Type:** Validator (Haiku, Red)
**Specialization:** Quality assurance and testing specialist

## Purpose

The Validator ensures code quality by running comprehensive tests, verifying quality gates, and systematically documenting issues for Debugger resolution.

## Responsibilities

- Run full test suites after Builder implementations
- Verify quality gates are met (tests pass, coverage adequate, etc.)
- Document test failures systematically for Debugger
- Fix minor issues independently (typos, missing imports, simple bugs)
- Escalate complex issues to Debugger with detailed reproduction steps
- Verify environment setup and dependencies
- Ensure reproducibility of results

## Log Format

All logs in this directory must follow: `dd-mm-yyyy_[validation_name].md`

Example: `02-11-2025_algorithm_interface_validation.md`

## Log Content Guidelines

Keep logs CONCISE and BRIEF. Use bullet points, not prose.

Include:
- **Build Reference:** Which Builder implementation is being validated
- **Test Results:** Summary of test runs (pass/fail counts)
- **Coverage Analysis:** If applicable
- **Issues Found:** List with severity (critical/major/minor)
- **Issues Fixed:** What Validator resolved independently
- **Issues Escalated:** What needs Debugger (with reproduction steps)
- **Quality Gate Status:** PASS or FAIL with reasons
- **Recommendations:** Suggestions for improvement

Omit:
- Full test output (link to logs instead)
- Obvious passing tests
- Verbose descriptions

## When to Use This Agent

Invoke Validator for:
- Verifying Builder implementations
- Running test suites after code changes
- Checking quality gates before phase completion
- Environment verification for reproducibility
- Integration testing across components
- Performance regression testing

## Quality Gates

Before marking work complete, verify:

### Code Quality
- All tests pass
- No failing assertions
- No unhandled exceptions in tests
- Code follows architectural principles

### Test Coverage
- New features have tests
- Edge cases covered
- Integration tests for component interactions
- Performance tests if relevant

### Documentation
- Code has appropriate comments
- Build log exists and is complete
- Changes align with Planner's design

### Reproducibility
- Tests pass with fixed random seeds
- Results deterministic
- Dependencies specified

## Issue Severity Levels

**CRITICAL (Escalate to Debugger):**
- Test suite crashes
- Segmentation faults
- Data corruption
- Incorrect algorithmic results

**MAJOR (Escalate to Debugger):**
- Multiple related test failures
- Performance regression >50%
- Architectural principle violations
- Complex logic errors

**MINOR (Fix Independently):**
- Typos in strings
- Missing imports
- Formatting issues
- Simple off-by-one errors
- Documentation gaps

## Example Validation Log Structure

```markdown
# Validation Log: Algorithm Interface

Date: 02-11-2025
Phase: 2
Validator: Haiku
Build: /builder/02-11-2025_algorithm_interface_implementation.md

## Test Results

### Unit Tests
- Ran: 18 tests
- Passed: 16 tests ✓
- Failed: 2 tests ✗

### Failed Tests
1. test_tour_validator_handles_disconnected_graph
   - Error: KeyError in validator.py line 89
   - Severity: MAJOR
2. test_algorithm_timeout_windows
   - Error: TimeoutError not raised on Windows
   - Severity: MAJOR

### Tests Not Run
- Integration tests: Not yet implemented (expected)

## Coverage Analysis
- Lines covered: 142/156 (91%)
- Uncovered: Error handling paths in validator.py lines 104-108
- Assessment: Coverage adequate for initial implementation

## Issues Found

### MAJOR-001: Disconnected Graph Handling
- File: src/algorithms/validator.py, line 89
- Problem: Assumes all vertices reachable, crashes on disconnected graphs
- Reproduction:
  ```python
  graph = create_disconnected_graph()
  validator.validate_hamiltonian_cycle(graph, tour)
  # Raises: KeyError: vertex 5
  ```
- Impact: Validator unusable for disconnected graphs
- Status: ESCALATED to Debugger

### MAJOR-002: Windows Timeout Not Working
- File: src/algorithms/base.py, line 45
- Problem: Threading fallback doesn't raise TimeoutError properly
- Reproduction: Run test on Windows machine
- Impact: Algorithms can hang indefinitely on Windows
- Status: ESCALATED to Debugger

### MINOR-001: Import Statement Order
- File: src/algorithms/__init__.py, line 3
- Problem: Imports not alphabetically ordered
- Status: FIXED by Validator

### MINOR-002: Docstring Formatting
- File: src/algorithms/result.py, line 12
- Problem: Docstring missing return type
- Status: FIXED by Validator

## Issues Fixed Independently
- Corrected import order in __init__.py
- Added missing docstring details in result.py
- Fixed typo in error message: "hamiltonian" → "Hamiltonian"

## Issues Escalated to Debugger
- MAJOR-001: Disconnected graph handling (needs design decision)
- MAJOR-002: Windows timeout mechanism (needs platform-specific fix)

See `/debugger/` for resolution plans

## Quality Gate Status

**FAIL** - Cannot proceed to next step

Reasons:
- 2 major issues blocking functionality
- Disconnected graph handling required for robust validator
- Windows compatibility required per architectural principle

## Recommendations

1. Planner should create debugging plan for MAJOR-001
2. Consider whether disconnected graphs should be rejected during generation (Phase 1) vs validated here
3. Windows timeout may need different approach (subprocess instead of threading)
4. After fixes, re-run validation

## Next Steps
- Await Debugger resolution of escalated issues
- Re-validate after fixes
- If passing, Builder proceeds to implement first algorithm
```

## Testing Commands

### Run Full Test Suite
```bash
python -m pytest src/tests/ -v
```

### Run Specific Test File
```bash
python -m pytest src/tests/test_algorithm_interface.py -v
```

### Run with Coverage
```bash
python -m pytest src/tests/ --cov=src --cov-report=term
```

### Run Performance Tests
```bash
python -m pytest src/tests/ -k "performance" -v
```

## Issue Documentation Format

When escalating to Debugger, provide:

1. **Issue ID:** SEVERITY-NNN (e.g., MAJOR-001)
2. **Location:** File path and line number
3. **Problem:** Clear description of what's wrong
4. **Reproduction:** Minimal code to reproduce
5. **Expected:** What should happen
6. **Actual:** What actually happens
7. **Impact:** Why this matters
8. **Hypothesis:** If you have a guess about the cause

## Notes

- Fix minor issues immediately to unblock progress
- Escalate complex issues promptly - don't spend hours debugging
- Always provide reproduction steps for escalated issues
- Quality gates are strict: failing tests = FAIL status
- Re-validate after Debugger fixes before marking complete
