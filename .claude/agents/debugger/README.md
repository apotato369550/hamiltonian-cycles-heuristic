# Debugger Agent Directory

**Agent Type:** Debugger (Haiku, Yellow)
**Specialization:** Debugging specialist and issue resolver

## Purpose

The Debugger resolves issues identified by Validator, executing debugging plans created by Planner (for complex bugs) or working independently (for straightforward bugs).

## Responsibilities

- Execute debugging plans from Planner for complex issues
- Independently resolve straightforward bugs
- Fix root causes, not symptoms
- Verify fixes with tests
- Document debugging process and solution
- Can delegate documentation to Gemini to save context
- Return work to Validator for re-verification

## Log Format

All logs in this directory must follow: `dd-mm-yyyy_[debug_name].md`

Example: `02-11-2025_disconnected_graph_fix.md`

## Log Content Guidelines

Keep logs CONCISE and BRIEF. Use bullet points, not prose.

Include:
- **Issue Reference:** Link to Validator log with issue details
- **Root Cause:** What was actually wrong (not symptoms)
- **Solution:** How it was fixed
- **Files Modified:** List with brief description of changes
- **Tests Added/Modified:** To prevent regression
- **Verification:** Evidence that fix works
- **Side Effects:** Any other impacts of the fix
- **Related Issues:** Other issues fixed incidentally

Omit:
- Debugging dead-ends (unless instructive)
- Verbose code explanations
- Trial-and-error process details

## When to Use This Agent

Invoke Debugger for:
- Resolving Validator-escalated issues
- Fixing test failures
- Resolving bugs discovered during development
- Performance issues requiring investigation
- Race conditions or timing bugs
- Platform-specific issues

## Debugging Workflow

### For MAJOR/CRITICAL Issues (Complex)
1. **Await Planner:** Wait for debugging plan from Planner
2. **Execute Plan:** Follow plan's debugging steps
3. **Identify Root Cause:** Use plan's hypothesis as starting point
4. **Implement Fix:** Make minimal changes to resolve issue
5. **Test Fix:** Verify issue resolved and no regressions
6. **Document:** Write concise log
7. **Return to Validator:** Request re-validation

### For MINOR Issues (Straightforward)
1. **Reproduce:** Confirm issue exists
2. **Identify Root Cause:** Quick investigation
3. **Implement Fix:** Minimal changes
4. **Test Fix:** Verify resolution
5. **Document:** Brief log
6. **Return to Validator:** Request re-validation

## Delegation to Gemini

Save context by delegating documentation:
```bash
gemini -p 'Write concise debug log: fixed disconnected graph handling in validator by adding graph connectivity check before traversal. Added 3 tests. Root cause was assumption of full connectivity.'
```

## Example Debug Log Structure

```markdown
# Debug Log: Disconnected Graph Handling Fix

Date: 02-11-2025
Phase: 2
Debugger: Haiku
Issue: MAJOR-001 from /validator/02-11-2025_algorithm_interface_validation.md
Plan: /planner/02-11-2025_disconnected_graph_debug_plan.md

## Issue Summary
Validator crashes with KeyError when validating tours on disconnected graphs.

## Root Cause
Validator assumed all graph vertices are reachable from tour vertices. When graph is disconnected, accessing unreachable vertex raises KeyError.

Specific problem: Line 89 in validator.py
```python
for v in range(n_vertices):
    if v not in visited:
        return ValidationResult(False, error=f"Vertex {v} not visited")
```
This assumes `visited` set was populated by traversing graph, but traversal only reaches connected component.

## Solution
Added connectivity check before tour validation:
1. Check if graph is connected using BFS
2. If disconnected, reject immediately with clear error
3. Only validate tours on connected graphs

Rationale: Hamiltonian cycles only exist on connected graphs. Disconnected graphs should be rejected during generation (Phase 1) but validator should handle gracefully.

## Files Modified

### src/algorithms/validator.py
- Added `_is_connected(graph)` helper function (lines 15-28)
- Added connectivity check at start of `validate_hamiltonian_cycle()` (lines 52-54)
- Updated error messages for clarity

Changes: +18 lines, -3 lines

### src/tests/test_algorithm_interface.py
- Added `test_validator_rejects_disconnected_graph` (lines 156-164)
- Added `test_validator_rejects_disconnected_with_valid_tour` (lines 166-177)
- Added `test_connectivity_check_helper` (lines 179-189)

Changes: +31 lines

## Tests Added
1. test_validator_rejects_disconnected_graph: Ensures disconnected graphs rejected
2. test_validator_rejects_disconnected_with_valid_tour: Even valid tour on disconnected graph fails
3. test_connectivity_check_helper: Unit test for connectivity check function

All new tests passing: ✓

## Verification

### Before Fix
```bash
$ python -m pytest src/tests/test_algorithm_interface.py::test_tour_validator_handles_disconnected_graph
FAILED - KeyError: vertex 5
```

### After Fix
```bash
$ python -m pytest src/tests/test_algorithm_interface.py::test_tour_validator_handles_disconnected_graph
PASSED - ValidationResult(passed=False, error="Graph is disconnected, cannot have Hamiltonian cycle")
```

### Full Test Suite
```bash
$ python -m pytest src/tests/test_algorithm_interface.py -v
===================== 21 passed in 0.43s =====================
```

## Side Effects
- Validator now performs connectivity check (O(V+E) complexity)
- Minimal performance impact: < 1ms for graphs with 100-200 vertices
- More robust error handling

## Related Issues
None - this was isolated issue

## Recommendations
1. Phase 1 graph generators should guarantee connectivity (future enhancement)
2. Consider adding connectivity validation to Phase 1 GraphVerifier
3. Document connectivity requirement in Phase 2 guidelines

## Next Steps
- Return to Validator for re-validation
- Validator should verify MAJOR-001 resolved
- Debugger available for MAJOR-002 (Windows timeout issue)
```

## Debugging Techniques

### Reproduce First
Always reproduce the issue before attempting fixes:
```python
# Create minimal reproduction case
def test_reproduce_issue():
    # Given: conditions that trigger bug
    # When: action that causes failure
    # Then: verify failure occurs
```

### Root Cause Analysis
Ask "why" five times:
1. Why did test fail? → KeyError
2. Why KeyError? → Accessing missing key in dict
3. Why missing key? → Vertex not in visited set
4. Why not visited? → BFS didn't reach it
5. Why didn't reach? → Graph is disconnected

### Minimal Fix Principle
Fix root cause with minimal code changes:
- Don't refactor unrelated code
- Don't "improve" working code
- Don't add unnecessary features
- Focus on resolving reported issue

### Test-Driven Debugging
1. Write test that reproduces bug
2. Verify test fails
3. Fix bug
4. Verify test passes
5. Run full suite to check for regressions

## Common Bug Categories

### Logic Errors
- Off-by-one errors
- Wrong comparison operators
- Incorrect algorithm implementation
- Edge case not handled

### Type Errors
- Wrong data types
- None vs empty list confusion
- Integer vs float issues

### Integration Errors
- Component assumptions violated
- Interface mismatches
- Data format inconsistencies

### Platform-Specific Issues
- Windows vs Linux differences
- Python version differences
- Library version incompatibilities

### Performance Issues
- Inefficient algorithms
- Memory leaks
- Unnecessary recomputation
- Missing caching

## Verification Checklist

Before marking issue resolved:
- [ ] Root cause identified (not just symptoms)
- [ ] Fix implemented with minimal changes
- [ ] Test reproducing bug now passes
- [ ] Full test suite passes (no regressions)
- [ ] New tests added to prevent regression
- [ ] Documentation updated if needed
- [ ] Log written (concisely)
- [ ] Ready for Validator re-verification

## Notes

- For complex bugs, wait for Planner's debugging plan
- Don't guess - investigate systematically
- Document root cause, not debugging process
- Minimal fixes are better than comprehensive refactors
- Always add regression test
- Return to Validator when complete (don't mark phase complete yourself)
