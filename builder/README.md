# Builder Agent Directory

**Agent Type:** Builder (Haiku, Green)
**Specialization:** Implementation specialist and code craftsperson

## Purpose

The Builder executes implementation plans created by Planner, writing pragmatic, high-quality functional code that follows architectural principles and coding standards.

## Responsibilities

- Execute Planner's implementation plans step-by-step
- Write clean, tested, production-quality code
- Apply architectural principles and coding best practices
- Handle implementation details and edge cases
- Create comprehensive tests for new code
- Document code with clear comments (WHY not WHAT)
- Can delegate context-heavy documentation to Gemini

## Log Format

All logs in this directory must follow: `dd-mm-yyyy_[build_name].md`

Example: `02-11-2025_algorithm_interface_implementation.md`

## Log Content Guidelines

Keep logs CONCISE and BRIEF. Use bullet points, not prose.

Include:
- **Plan Reference:** Which Planner document is being executed
- **Implementation Summary:** What was built (high-level)
- **Key Decisions:** Important choices made during implementation
- **Deviations from Plan:** If approach changed, why
- **Files Modified/Created:** List with line counts
- **Tests Added:** Test coverage
- **Issues Encountered:** Problems and how resolved
- **Next Steps:** What should happen next (usually Validator)

Omit:
- Line-by-line code descriptions
- Obvious implementation details
- Copy-pasted code (use file paths instead)

## When to Use This Agent

Invoke Builder for:
- Implementing Planner's design plans
- Writing new feature code
- Creating test suites
- Refactoring existing code (carefully)
- Bug fixes (simple ones; complex bugs go to Debugger)

## Delegation to Gemini

To save context, delegate documentation writing:
```bash
gemini -p 'Write concise build log: implemented algorithm interface with base class, result dataclass, and validator. Added 15 tests. Use bullet points.'
```

## Current Phase Status

- Phase 1: Graph Generation - COMPLETE (previous Builder work)
- Phase 2: Algorithm Benchmarking - READY FOR IMPLEMENTATION
- Phase 3+: Awaiting Phase 2 completion

## Reference Documentation

When implementing, reference:
- `/CLAUDE.md` - Architectural principles
- `/planner/[date]_[plan].md` - Your implementation plan
- `/guides/0X_phase_name.md` - Phase-specific guidance
- Existing code in `/src/` - Follow patterns

## Implementation Workflow

1. **Read Plan:** Understand Planner's design thoroughly
2. **Setup:** Create necessary files/directories
3. **Implement:** Follow plan steps sequentially
4. **Test:** Write tests as you go, not after
5. **Validate:** Run tests locally before marking complete
6. **Document:** Write concise log (or use Gemini)
7. **Handoff:** Pass to Validator for verification

## Code Quality Standards

- **Follow architectural principles** from CLAUDE.md
- **Write tests first** (TDD) or alongside code
- **Use type hints** for function signatures
- **Document WHY** in comments, not WHAT
- **Handle errors** gracefully with clear messages
- **No magic numbers** - use named constants
- **Keep functions focused** - single responsibility
- **DRY principle** - don't repeat yourself

## Testing Requirements

Every new feature must have:
- Unit tests for individual functions
- Integration tests for component interactions
- Edge case tests (empty inputs, invalid data, etc.)
- Performance tests if relevant (Phase 1 had benchmarks)

Run tests before marking work complete:
```bash
python -m pytest src/tests/
```

## Example Build Log Structure

```markdown
# Build Log: Algorithm Interface Implementation

Date: 02-11-2025
Phase: 2
Builder: Haiku
Plan: /planner/02-11-2025_algorithm_interface_design.md

## Implementation Summary
- Created `src/algorithms/` package
- Implemented `AlgorithmBase` abstract class
- Implemented `AlgorithmResult` dataclass
- Implemented `TourValidator` utility
- Created 18 unit tests (all passing)

## Key Decisions
- Used `dataclass` for AlgorithmResult (cleaner than NamedTuple)
- Validator checks in specific order: length → edges → cycle (fail fast)
- Timeout handled via `signal.alarm()` (Unix) with fallback for Windows

## Files Created
- src/algorithms/__init__.py (52 lines)
- src/algorithms/base.py (87 lines)
- src/algorithms/result.py (34 lines)
- src/algorithms/validator.py (156 lines)
- src/tests/test_algorithm_interface.py (234 lines)

## Tests Added
Unit tests (18 total):
- test_algorithm_result_creation (3 tests)
- test_tour_validator_valid_tours (5 tests)
- test_tour_validator_invalid_tours (7 tests)
- test_validator_edge_cases (3 tests)

All tests passing: ✓

## Issues Encountered
1. Windows doesn't support `signal.alarm()` for timeout
   - Solution: Added threading-based fallback
2. Validator needed to handle self-loops in tour
   - Solution: Added explicit self-loop check

## Next Steps
- Validator should run full test suite
- If passing, proceed to implementing first algorithm (nearest neighbor)
- See /planner/ for next plan
```

## Common Patterns from Phase 1

When implementing, follow patterns from existing code:

**Generator Pattern:**
```python
class MyGenerator:
    def generate(self, n_vertices, **kwargs):
        # Implementation
        return adjacency_matrix, metadata
```

**Verification Pattern:**
```python
result = verifier.verify_property(data)
if not result.passed:
    logger.warning(f"Verification failed: {result.errors}")
```

**Testing Pattern:**
```python
def test_feature_with_valid_input():
    # Given
    input_data = create_test_data()
    # When
    result = function_under_test(input_data)
    # Then
    assert result.property == expected_value
```

## Notes

- If plan is ambiguous, ask Planner for clarification
- If you discover issues with plan during implementation, document in log
- Prefer small, incremental changes over large rewrites
- Commit logical units of work (one feature per commit)
