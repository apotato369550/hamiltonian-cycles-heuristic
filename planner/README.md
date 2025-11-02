# Planner Agent Directory

**Agent Type:** Planner (Sonnet 4, Blue)
**Specialization:** Elite technical planning and software architecture

## Purpose

The Planner creates detailed, executable implementation plans for complex phases of the TSP research pipeline. Plans are optimized for execution by less capable agents (Builder/Haiku) and account for architectural constraints.

## Responsibilities

- Translate high-level requirements into concrete, step-by-step plans
- Break down complex phases into manageable subtasks
- Analyze existing codebase and account for architectural principles
- Identify dependencies, risks, and edge cases
- Optimize implementation approach for efficiency and maintainability
- Create debugging plans when Validator identifies complex issues

## Log Format

All logs in this directory must follow: `dd-mm-yyyy_[plan_name].md`

Example: `02-11-2025_algorithm_interface_design.md`

## Log Content Guidelines

Keep logs CONCISE and BRIEF. Use bullet points, not prose.

Include:
- **Objective:** What is being planned
- **Context:** Relevant background from guides or CLAUDE.md
- **Approach:** High-level strategy
- **Steps:** Numbered, concrete implementation steps
- **Dependencies:** What must exist before this can be implemented
- **Risks:** Potential issues and mitigation strategies
- **Success Criteria:** How to know when implementation is complete
- **Testing Strategy:** How to verify correctness

Omit:
- Verbose explanations (link to guides instead)
- Implementation details (that's Builder's job)
- Repetitive content

## When to Use This Agent

Invoke Planner for:
- Starting a new phase (Phase 2, 3, 4, 5)
- Complex architectural decisions
- Breaking down ambiguous requirements
- Planning debugging strategies for difficult bugs
- Optimizing implementation approaches with trade-offs

## Current Phase Status

- Phase 1: Graph Generation - COMPLETE
- Phase 2: Algorithm Benchmarking - NEXT (needs planning)
- Phase 3: Feature Engineering - FUTURE
- Phase 4: Machine Learning - FUTURE
- Phase 5: Pipeline Integration - FUTURE
- Phase 6: Analysis - ONGOING

## Reference Documentation

When creating plans, reference:
- `/CLAUDE.md` - Project context and architectural principles
- `/guides/README.md` - Master guide overview
- `/guides/0X_phase_name.md` - Phase-specific implementation guidance

## Example Plan Structure

```markdown
# Plan: Algorithm Benchmarking Interface Design

Date: 02-11-2025
Phase: 2
Planner: Sonnet 4

## Objective
Design unified algorithm interface for Phase 2 benchmarking pipeline.

## Context
Per /guides/02_algorithm_benchmarking_pipeline.md, Prompt 1:
- Need consistent interface for all TSP algorithms
- Must support: solve(graph) → tour, quality, metadata
- Enable fair comparison and easy addition of new algorithms

## Approach
Interface pattern: Abstract base class + concrete implementations
- Base: Algorithm interface with solve() method
- Derived: NearestNeighbor, Greedy, Christofides, SingleAnchor, etc.

## Implementation Steps

1. Create `src/algorithms/__init__.py` package
2. Define `AlgorithmBase` abstract class:
   - Abstract method: solve(graph) → AlgorithmResult
   - Properties: name, version, parameters
   - Optional: timeout support, progress callbacks
3. Define `AlgorithmResult` dataclass:
   - tour: List[int]
   - tour_weight: float
   - runtime_seconds: float
   - metadata: Dict[str, Any]
   - success: bool
4. Define `TourValidator` utility:
   - validate_hamiltonian_cycle(graph, tour) → ValidationResult
   - Check: correct length, valid edges, forms cycle
5. Create test file: `src/tests/test_algorithm_interface.py`

## Dependencies
- Phase 1 graph generation (COMPLETE)
- Graph data structures from `src/graph_generation/`

## Risks
- Algorithm timeouts: Need clean timeout handling
- Invalid tours: Validator must catch all cases
- Metadata consistency: Different algorithms track different metadata

Mitigation:
- Timeout via signals or threading
- Comprehensive validator tests
- Document standard metadata keys

## Success Criteria
- Base class defined with clear interface
- Example implementation (nearest neighbor) passes tests
- Validator catches all invalid tour types
- Tests cover: valid tours, invalid tours, edge cases

## Testing Strategy
Unit tests:
- Test validator with known valid/invalid tours
- Test base class instantiation
- Test result dataclass

Integration tests:
- Implement simple algorithm (nearest neighbor)
- Run on Phase 1 graphs
- Verify results are valid
```

## Notes

- Plans should be complete enough that Builder can execute without ambiguity
- If Builder encounters issues, they should ask Planner for clarification
- Plans are living documents: update if approach changes during implementation
