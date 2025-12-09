# Phase 5 Testing Plan: Prompts 5-8

**Status**: Implementation Complete (11-27-2025), Tests Needed
**Target**: ~50 new tests to achieve complete Phase 5 coverage
**Current**: 45 tests (Prompts 1-4 only)
**Goal**: 95 tests total for Phase 5

---

## Overview

Phase 5 Prompts 5-8 have complete implementations but no test coverage:
- **Prompt 5**: validation.py (352 lines) - Stage output validation
- **Prompt 6**: profiling.py (366 lines) - Performance monitoring
- **Prompt 7**: parallel.py (346 lines) - Parallel execution
- **Prompt 8**: error_handling.py (360 lines) - Fault tolerance

All modules are exported in `src/pipeline/__init__.py` and have comprehensive documentation in `src/pipeline/CLAUDE.md`.

---

## Test Suite Structure

Add to existing `src/tests/test_phase5_pipeline.py` file.

### Imports Needed
```python
from pipeline import (
    # Existing imports (Prompts 1-4)...

    # Prompt 5: Validation
    StageValidator,
    ValidationError,

    # Prompt 6: Profiling
    PerformanceMonitor,
    PerformanceMetrics,
    RuntimeProfiler,
    profile_stage,

    # Prompt 7: Parallel
    ParallelExecutor,
    ParallelConfig,
    ResourceManager,
    create_parallel_executor,

    # Prompt 8: Error Handling
    ErrorHandler,
    ErrorRecord,
    Checkpoint,
    retry_with_backoff,
    try_continue,
    graceful_degradation
)
```

---

## Prompt 5: Stage Validation Tests

**Test Class**: `TestStageValidator` (~12 tests)

### Test Coverage

1. **Graph Generation Validation** (3 tests)
   - Test valid graph directory with proper JSON files
   - Test missing directory raises ValidationError
   - Test empty directory raises ValidationError

2. **Benchmarking Validation** (3 tests)
   - Test valid benchmark results with tours
   - Test invalid tours (not Hamiltonian) caught
   - Test missing algorithm-graph combinations detected

3. **Feature Extraction Validation** (3 tests)
   - Test valid features (no NaN/inf values)
   - Test NaN values caught and reported
   - Test feature count matches expected

4. **Model Training Validation** (3 tests)
   - Test valid model files loadable
   - Test missing model files raise ValidationError
   - Test validation report structure

### Implementation Notes
- Create temporary test directories with mock data
- Use `tempfile.TemporaryDirectory()` for cleanup
- Test both success and failure paths
- Verify ValidationError messages are helpful

---

## Prompt 6: Performance Profiling Tests

**Test Classes**:
- `TestPerformanceMonitor` (~7 tests)
- `TestRuntimeProfiler` (~5 tests)

### TestPerformanceMonitor Coverage

1. **Basic Monitoring** (3 tests)
   - Test start/stop tracking duration correctly
   - Test memory delta tracking (before/after)
   - Test CPU usage capture

2. **Multiple Operations** (2 tests)
   - Test tracking multiple named operations
   - Test get_all_metrics returns all tracked operations

3. **Persistence** (2 tests)
   - Test save_metrics to JSON
   - Test metrics serialization/deserialization

### TestRuntimeProfiler Coverage

1. **Context Manager Profiling** (2 tests)
   - Test profiling within context manager
   - Test nested profiling contexts

2. **Statistics Generation** (2 tests)
   - Test get_statistics returns min/max/mean/median
   - Test scaling analysis with graph sizes

3. **Decorator Profiling** (1 test)
   - Test @profile_stage decorator auto-profiles function

### Implementation Notes
- Use `time.sleep()` for predictable timing tests
- Mock or measure actual memory allocation
- Test decorator functionality with dummy functions
- Verify metrics are reasonable (no negative durations)

---

## Prompt 7: Parallel Execution Tests

**Test Classes**:
- `TestParallelExecutor` (~7 tests)
- `TestResourceManager` (~5 tests)

### TestParallelExecutor Coverage

1. **Parallel Graph Generation** (2 tests)
   - Test parallel generation produces correct count
   - Test results equivalent to sequential generation

2. **Parallel Benchmarking** (2 tests)
   - Test parallel algorithm execution
   - Test timeout handling for slow algorithms

3. **Parallel Feature Extraction** (2 tests)
   - Test parallel feature extraction
   - Test results match sequential extraction

4. **Configuration** (1 test)
   - Test ParallelConfig initialization and validation

### TestResourceManager Coverage

1. **Resource Tracking** (2 tests)
   - Test reserve/release resources correctly tracked
   - Test available resources calculated correctly

2. **Task Admission Control** (2 tests)
   - Test can_start_task respects memory limits
   - Test can_start_task respects CPU limits

3. **System Monitoring** (1 test)
   - Test get_system_resources returns current state

### Implementation Notes
- Use small test cases to avoid long execution times
- Test with n_workers=2 for predictable behavior
- Mock psutil functions for deterministic tests
- Verify parallel results match sequential baseline
- Test both multiprocessing backend behavior

---

## Prompt 8: Error Handling Tests

**Test Classes**:
- `TestErrorHandler` (~8 tests)
- `TestCheckpoint` (~3 tests)
- `TestRetryDecorators` (~3 tests)

### TestErrorHandler Coverage

1. **Error Recording** (3 tests)
   - Test record_error stores error details
   - Test error categorization (recoverable vs fatal)
   - Test error summary generation

2. **Error Retrieval** (2 tests)
   - Test get_errors_by_stage filtering
   - Test get_summary statistics

3. **Error Persistence** (2 tests)
   - Test save_errors to JSON
   - Test load errors from file

4. **Multiple Errors** (1 test)
   - Test handling many errors across stages

### TestCheckpoint Coverage

1. **Save/Load** (2 tests)
   - Test save checkpoint data
   - Test load checkpoint data

2. **File Handling** (1 test)
   - Test exists() method
   - Test checkpoint file creation

### TestRetryDecorators Coverage

1. **Retry with Backoff** (1 test)
   - Test retry_with_backoff eventually succeeds
   - Test max_attempts limit respected

2. **Try Continue** (1 test)
   - Test try_continue logs error but continues
   - Test function execution proceeds after failure

3. **Graceful Degradation** (1 test)
   - Test graceful_degradation returns default on failure
   - Test successful execution returns actual value

### Implementation Notes
- Use mock exceptions for controlled failure testing
- Test decorator behavior with dummy functions
- Verify retry delays with time tracking
- Test checkpoint persistence with tempfile
- Ensure error messages contain useful information

---

## Testing Strategy

### Test Organization
1. Add all tests to existing `test_phase5_pipeline.py`
2. Group by prompt (Prompt 5, 6, 7, 8)
3. Use descriptive test names: `test_validator_catches_nan_features`

### Test Independence
- Each test should be independent (no shared state)
- Use `setUp()` and `tearDown()` for common initialization
- Clean up temp files/directories in tearDown

### Mock Strategy
- Mock file I/O where appropriate (faster tests)
- Mock psutil for deterministic resource tests
- Use real implementations for integration smoke tests

### Edge Cases to Test
- Empty inputs (no graphs, no algorithms)
- Invalid inputs (malformed JSON, negative numbers)
- Boundary conditions (single graph, maximum workers)
- Error conditions (missing files, timeouts)

---

## Dependencies

All tests should work with existing dependencies:
- Python stdlib (unittest, tempfile, json, time)
- numpy (already required)
- psutil (already in requirements.txt for profiling)

No additional dependencies needed.

---

## Success Criteria

- **Coverage**: All 50 tests pass (100% pass rate)
- **Completeness**: Every public method in Prompts 5-8 tested
- **Quality**: Tests catch actual bugs if implementation broken
- **Speed**: Full Phase 5 suite runs in <30 seconds
- **Documentation**: Test names clearly describe what's tested

---

## Estimated Effort

- **Setup**: 1 hour (review implementations, plan structure)
- **Prompt 5 Tests**: 2-3 hours (12 tests, file I/O complexity)
- **Prompt 6 Tests**: 2-3 hours (12 tests, timing/profiling complexity)
- **Prompt 7 Tests**: 3-4 hours (12 tests, multiprocessing complexity)
- **Prompt 8 Tests**: 2-3 hours (14 tests, decorator testing)
- **Integration/Cleanup**: 1-2 hours (verify all pass, fix issues)

**Total**: 11-16 hours for complete test implementation

---

## Next Steps

1. Review existing test structure in `test_phase5_pipeline.py`
2. Implement Prompt 5 tests (validation - lowest complexity)
3. Implement Prompt 6 tests (profiling - moderate complexity)
4. Implement Prompt 8 tests (error handling - moderate complexity)
5. Implement Prompt 7 tests (parallel - highest complexity, save for last)
6. Run full test suite and verify 95 tests pass
7. Update documentation with final test counts

---

## References

- **Implementation**: `src/pipeline/validation.py`, `profiling.py`, `parallel.py`, `error_handling.py`
- **Documentation**: `src/pipeline/CLAUDE.md` (sections for Prompts 5-8 with usage examples)
- **Existing Tests**: `src/tests/test_phase5_pipeline.py` (Prompts 1-4 as reference)
- **Test Guide**: `src/tests/CLAUDE.md` (testing standards and conventions)

---

**Plan Created**: 2025-11-28
**Status**: Ready for implementation
**Priority**: High (blocks Phase 5 completion)
