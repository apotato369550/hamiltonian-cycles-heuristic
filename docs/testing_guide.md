# 1. Quick sanity check - validate configuration without running
  python experiments/run_experiment.py config/test_config_small.yaml --dry-run

  # 2. Small test run - fastest way to test end-to-end (~2-3 minutes)
  #    Generates 2 small graphs, runs 1 algorithm, extracts features, trains model
  python experiments/run_experiment.py config/test_config_small.yaml

  # 3. Test individual stages (if you want to test incrementally)
  python experiments/run_experiment.py config/test_config_small.yaml --stage graph_generation
  python experiments/run_experiment.py config/test_config_small.yaml --stage benchmarking
  python experiments/run_experiment.py config/test_config_small.yaml --stage feature_extraction
  python experiments/run_experiment.py config/test_config_small.yaml --stage training

  # 4. Check the output structure after running
  ls -la experiments/  # Should show experiment directory with timestamp
  ls -la experiments/<experiment_id>/  # Should show metadata.json, reproducibility.json, 
  logs/, graphs/, etc.

  # 5. If small test works, try the full template (WARNING: ~30-60 minutes runtime)
  python experiments/run_experiment.py config/complete_experiment_template.yaml

  Expected behavior for successful run:
  1. Creates experiments/<timestamp>/ directory
  2. Prints stage execution logs to console
  3. Saves outputs to subdirectories (graphs/, benchmarks/, features/, models/)
  4. Creates metadata.json and reproducibility.json
  5. Exit code 0 on success

  To verify it worked:
  # Check experiment was created
  ls experiments/

  # View metadata
  cat experiments/<experiment_id>/metadata.json

  # Check reproducibility info
  cat experiments/<experiment_id>/reproducibility.json

  # View generated graphs
  ls experiments/<experiment_id>/graphs/

  # View benchmark results
  ls experiments/<experiment_id>/benchmarks/

  Start with command #2 (test_config_small.yaml) - it's the fastest way to validate everything
  works end-to-end!