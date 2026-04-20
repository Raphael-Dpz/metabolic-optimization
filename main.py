from src.pipeline import Pipeline

if __name__ == "__main__":
    # The ultimate configuration dictionary
    config = {
        # General Data settings
        "num_samples": 500,
        "batch_size": 32,
        "feature_dim": 5,
        "num_metabolites": 10,
        "num_fluxes": 20,
        "l2_reg": 0.1,
        
        # Solver configurations
        "oracle_solver": "CLARABEL",    # High precision for ground truth
        "benchmark_solver": "OSQP",     # The solver we are benchmarking
        "diff_solver": "SCS",           # Solver used in cvxpylayers (must be supported by cvxpylayers)
        "diff_solver_iters": 2000,      # Max iters for cvxpylayers (SCS)
        
        # Experiment settings
        "task": "semi_amortized",       # Choose: "semi_amortized" or "predict_then_optimize"
        "feature_dependent": ("lb", "ub",),    # Parameters that depend on features
        "epochs": 30,
        "lr": 1e-3,
        "results_dir": "results/"
    }

    # Execute
    pipeline = Pipeline(config)
    pipeline.run()