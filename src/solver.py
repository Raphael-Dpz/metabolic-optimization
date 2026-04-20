import time
import cvxpy as cp
import numpy as np

class BenchmarkSolver:
    """
    Standard non-differentiable solver used for evaluating benchmarks.
    It handles Cold-Starts, Warm-Starts, and Predict-Then-Optimize tasks.
    """
    def __init__(self, config):
        self.l2_reg = config.get("l2_reg", 0.1)
        # The default solver to use if not specified during the call (e.g., OSQP)
        self.default_solver = config.get("benchmark_solver", "OSQP")

    def _to_numpy(self, tensor):
        """Helper to safely convert PyTorch tensors to Numpy arrays."""
        if hasattr(tensor, 'detach'):
            return tensor.detach().cpu().numpy()
        return tensor

    def solve(self, A, b, c, lb, ub, solver_name=None, warm_start_x=None, max_iters=None):
        """
        Universal solve method.
        - If 'warm_start_x' is provided, it performs a Semi-Amortized Warm Start.
        - If 'warm_start_x' is None, it performs a Cold Start.
        - For Predict-Then-Optimize, you just pass the predicted parameters (like predicted_c)
          instead of the true parameters and no warm_start_x.
        """
        # 1. Convert everything to Numpy
        A_np = self._to_numpy(A)
        b_np = self._to_numpy(b)
        c_np = self._to_numpy(c)
        lb_np = self._to_numpy(lb)
        ub_np = self._to_numpy(ub)

        num_fluxes = c_np.shape[0]
        x = cp.Variable(num_fluxes)

        # 2. Warm Start Initialization
        is_warm_start = False
        if warm_start_x is not None:
            # We inject the Neural Network's prediction directly into the CVXPY variable
            x.value = self._to_numpy(warm_start_x)
            is_warm_start = True

        # 3. Formulate the CVXPY Problem
        objective = cp.Minimize(c_np.T @ x + self.l2_reg * cp.sum_squares(x))
        constraints = [
            A_np @ x == b_np,
            x >= lb_np,
            x <= ub_np
        ]
        prob = cp.Problem(objective, constraints)

        # 4. Configure the chosen solver backend
        # If a solver is specified in the arguments, use it; otherwise use the default.
        backend_name = solver_name if solver_name else self.default_solver
        backend = getattr(cp, backend_name.upper())
        
        solve_kwargs = {"solver": backend, "warm_start": is_warm_start}
        
        # Some solvers (like OSQP/SCS) accept max_iters to limit refinement steps
        if max_iters is not None:
            solve_kwargs["max_iters"] = max_iters

        # 5. Solve and Track Metrics
        start_time = time.time()
        try:
            prob.solve(**solve_kwargs)
            solve_time = time.time() - start_time

            # Check if the solver successfully found a solution
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Extract iteration count safely (some solvers don't report it)
                iters = prob.solver_stats.num_iters if hasattr(prob.solver_stats, 'num_iters') else -1
                
                return {
                    "status": prob.status,
                    "x_res": x.value,
                    "obj_val": prob.value,
                    "solve_time": solve_time,
                    "num_iters": iters
                }
            else:
                # Infeasible or unbounded
                return {"status": prob.status, "x_res": None, "obj_val": np.inf, "solve_time": solve_time, "num_iters": 0}

        except cp.SolverError:
            return {"status": "SolverError", "x_res": None, "obj_val": np.inf, "solve_time": time.time() - start_time, "num_iters": 0}