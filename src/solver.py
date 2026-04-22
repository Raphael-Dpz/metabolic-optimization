import time
import cvxpy as cp
import numpy as np

class BenchmarkSolver:
    """
    Standard non-differentiable solver used for evaluating benchmarks.
    It handles Cold-Starts, Warm-Starts, and Predict-Then-Optimize tasks.
    Uses cp.Parameter to prevent CVXPY from deleting the C++ solver cache.
    """
    def __init__(self, config):
        self.l2_reg = config.get("l2_reg", 0.1)
        self.default_solver = config.get("benchmark_solver", "OSQP")
        
        num_fluxes = config.get("num_fluxes", 20)
        num_metabolites = config.get("num_metabolites", 10)
        
        # 1. Define everything as Parameters so the problem is compiled once
        self.A_param = cp.Parameter((num_metabolites, num_fluxes))
        self.b_param = cp.Parameter(num_metabolites)
        self.c_param = cp.Parameter(num_fluxes)
        self.lb_param = cp.Parameter(num_fluxes)
        self.ub_param = cp.Parameter(num_fluxes)

        self.x = cp.Variable(num_fluxes)

        # 2. Build the static problem structure
        objective = cp.Minimize(self.c_param.T @ self.x + self.l2_reg * cp.sum_squares(self.x))
        constraints = [
            self.A_param @ self.x == self.b_param,
            self.x >= self.lb_param,
            self.x <= self.ub_param
        ]
        self.prob = cp.Problem(objective, constraints)

    def _to_numpy(self, tensor):
        if hasattr(tensor, 'detach'):
            return tensor.detach().cpu().numpy()
        return tensor

    def solve(self, A, b, c, lb, ub, solver_name=None, warm_start_x=None, max_iters=None):
        # 3. Update the values of the parameters
        self.A_param.value = self._to_numpy(A)
        self.b_param.value = self._to_numpy(b)
        self.c_param.value = self._to_numpy(c)
        self.lb_param.value = self._to_numpy(lb)
        self.ub_param.value = self._to_numpy(ub)

        # 4. Inject the Neural Network's guess
        is_warm_start = False
        if warm_start_x is not None:
            self.x.value = self._to_numpy(warm_start_x)
            is_warm_start = True

        backend_name = solver_name if solver_name else self.default_solver
        backend = getattr(cp, backend_name.upper())
        
        solve_kwargs = {"solver": backend, "warm_start": is_warm_start}
        if max_iters is not None:
            solve_kwargs["max_iters"] = max_iters

        start_time = time.time()
        try:
            # The solver is now called on the cached workspace
            self.prob.solve(**solve_kwargs)
            solve_time = time.time() - start_time

            if self.prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                iters = self.prob.solver_stats.num_iters if hasattr(self.prob.solver_stats, 'num_iters') else 0
                return {
                    "status": self.prob.status,
                    "x_res": self.x.value,
                    "obj_val": self.prob.value,
                    "solve_time": solve_time,
                    "num_iters": iters
                }
            else:
                return {"status": self.prob.status, "x_res": None, "obj_val": np.inf, "solve_time": solve_time, "num_iters": 0}

        except cp.SolverError:
            return {"status": "SolverError", "x_res": None, "obj_val": np.inf, "solve_time": time.time() - start_time, "num_iters": 0}