import torch
import cvxpy as cp
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

class DataModule:
    def __init__(self, config):
        self.num_samples = config.get("num_samples", 1000)
        self.num_fluxes = config.get("num_fluxes", 20)          
        self.num_metabolites = config.get("num_metabolites", 10) 
        self.feature_dim = config.get("feature_dim", 5)          
        self.batch_size = config.get("batch_size", 32)
        self.l2_reg = config.get("l2_reg", 0.1)
        self.feature_dependent = config.get("feature_dependent", ("lb", "ub",)) 
        self.oracle_solver_name = config.get("oracle_solver", "CLARABEL")
        self.oracle_backend = getattr(cp, self.oracle_solver_name.upper())

    def _generate_static_bases(self):
        bases = {
            "A": np.random.randn(self.num_metabolites, self.num_fluxes),
            "c": np.random.randn(self.num_fluxes),
            "lb": np.zeros(self.num_fluxes),
            "ub": np.ones(self.num_fluxes) * 10.0
        }
        
        projectors = {}
        if "c" in self.feature_dependent:
            projectors["c"] = np.random.randn(self.feature_dim, self.num_fluxes)
        if "lb" in self.feature_dependent:
            projectors["lb"] = np.random.randn(self.feature_dim, self.num_fluxes)
        if "ub" in self.feature_dependent:
            projectors["ub"] = np.random.randn(self.feature_dim, self.num_fluxes)
        if "A" in self.feature_dependent:
            projectors["A"] = np.random.randn(self.feature_dim, self.num_metabolites * self.num_fluxes)

        return bases, projectors

    def generate_datasets(self):
        bases, projectors = self._generate_static_bases()
        
        data_store = {"feat": [], "A": [], "b": [], "c": [], "lb": [], "ub": [], "x": [],
                      "oracle_time": [], "oracle_iters": []}
        
        skipped_count = 0
        total_attempts = 0
        
        while len(data_store["feat"]) < self.num_samples:
            total_attempts += 1
            feat = np.random.randn(self.feature_dim)
            
            A_curr = bases["A"].copy()
            c_curr = bases["c"].copy()
            lb_curr = bases["lb"].copy()
            ub_curr = bases["ub"].copy()
            
            if "c" in self.feature_dependent:
                c_curr = np.dot(feat, projectors["c"])
            if "lb" in self.feature_dependent:
                lb_curr = np.abs(np.dot(feat, projectors["lb"]))
            if "ub" in self.feature_dependent:
                ub_curr = lb_curr + 1.0 + np.abs(np.dot(feat, projectors["ub"]))
            if "A" in self.feature_dependent:
                flat_A = np.dot(feat, projectors["A"])
                A_curr = flat_A.reshape((self.num_metabolites, self.num_fluxes))

            # help mathematical feasibility: generate valid hidden x, then compute b
            x_hidden = lb_curr + 0.5 * (ub_curr - lb_curr)
            b_curr = A_curr @ x_hidden

            x = cp.Variable(self.num_fluxes)
            objective = cp.Minimize(c_curr.T @ x + self.l2_reg * cp.sum_squares(x))
            constraints = [
                A_curr @ x == b_curr,
                x >= lb_curr,
                x <= ub_curr
            ]
            prob = cp.Problem(objective, constraints)
            
            try:
                prob.solve(solver=self.oracle_backend, warm_start=False)
            except cp.SolverError:
                skipped_count += 1

            status = prob.status if hasattr(prob, 'status') else "SolverError"
            
            if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and x.value is not None:
                if np.mean(np.abs(x.value)) > 0.1:  
                    data_store["feat"].append(feat)
                    data_store["A"].append(A_curr)
                    data_store["b"].append(b_curr)
                    data_store["c"].append(c_curr)
                    data_store["lb"].append(lb_curr)
                    data_store["ub"].append(ub_curr)
                    data_store["x"].append(x.value)
                    data_store["oracle_time"].append(prob.solver_stats.solve_time)
                    data_store["oracle_iters"].append(prob.solver_stats.num_iters)
                else:
                    skipped_count += 1
            else:
                skipped_count += 1

            current_valid = len(data_store["feat"])
            
            if total_attempts % 100 == 0:
                print(f"Attempts: {total_attempts} | Valid: {current_valid}/{self.num_samples} | Skipped: {skipped_count}")

            if skipped_count > 10000 and current_valid < 10:
                raise ValueError("Too many infeasible problems. Check parameter generation.")

        tensors = tuple(torch.tensor(np.array(data_store[k]), dtype=torch.float32) 
                        for k in ["feat", "A", "b", "c", "lb", "ub", "x", "oracle_time", "oracle_iters"])
        dataset = TensorDataset(*tensors)
        
        train_size = int(0.8 * self.num_samples)
        test_size = self.num_samples - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        metadata = {
            "oracle_solver": self.oracle_solver_name,
            "feature_dependent_params": self.feature_dependent,
            "num_samples": self.num_samples,
            "l2_reg_applied": self.l2_reg
        }
        return train_dataset, test_dataset, metadata

    def get_dataloaders(self):
        train_dataset, test_dataset, metadata = self.generate_datasets()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, metadata