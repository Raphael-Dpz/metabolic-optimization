import torch
import cvxpy as cp
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

class DataModule:
    """
    Generates Problems with parameters feature-dependant or not. 
    x* solution is generated using CLARABEL, the most precise of the available solvers
    """
    def __init__(self, config):
        self.num_samples = config.get("num_samples", 1000)
        self.num_fluxes = config.get("num_fluxes", 20)          
        self.num_metabolites = config.get("num_metabolites", 10) 
        self.feature_dim = config.get("feature_dim", 5)          
        self.batch_size = config.get("batch_size", 32)
        self.l2_reg = config.get("l2_reg", 0.1)
        
        # A tuple of strings dictating which parameters depend on the features/data
        self.feature_dependent = config.get("feature_dependent", ("lb", "ub",)) 
        
        # Define the Oracle differentiable solver method. Must be a cp method
        self.oracle_solver_name = config.get("oracle_solver", "CLARABEL")
        self.oracle_backend = getattr(cp, self.oracle_solver_name.upper())

    def _generate_static_bases(self):
        """
        Generates the base random values for all parameters. 
        If a parameter is NOT feature-dependent, it will stay at this exact value for every sample.
        """
        bases = {
            "A": np.random.randn(self.num_metabolites, self.num_fluxes),
            "b": np.random.randn(self.num_metabolites),
            "c": np.random.randn(self.num_fluxes),
            "lb": np.zeros(self.num_fluxes),
            "ub": np.ones(self.num_fluxes) * 10.0
        }
        
        # Projection matrices to map features to the specific parameter dimensions
        # We only create these if the parameter is in the feature_dependent list
        projectors = {}
        if "c" in self.feature_dependent:
            projectors["c"] = np.random.randn(self.feature_dim, self.num_fluxes)
        if "b" in self.feature_dependent:
            projectors["b"] = np.random.randn(self.feature_dim, self.num_metabolites)
        if "lb" in self.feature_dependent:
            projectors["lb"] = np.random.randn(self.feature_dim, self.num_fluxes)
        if "ub" in self.feature_dependent:
            projectors["ub"] = np.random.randn(self.feature_dim, self.num_fluxes)
        if "A" in self.feature_dependent:
            # A is a matrix, so we map features to a flattened A, then reshape
            projectors["A"] = np.random.randn(self.feature_dim, self.num_metabolites * self.num_fluxes)

        return bases, projectors

    def generate_datasets(self):
        bases, projectors = self._generate_static_bases()
        
        # Storage for our valid, solvable dataset
        data_store = {"feat": [], "A": [], "b": [], "c": [], "lb": [], "ub": [], "x": [],
                      "oracle_time": [], "oracle_iters": []}
        
        print(f"Generating data. Feature-dependent parameters: {self.feature_dependent}")
        
        skipped_count = 0
        milestone_step = max(1, self.num_samples // 10) # 10% increments
        last_milestone_printed = 0
        
        while len(data_store["feat"]) < self.num_samples:
            feat = np.random.randn(self.feature_dim)
            
            # --- 1. Construct parameters for this specific sample ---
            # Start with static bases
            A_curr = bases["A"].copy()
            b_curr = bases["b"].copy()
            c_curr = bases["c"].copy()
            lb_curr = bases["lb"].copy()
            ub_curr = bases["ub"].copy()
            
            # Add feature dependencies dynamically
            if "c" in self.feature_dependent:
                c_curr = np.dot(feat, projectors["c"])
            if "b" in self.feature_dependent:
                b_curr = np.dot(feat, projectors["b"])
            if "lb" in self.feature_dependent:
                # We use absolute value to ensure bounds make physical sense
                lb_curr = np.abs(np.dot(feat, projectors["lb"]))
            if "ub" in self.feature_dependent:
                # Ensure ub is strictly greater than lb
                ub_curr = lb_curr + 1.0 + np.abs(np.dot(feat, projectors["ub"]))
            if "A" in self.feature_dependent:
                flat_A = np.dot(feat, projectors["A"])
                A_curr = flat_A.reshape((self.num_metabolites, self.num_fluxes))

            # --- 2. Solve the Ground Truth using CVXPY ---
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
                continue 

            # --- 3. Filter valid solutions ---
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and x.value is not None:
                if np.mean(np.abs(x.value)) > 0.1:  # Non-triviality check -> maybe to change                    
                    data_store["feat"].append(feat)
                    data_store["A"].append(A_curr)
                    data_store["b"].append(b_curr)
                    data_store["c"].append(c_curr)
                    data_store["lb"].append(lb_curr)
                    data_store["ub"].append(ub_curr)
                    data_store["x"].append(x.value)
                    data_store["oracle_time"].append(prob.solver_stats.solve_time)
                    data_store["oracle_iters"].append(prob.solver_stats.num_iters)
                    
                    # Check if we hit a 10% milestone
                    current_n = len(data_store["feat"])
                    if current_n % milestone_step == 0 and current_n > last_milestone_printed:
                        percentage = int((current_n / self.num_samples) * 100)
                        print(f"({current_n} generated ({percentage}%), {skipped_count} skipped)")
                        last_milestone_printed = current_n
                else:
                    # Failed triviality check
                    skipped_count += 1
            else:
                # Failed feasibility/optimality check
                skipped_count += 1

        # Convert everything to PyTorch Tensors
        tensors = tuple(torch.tensor(np.array(data_store[k]), dtype=torch.float32) 
                        for k in ["feat", "A", "b", "c", "lb", "ub", "x"])
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