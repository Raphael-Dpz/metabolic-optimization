import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class FeaturePredictor(nn.Module):
    """
    ARCHITECTURE: Predict-Then-Optimize (Two-Stage)
    The Neural Network that learns to predict optimization parameters from features.
    It dynamically adjusts its output size based on 'config["feature_dependent"]'.
    """
    def __init__(self, config):
        super().__init__()
        self.feature_dim = config.get("feature_dim", 5)
        self.num_fluxes = config.get("num_fluxes", 20)
        self.num_metabolites = config.get("num_metabolites", 10)
        self.feature_dependent = config.get("feature_dependent", ("lb", "up",))

        # Calculate the exact number of output neurons needed
        self.output_dim = 0
        if "c" in self.feature_dependent: self.output_dim += self.num_fluxes
        if "b" in self.feature_dependent: self.output_dim += self.num_metabolites
        if "lb" in self.feature_dependent: self.output_dim += self.num_fluxes
        if "ub" in self.feature_dependent: self.output_dim += self.num_fluxes
        if "A" in self.feature_dependent: self.output_dim += (self.num_metabolites * self.num_fluxes)

        # A simple MLP for now
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, features):
        flat_output = self.net(features)
        
        predictions = {}
        idx = 0
        
        # Slice the output tensor dynamically based on what was predicted
        if "c" in self.feature_dependent:
            predictions["c"] = flat_output[:, idx : idx + self.num_fluxes]
            idx += self.num_fluxes
            
        if "b" in self.feature_dependent:
            predictions["b"] = flat_output[:, idx : idx + self.num_metabolites]
            idx += self.num_metabolites
            
        if "lb" in self.feature_dependent:
            # Absolute value ensures lower bounds are strictly positive/valid
            predictions["lb"] = torch.abs(flat_output[:, idx : idx + self.num_fluxes])
            idx += self.num_fluxes
            
        if "ub" in self.feature_dependent:
            predictions["ub"] = flat_output[:, idx : idx + self.num_fluxes]
            idx += self.num_fluxes
            
        if "A" in self.feature_dependent:
            size = self.num_metabolites * self.num_fluxes
            flat_A = flat_output[:, idx : idx + size]
            # Reshape back to matrix format (Batch_size, Metabolites, Fluxes)
            predictions["A"] = flat_A.view(-1, self.num_metabolites, self.num_fluxes)
            idx += size

        return predictions


class DifferentiableSolver(nn.Module):
    """
    The Mathematical Optimization Layer using cvxpylayers.
    This layer takes A, b, c, lb, ub and outputs the optimal x.
    Gradients can flow backward through this layer to train the FeaturePredictor.
    """
    def __init__(self, config):
        super().__init__()
        num_fluxes = config.get("num_fluxes", 20)
        num_metabolites = config.get("num_metabolites", 10)
        l2_reg = config.get("l2_reg", 0.1)
        
        # Solver configurations for the backward pass
        self.max_iters = config.get("diff_solver_iters", 5000)
        self.eps = config.get("diff_solver_eps", 1e-4)
        self.verbose = config.get("diff_solver_verbose", False)

        self.A_param = cp.Parameter((num_metabolites, num_fluxes))
        self.b_param = cp.Parameter(num_metabolites)
        self.c_param = cp.Parameter(num_fluxes)
        self.lb_param = cp.Parameter(num_fluxes)
        self.ub_param = cp.Parameter(num_fluxes)

        self.x = cp.Variable(num_fluxes)

        objective = cp.Minimize(self.c_param.T @ self.x + l2_reg * cp.sum_squares(self.x))
        constraints = [
            self.A_param @ self.x == self.b_param,
            self.x >= self.lb_param,
            self.x <= self.ub_param
        ]
        problem = cp.Problem(objective, constraints)

        self.cvx_layer = CvxpyLayer(
            problem,
            parameters=[self.A_param, self.b_param, self.c_param, self.lb_param, self.ub_param],
            variables=[self.x]
        )

    def forward(self, A, b, c, lb, ub):
        solver_args = {
            "solve_method": self.config.get("diff_solver", "SCS"), # currently only CSC (or ECOS ?) are supported
            "max_iters": self.max_iters,
            "eps": self.eps,
            "verbose": self.verbose
        }
        x_star, = self.cvx_layer(A, b, c, lb, ub, solver_args=solver_args)
        return x_star


class EndToEndModel(nn.Module):
    """
    ARCHITECTURE: End-to-End Differentiable Model
    Combines the FeaturePredictor and DifferentiableSolver into a single pipeline.
    """
    def __init__(self, config):
        super().__init__()
        self.feature_dependent = config.get("feature_dependent", ("lb", "ub",))
        self.predictor = FeaturePredictor(config)
        self.solver = DifferentiableSolver(config)

    def forward(self, features, true_A, true_b, true_c, true_lb, true_ub):
        # 1. Predict missing parameters
        preds = self.predictor(features)

        # 2. Merge predictions with known static ground truths
        A_final = preds["A"] if "A" in self.feature_dependent else true_A
        b_final = preds["b"] if "b" in self.feature_dependent else true_b
        c_final = preds["c"] if "c" in self.feature_dependent else true_c
        lb_final = preds["lb"] if "lb" in self.feature_dependent else true_lb
        
        if "ub" in self.feature_dependent:
            # Force ub > lb mathematically to prevent infeasible solver states
            ub_final = lb_final + 1.0 + torch.abs(preds["ub"])
        else:
            ub_final = true_ub

        # 3. Solve via differentiable layer
        x_pred = self.solver(A_final, b_final, c_final, lb_final, ub_final)
        
        return x_pred, preds


class SemiAmortizedPredictor(nn.Module):
    """
    ARCHITECTURE: Semi-Amortized Model
    Neural Network designed to guess the optimal solution x*.
    It takes the entire flattened problem topology as input and outputs a warm-start x.
    """
    def __init__(self, config):
        super().__init__()
        self.num_fluxes = config.get("num_fluxes", 20)
        self.num_metabolites = config.get("num_metabolites", 10)
        
        # Input dimension: Flattened A matrix + b + c + lb + ub
        input_dim = (self.num_metabolites * self.num_fluxes) + \
                    self.num_metabolites + \
                    (3 * self.num_fluxes) 
                    
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_fluxes)
        )

    def forward(self, A, b, c, lb, ub):
        batch_size = A.shape[0]
        
        # Flatten A matrix
        A_flat = A.view(batch_size, -1)
        
        # Concatenate all parameters into one massive input vector per sample
        problem_state = torch.cat([A_flat, b, c, lb, ub], dim=1)
        
        # Predict the initial guess x^(0)
        x_init = self.net(problem_state)
        
        return x_init