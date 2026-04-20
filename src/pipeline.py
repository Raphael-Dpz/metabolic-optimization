import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.data import DataModule
from src.model import FeaturePredictor, EndToEndModel, SemiAmortizedPredictor
from src.solver import BenchmarkSolver

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"run_{config.get('task')}_{timestamp}"
        self.exp_dir = os.path.join(config.get("results_dir", "results/"), exp_name)
        self.weights_dir = os.path.join(self.exp_dir, "weights")
        
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

        # 1. Save the exact configuration used for this run
        with open(os.path.join(self.exp_dir, "config_used.json"), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # 2. Initialize Data
        print(f"Initializing Experiment in: {self.exp_dir}")
        self.data_module = DataModule(config)
        self.train_loader, self.test_loader, self.metadata = self.data_module.get_dataloaders()
        
        # 3. Save dataset metadata strictly linked to this run
        with open(os.path.join(self.exp_dir, "dataset_metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=4)
            
        self.benchmark_solver = BenchmarkSolver(config)
        self.target_solver = config.get("benchmark_solver", "OSQP")

    # ==========================================
    # TRAINING LOOPS
    # ==========================================

    def train_semi_amortized(self, epochs=50):
        """Trains the NN to guess x* from the problem parameters."""
        print(f"\n--- Training Semi-Amortized Predictor on {self.device} ---")
        model = SemiAmortizedPredictor(self.config).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.get("lr", 1e-3))
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.train_loader:
                feat, A, b, c, lb, ub, true_x, oracle_time, oracle_iters = [tensor.to(self.device) for tensor in batch]
                
                optimizer.zero_grad()
                x_init = model(A, b, c, lb, ub)
                
                loss = criterion(x_init, true_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(self.train_loader):.6f}")
                checkpoint_path = os.path.join(self.weights_dir, f"semi_amortized_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                
        final_path = os.path.join(self.weights_dir, "semi_amortized_final.pth")
        torch.save(model.state_dict(), final_path)
        print(f"Model saved to {self.weights_dir}")
                
        return model

    def train_pto_models(self, epochs=50):
        """Trains both the Two-Stage and End-to-End models for comparison."""
        print(f"\n--- Training Two-Stage Model ---")
        two_stage_model = FeaturePredictor(self.config).to(self.device)
        ts_optimizer = optim.Adam(two_stage_model.parameters(), lr=self.config.get("lr", 1e-3))
        ts_criterion = nn.MSELoss()

        print(f"--- Training End-to-End Model ---")
        e2e_model = EndToEndModel(self.config).to(self.device)
        e2e_optimizer = optim.Adam(e2e_model.parameters(), lr=self.config.get("lr", 1e-3))
        e2e_criterion = nn.MSELoss()

        for epoch in range(epochs):
            ts_loss_total, e2e_loss_total = 0, 0
            
            for batch in self.train_loader:
                feat, A, b, c, lb, ub, true_x, _, _ = [tensor.to(self.device) for tensor in batch]
                
                # --- Two-Stage Training Step ---
                ts_optimizer.zero_grad()
                ts_preds = two_stage_model(feat)
                # Sum MSE loss over all predicted parameters (e.g., just 'c', or 'c' and 'ub')
                ts_loss = sum(ts_criterion(ts_preds[k], eval(k)) for k in ts_preds.keys())
                ts_loss.backward()
                ts_optimizer.step()
                ts_loss_total += ts_loss.item()
                
                # --- End-to-End Training Step ---
                e2e_optimizer.zero_grad()
                x_pred, _ = e2e_model(feat, A, b, c, lb, ub)
                e2e_loss = e2e_criterion(x_pred, true_x)
                e2e_loss.backward()
                e2e_optimizer.step()
                e2e_loss_total += e2e_loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | TS Loss (Params): {ts_loss_total/len(self.train_loader):.4f} | E2E Loss (Decisions): {e2e_loss_total/len(self.train_loader):.4f}")
                ts_ckpt_path = os.path.join(self.weights_dir, f"two_stage_epoch_{epoch+1}.pth")
                e2e_ckpt_path = os.path.join(self.weights_dir, f"end_to_end_epoch_{epoch+1}.pth")
                
                torch.save(two_stage_model.state_dict(), ts_ckpt_path)
                torch.save(e2e_model.state_dict(), e2e_ckpt_path)

        torch.save(two_stage_model.state_dict(), os.path.join(self.weights_dir, "two_stage_final.pth"))
        torch.save(e2e_model.state_dict(), os.path.join(self.weights_dir, "end_to_end_final.pth"))
        print(f"PTO Models saved to {self.weights_dir}")
        
        return two_stage_model, e2e_model

    # ==========================================
    # BENCHMARK TRACKS
    # ==========================================

    def run_semi_amortized_benchmark(self, model):
        """Track 1: Tests if the NN warm-start speeds up the target solver."""
        print(f"\n--- Running Semi-Amortized Benchmark with {self.target_solver} ---")
        model.eval()
        
        results = {"cold_iters": [], "warm_iters": [], "cold_time": [], "warm_time": []}
        
        with torch.no_grad():
            for batch in self.test_loader:
                feat, A, b, c, lb, ub, true_x, _, _ = batch
                
                # We benchmark one sample at a time
                for i in range(len(feat)):
                    Ai, bi, ci, lbi, ubi = A[i], b[i], c[i], lb[i], ub[i]
                    
                    # 1. NN Prediction for Warm Start
                    # Reshape to keep batch dimension of 1
                    x_init = model(Ai.unsqueeze(0), bi.unsqueeze(0), ci.unsqueeze(0), lbi.unsqueeze(0), ubi.unsqueeze(0)).squeeze(0)
                    
                    # 2. Cold Start Run
                    cold_res = self.benchmark_solver.solve(
                        Ai, bi, ci, lbi, ubi, 
                        solver_name=self.target_solver, warm_start_x=None
                    )
                    
                    # 3. Warm Start Run
                    warm_res = self.benchmark_solver.solve(
                        Ai, bi, ci, lbi, ubi, 
                        solver_name=self.target_solver, warm_start_x=x_init
                    )
                    
                    if cold_res["status"] == "optimal" and warm_res["status"] == "optimal":
                        results["cold_iters"].append(cold_res["num_iters"])
                        results["warm_iters"].append(warm_res["num_iters"])
                        results["cold_time"].append(cold_res["solve_time"])
                        results["warm_time"].append(warm_res["solve_time"])

        # Print Aggregated Results
        print("\n=== SEMI-AMORTIZED RESULTS ===")
        print(f"Avg {self.target_solver} Cold Iters: {np.mean(results['cold_iters']):.1f}")
        print(f"Avg {self.target_solver} Warm Iters: {np.mean(results['warm_iters']):.1f}")
        print(f"Avg Time Saved: {(np.mean(results['cold_time']) - np.mean(results['warm_time'])) * 1000:.2f} ms per solve")
        
        # Save to disk
        results_path = os.path.join(self.exp_dir, "benchmark_metrics.npy")
        np.save(results_path, results)
        print(f"Benchmark metrics saved to {results_path}")

    def run_pto_benchmark(self, ts_model, e2e_model):
        """Track 2: Compares objective regret between Two-Stage and End-to-End."""
        print(f"\n--- Running Predict-Then-Optimize Benchmark ---")
        ts_model.eval()
        e2e_model.eval()
        
        results = {"ts_regret": [], "e2e_regret": []}
        
        with torch.no_grad():
            for batch in self.test_loader:
                feat, A, b, c, lb, ub, true_x, _, _ = batch
                
                for i in range(len(feat)):
                    fi, Ai, bi, ci, lbi, ubi = feat[i], A[i], b[i], c[i], lb[i], ub[i]
                    
                    # Oracle True Objective (calculated perfectly during data generation)
                    oracle_obj = (ci.T @ true_x[i] + self.config.get("l2_reg", 0.1) * torch.sum(true_x[i]**2)).item()
                    
                    # 1. Two-Stage Decision
                    ts_preds = ts_model(fi.unsqueeze(0))
                    # Fallback to true parameter if not predicted
                    ts_c = ts_preds["c"].squeeze(0) if "c" in ts_preds else ci
                    ts_lb = ts_preds["lb"].squeeze(0) if "lb" in ts_preds else lbi
                    ts_ub = ts_preds["ub"].squeeze(0) if "ub" in ts_preds else ubi
                    
                    ts_res = self.benchmark_solver.solve(Ai, bi, ts_c, ts_lb, ts_ub, solver_name=self.target_solver)
                    
                    # 2. End-to-End Decision (Using its internal FeaturePredictor)
                    e2e_preds = e2e_model.predictor(fi.unsqueeze(0))
                    e2e_c = e2e_preds["c"].squeeze(0) if "c" in e2e_preds else ci
                    e2e_lb = e2e_preds["lb"].squeeze(0) if "lb" in e2e_preds else lbi
                    e2e_ub = e2e_preds["ub"].squeeze(0) if "ub" in e2e_preds else ubi
                    
                    e2e_res = self.benchmark_solver.solve(Ai, bi, e2e_c, e2e_lb, e2e_ub, solver_name=self.target_solver)
                    
                    # 3. Calculate Regret (How much worse the decision is compared to the Oracle)
                    if ts_res["status"] == "optimal" and e2e_res["status"] == "optimal":
                        ts_actual_obj = ci.numpy().T @ ts_res["x_res"] + self.config.get("l2_reg", 0.1) * np.sum(ts_res["x_res"]**2)
                        e2e_actual_obj = ci.numpy().T @ e2e_res["x_res"] + self.config.get("l2_reg", 0.1) * np.sum(e2e_res["x_res"]**2)
                        
                        results["ts_regret"].append(abs(ts_actual_obj - oracle_obj))
                        results["e2e_regret"].append(abs(e2e_actual_obj - oracle_obj))

        print("\n=== PREDICT-THEN-OPTIMIZE RESULTS ===")
        print(f"Avg Two-Stage Regret: {np.mean(results['ts_regret']):.4f}")
        print(f"Avg End-to-End Regret: {np.mean(results['e2e_regret']):.4f}")
        
        results_path = os.path.join(self.exp_dir, "pto_benchmark_metrics.npy")
        np.save(results_path, results)
        print(f"Benchmark metrics saved to {results_path}")

    def run(self):
        """Main execution flow based on config settings."""
        task = self.config.get("task", "semi_amortized")
        epochs = self.config.get("epochs", 50)
        
        if task == "semi_amortized":
            model = self.train_semi_amortized(epochs=epochs)
            self.run_semi_amortized_benchmark(model)
            
        elif task == "predict_then_optimize":
            ts_model, e2e_model = self.train_pto_models(epochs=epochs)
            self.run_pto_benchmark(ts_model, e2e_model)