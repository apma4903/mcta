import numpy as np
from scipy.linalg import null_space
import time
import random

class OptimizationMonitor:
    def __init__(self):
        self.iterations = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.best_density = float('inf')
        
    def print_status(self, temp, curr_density, best_density, accept_rate):
        current_time = time.time()
        if (current_time - self.last_print_time) >= 1.0:
            elapsed = current_time - self.start_time
            print(f"\rTime: {elapsed:.1f}s, Iter: {self.iterations}, Temp: {temp:.2e}, "
                  f"Current: {curr_density:.2f}, Best: {best_density:.2f}, "
                  f"Accept Rate: {accept_rate:.2f}", end="", flush=True)
            self.last_print_time = current_time

def compute_stationary_distribution(P_hat):
    """Compute stationary distribution π for given P_hat"""
    n = P_hat.shape[0]
    
    try:
        A = P_hat.T - np.eye(n)
        ns = null_space(A)
        
        if ns.shape[1] == 0:
            return None
        
        # Try each null space vector until we find a valid one
        for i in range(ns.shape[1]):
            pi = ns[:,i].real
            if np.all(pi >= -1e-10):  # Allow for small numerical errors
                pi = np.maximum(pi, 0)  # Clean up tiny negative values
                pi = pi / np.sum(pi)
                return pi
        return None
    except:
        return None

def calculate_density(pi, V, lengths, lanes, x, road_pairs):
    """Calculate density for each road given stationary distribution"""
    n = len(lengths)
    densities = np.zeros(n)
    
    for i in range(n):
        effective_lanes = lanes[i] * x[i]
        if i % 2 == 0 and x[i+1] == 0:  # Add capacity from reversed opposing road
            effective_lanes += lanes[i+1]
        elif i % 2 == 1 and x[i-1] == 0:
            effective_lanes += lanes[i-1]
            
        if effective_lanes > 0 and pi[i] > 0:
            densities[i] = V * pi[i] / (lengths[i] * effective_lanes)
            
    return densities

def evaluate_solution(x, P, lengths, lanes, V, road_pairs):
    """Evaluate a candidate solution"""
    n = P.shape[0]
    X_d = np.diag(x)
    
    # Compute scaling matrix S_d
    S_d = np.eye(n)
    for i in range(n):
        if x[i] == 1:  # Only scale if road is not reversed
            row_sum = np.sum(P[i] * x)  # Sum of remaining transitions
            if row_sum > 0:
                S_d[i,i] = 1/row_sum
    
    # Modified transition matrix
    P_hat = S_d @ X_d @ P @ X_d
    
    # Get stationary distribution
    pi = compute_stationary_distribution(P_hat)
    
    if pi is None:
        return float('inf'), None, None
    
    # Calculate densities
    densities = calculate_density(pi, V, lengths, lanes, x, road_pairs)
    max_density = np.max(densities)
    
    return max_density, pi, densities

def get_neighbor(x, road_pairs):
    """Generate neighbor by flipping one road pair's directions"""
    x_new = x.copy()
    
    # Pick a random road pair
    i, j = random.choice(road_pairs)
    
    if x_new[i] == 1 and x_new[j] == 1:
        # If both are forward, reverse one randomly
        k = random.choice([i, j])
        x_new[k] = 0
    elif x_new[i] == 0:
        # If i is reversed, unreverse it and maybe reverse j
        x_new[i] = 1
        if random.random() < 0.5:
            x_new[j] = 0
    else:  # x_new[j] == 0
        # If j is reversed, unreverse it and maybe reverse i
        x_new[j] = 1
        if random.random() < 0.5:
            x_new[i] = 0
            
    return x_new

def solve_annealing_problem(lengths, lanes, P, VehiclesCount=1,
                         Objectives=["D", "K", "C", "PI", "E"],
                         FreeFlowSpeeds=[], SkipChecksForSpeed=False):
    """Solve using simulated annealing"""
    
    TOTAL_ITERATIONS = 200  # Hard-coded limit
    
    n_roads = len(lengths)
    road_pairs = [(i, i+1) for i in range(0, n_roads-1, 2)]
    
    # Initial solution - all roads forward
    x_curr = np.ones(n_roads)
    curr_density, curr_pi, curr_densities = evaluate_solution(
        x_curr, P, lengths, lanes, VehiclesCount, road_pairs)
    
    # Track best solution
    x_best = x_curr.copy()
    best_density = curr_density
    best_pi = curr_pi
    best_densities = curr_densities
    
    # Annealing parameters - modified for fixed total iterations
    T = 1.0  # Initial temperature 
    T_min = 1e-5  # Minimum temperature
    total_iterations = 0  # Track total iterations
    iterations_per_temp = 20  # Reduced to allow more temperature steps
    
    # Monitoring
    monitor = OptimizationMonitor()
    accepted = 0
    
    try:
        while T > T_min and total_iterations < TOTAL_ITERATIONS:
            accepted_at_T = 0
            
            for i in range(iterations_per_temp):
                total_iterations += 1
                monitor.iterations += 1
                
                if total_iterations >= TOTAL_ITERATIONS:
                    print("\nReached maximum iterations!")
                    break
                
                # Generate neighbor
                x_new = get_neighbor(x_curr, road_pairs)
                new_density, new_pi, new_densities = evaluate_solution(
                    x_new, P, lengths, lanes, VehiclesCount, road_pairs)
                
                # Accept based on metropolis criterion
                dE = new_density - curr_density
                if dE < 0 or random.random() < np.exp(-dE / T):
                    x_curr = x_new
                    curr_density = new_density
                    curr_pi = new_pi
                    curr_densities = new_densities
                    accepted += 1
                    accepted_at_T += 1
                    
                    # Update best if improved
                    if curr_density < best_density:
                        x_best = x_curr.copy()
                        best_density = curr_density
                        best_pi = curr_pi
                        best_densities = curr_densities
                
                # Print status
                monitor.print_status(T, curr_density, best_density, 
                                   accepted_at_T/iterations_per_temp)
            
            # Cool down faster
            T *= 0.9  # More aggressive cooling
            
            if total_iterations >= TOTAL_ITERATIONS:
                break
                
    except KeyboardInterrupt:
        print(f"\nOptimization stopped after {total_iterations} iterations")
    
    # Prepare results
    results = {
        "Message": "success",
        "RoadReversals": x_best,
        "StationaryProb": best_pi,
        "Density": best_densities
    }
    
    if "K" in Objectives:
        # Compute final P_hat
        X_d = np.diag(x_best)
        S_d = np.eye(n_roads)
        for i in range(n_roads):
            if x_best[i] == 1:
                row_sum = np.sum(P[i] * x_best)
                if row_sum > 0:
                    S_d[i,i] = 1/row_sum
        P_hat = S_d @ X_d @ P @ X_d
        
        # Calculate Kemeny constant
        eigenvals = np.linalg.eigvals(P_hat)
        eigenvals = sorted(eigenvals, reverse=True)
        K = sum(1/(1-x.real) for x in eigenvals[1:] if abs(x) < 0.999999)
        results["KemenyConst"] = K
    
    if "E" in Objectives:
        results["TotalNetworkEmissionCost"] = np.sum(best_densities * lengths)
    
    print(f"\nOptimization complete! Best density: {best_density:.2f}")
    print(f"Total iterations: {total_iterations}")
    print(f"Total time: {time.time() - monitor.start_time:.1f}s")
    print(f"Acceptance rate: {accepted/total_iterations:.2%}")
    
    return results
