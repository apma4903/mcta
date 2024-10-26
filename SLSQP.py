import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space

def compute_modified_transition_matrix(x, P, road_pairs):
    """Compute P_hat given decision variables x"""
    #n = len(P)
    # P is sparse
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
    return P_hat

def compute_stationary_distribution(P_hat):
    """Compute stationary distribution π for given P_hat"""
    #n = len(P_hat)
    # P_hat is sparse
    n = P_hat.shape[0]
    
    # Find nullspace of (P_hat.T - I)
    A = (P_hat.T - np.eye(n))
    ns = null_space(A)
    
    if ns.shape[1] == 0:
        return None
        
    # Extract π from nullspace and normalize
    pi = ns[:,0].real
    if np.all(pi >= 0):  # Ensure non-negative
        pi = pi / np.sum(pi)
        return pi
    
    # Try other null space vectors if first one wasn't valid
    for i in range(1, ns.shape[1]):
        pi = ns[:,i].real
        if np.all(pi >= 0):
            pi = pi / np.sum(pi)
            return pi
            
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

def road_pair_constraint(x, road_pairs):
    """No simultaneous reversal constraint"""
    violations = 0
    for i, j in road_pairs:
        if x[i] + x[j] < 1:
            violations += 1
    return -violations

def stochastic_matrix_constraint(x, P, road_pairs):
    """P_hat must be a valid stochastic matrix for active roads"""
    P_hat = compute_modified_transition_matrix(x, P, road_pairs)
    row_sums = np.sum(P_hat, axis=1)
    
    # Row sums should be 1 for active roads (x[i]=1) and 0 for reversed roads (x[i]=0)
    return np.sum(np.abs(row_sums - x))

def stationary_distribution_constraint(x, P, road_pairs):
    """Must have valid stationary distribution"""
    P_hat = compute_modified_transition_matrix(x, P, road_pairs)
    pi = compute_stationary_distribution(P_hat)
    
    if pi is None:
        return -1  # No valid stationary distribution exists
    
    # Check π * P_hat = π
    error = np.sum(np.abs(np.dot(pi, P_hat) - pi))
    return -error

def solve_SLSQP_problem(lengths, lanes, P, VehiclesCount=1, 
                      Objectives=["D","K","C","PI","E"],
                      FreeFlowSpeeds=[], SkipChecksForSpeed=False):
    """Solve the road network optimization problem using continuous optimization"""
    
    n_roads = len(lengths)
    results = {}
    
    # Create road pairs list
    road_pairs = [(i, i+1) for i in range(0, n_roads-1, 2)]
    
    def objective(x):
        """Minimize maximum density"""
        P_hat = compute_modified_transition_matrix(x, P, road_pairs)
        pi = compute_stationary_distribution(P_hat)
        
        if pi is None:
            return 1e6  # Penalty for invalid solutions
            
        densities = calculate_density(pi, VehiclesCount, lengths, lanes, x, road_pairs)
        max_density = np.max(densities)
        
        # Add penalties for constraint violations
        penalty = 0
        
        # Stochastic matrix penalty
        stoch_violation = stochastic_matrix_constraint(x, P, road_pairs)
        if stoch_violation > 1e-6:
            penalty += 1e4 * stoch_violation
            
        # Stationary distribution penalty
        stat_violation = stationary_distribution_constraint(x, P, road_pairs)
        if stat_violation < -1e-6:
            penalty += 1e4 * abs(stat_violation)
        
        return max_density + penalty

    # Initial guess - all roads open
    x0 = np.ones(n_roads)
    
    # Bounds and constraints
    bounds = [(0, 1) for _ in range(n_roads)]
    
    constraints = [
        {'type': 'ineq', 'fun': road_pair_constraint, 'args': (road_pairs,)},
        {'type': 'eq', 'fun': stochastic_matrix_constraint, 'args': (P, road_pairs,)},
        {'type': 'eq', 'fun': stationary_distribution_constraint, 'args': (P, road_pairs,)}
    ]

    # Solve optimization problem
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
    )

    # Process results
    if result.success:
        # Round solution to binary
        final_dirs = np.round(result.x).astype(int)
        
        # Calculate final metrics
        P_hat = compute_modified_transition_matrix(final_dirs, P, road_pairs)
        pi = compute_stationary_distribution(P_hat)
        densities = calculate_density(pi, VehiclesCount, lengths, lanes, final_dirs, road_pairs)
        
        results["Message"] = "success"
        results["RoadReversals"] = final_dirs
        results["StationaryProb"] = pi
        results["Density"] = densities
        
        # Calculate Kemeny constant if requested
        if "K" in Objectives:
            eigenvals = np.linalg.eigvals(P_hat)
            eigenvals = sorted(eigenvals, reverse=True)
            K = sum(1/(1-x.real) for x in eigenvals[1:] if abs(x) < 0.999999)
            results["KemenyConst"] = K
        
        # Calculate total network emission cost
        if "E" in Objectives:
            results["TotalNetworkEmissionCost"] = np.sum(densities * lengths)
            
    else:
        # Fallback values if optimization fails
        results["Message"] = "optimization failed"
        results["RoadReversals"] = np.ones(n_roads)
        results["StationaryProb"] = np.zeros(n_roads)
        results["Density"] = np.zeros(n_roads)
        results["KemenyConst"] = 0
        results["TotalNetworkEmissionCost"] = 0.0

    return results
