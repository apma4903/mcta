import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space
import time


class OptimizationMonitor:
    def __init__(self):
        self.objective_evals = 0
        self.constraint_evals = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time

    def print_status(self, max_density=None, penalty=None, force=False):
        current_time = time.time()
        # Print every 2 seconds or if forced
        if force or (current_time - self.last_print_time) >= 2.0:
            elapsed = current_time - self.start_time
            status = f"\rTime: {elapsed:.1f}s, Obj evals: {self.objective_evals}, Const evals: {self.constraint_evals}"
            if max_density is not None:
                status += f", Density: {max_density:.2f}"
            if penalty is not None:
                status += f", Penalty: {penalty:.2f}"
            print(status, end="", flush=True)
            self.last_print_time = current_time


monitor = OptimizationMonitor()


def compute_modified_transition_matrix(x, P, road_pairs):
    """Compute P_hat given decision variables x"""
    n = P.shape[0]
    X_d = np.diag(x)

    # Compute scaling matrix S_d
    S_d = np.eye(n)
    for i in range(n):
        if x[i] == 1:  # Only scale if road is not reversed
            row_sum = np.sum(P[i] * x)  # Sum of remaining transitions
            if row_sum > 0:
                S_d[i, i] = 1 / row_sum

    # Modified transition matrix
    P_hat = S_d @ X_d @ P @ X_d
    return P_hat


def compute_stationary_distribution(P_hat, debug=False):
    """Compute stationary distribution π for given P_hat"""
    n = P_hat.shape[0]

    if debug:
        print(f"\nP_hat properties:")
        print(f"Shape: {P_hat.shape}")
        print(
            f"Min value: {P_hat.data.min() if hasattr(P_hat, 'data') else P_hat.min()}"
        )
        print(
            f"Max value: {P_hat.data.max() if hasattr(P_hat, 'data') else P_hat.max()}"
        )
        print(
            f"Number of nonzeros: {P_hat.nnz if hasattr(P_hat, 'nnz') else np.count_nonzero(P_hat)}"
        )

    try:
        # Find nullspace of (P_hat.T - I)
        A = P_hat.T - np.eye(n)
        ns = null_space(A)

        if ns.shape[1] == 0:
            if debug:
                print("No null space found")
            return None

        # Try each null space vector until we find a valid one
        for i in range(ns.shape[1]):
            pi = ns[:, i].real
            if np.all(pi >= -1e-10):  # Allow for small numerical errors
                pi = np.maximum(pi, 0)  # Clean up tiny negative values
                pi = pi / np.sum(pi)
                return pi

        if debug:
            print("No valid stationary distribution found in null space")
        return None

    except Exception as e:
        if debug:
            print(f"Error in compute_stationary_distribution: {str(e)}")
        return None


def calculate_density(pi, V, lengths, lanes, x, road_pairs):
    """Calculate density for each road given stationary distribution"""
    n = len(lengths)
    densities = np.zeros(n)

    for i in range(n):
        effective_lanes = lanes[i] * x[i]
        if i % 2 == 0 and x[i + 1] == 0:  # Add capacity from reversed opposing road
            effective_lanes += lanes[i + 1]
        elif i % 2 == 1 and x[i - 1] == 0:
            effective_lanes += lanes[i - 1]

        if effective_lanes > 0 and pi[i] > 0:
            densities[i] = V * pi[i] / (lengths[i] * effective_lanes)

    return densities


def road_pair_constraint(x, road_pairs):
    """No simultaneous reversal constraint"""
    monitor.constraint_evals += 1
    violations = 0
    for i, j in road_pairs:
        if x[i] + x[j] < 1:
            violations += 1
    return -violations


def stochastic_matrix_constraint(x, P, road_pairs):
    """P_hat must be a valid stochastic matrix for active roads"""
    monitor.constraint_evals += 1
    P_hat = compute_modified_transition_matrix(x, P, road_pairs)
    row_sums = np.sum(P_hat, axis=1)

    # Row sums should be 1 for active roads (x[i]=1) and 0 for reversed roads (x[i]=0)
    return np.sum(np.abs(row_sums - x))


def stationary_distribution_constraint(x, P, road_pairs):
    """Must have valid stationary distribution"""
    monitor.constraint_evals += 1
    P_hat = compute_modified_transition_matrix(x, P, road_pairs)
    pi = compute_stationary_distribution(P_hat)

    if pi is None:
        return -1  # No valid stationary distribution exists

    # Check π * P_hat = π
    error = np.sum(np.abs(np.dot(pi, P_hat) - pi))
    return -error


def solve_SLSQP_problem(
    lengths,
    lanes,
    P,
    VehiclesCount=1,
    Objectives=["D", "K", "C", "PI", "E"],
    FreeFlowSpeeds=[],
    SkipChecksForSpeed=False,
):
    """Solve the road network optimization problem using SLSQP"""

    global monitor
    monitor = OptimizationMonitor()  # Reset monitor

    n_roads = len(lengths)
    results = {}

    print(f"\nStarting optimization with {n_roads} roads...")

    # Create road pairs list
    road_pairs = [(i, i + 1) for i in range(0, n_roads - 1, 2)]

    def objective(x):
        """Minimize maximum density"""
        monitor.objective_evals += 1

        P_hat = compute_modified_transition_matrix(x, P, road_pairs)
        pi = compute_stationary_distribution(P_hat)

        if pi is None:
            penalty = 1e6
            monitor.print_status(max_density=np.inf, penalty=penalty)
            return penalty

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

        monitor.print_status(max_density=max_density, penalty=penalty)
        return max_density + penalty

    # Initial guess - all roads open
    x0 = np.ones(n_roads)

    # Bounds and constraints
    bounds = [(0, 1) for _ in range(n_roads)]
    constraints = [
        {"type": "ineq", "fun": road_pair_constraint, "args": (road_pairs,)},
        {
            "type": "eq",
            "fun": stochastic_matrix_constraint,
            "args": (
                P,
                road_pairs,
            ),
        },
        {
            "type": "eq",
            "fun": stationary_distribution_constraint,
            "args": (
                P,
                road_pairs,
            ),
        },
    ]

    # Solve optimization problem
    try:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-6, "disp": True},
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user!")
        return None

    print("\nOptimization complete!")
    print(f"Total time: {time.time() - monitor.start_time:.1f}s")
    print(
        f"Total evaluations - Objective: {monitor.objective_evals}, Constraints: {monitor.constraint_evals}"
    )
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    # Process results
    if result.success:
        # Round solution to binary
        final_dirs = np.round(result.x).astype(int)

        # Calculate final metrics
        P_hat = compute_modified_transition_matrix(final_dirs, P, road_pairs)
        pi = compute_stationary_distribution(P_hat)
        densities = calculate_density(
            pi, VehiclesCount, lengths, lanes, final_dirs, road_pairs
        )

        results["Message"] = "success"
        results["RoadReversals"] = final_dirs
        results["StationaryProb"] = pi
        results["Density"] = densities

        # Calculate Kemeny constant if requested
        if "K" in Objectives:
            eigenvals = np.linalg.eigvals(P_hat)
            eigenvals = sorted(eigenvals, reverse=True)
            K = sum(1 / (1 - x.real) for x in eigenvals[1:] if abs(x) < 0.999999)
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
