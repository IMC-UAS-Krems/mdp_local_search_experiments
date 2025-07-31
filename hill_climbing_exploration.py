import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Optimization Problem (OneMax) ---
def objective_function(solution):
    """Calculates the fitness of a solution (number of 1s)."""
    return np.sum(solution)

# --- 2. Define Neighborhood and Coefficient Calculations ---
def get_neighbors(solution):
    """Generates all neighbors by flipping one bit at a time."""
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution.copy()
        neighbor[i] = 1 - neighbor[i] # Flip the bit
        neighbors.append(neighbor)
    return neighbors

def calculate_gamma_coefficient(solution, objective_function):
    """Calculates the CONVERGENCE coefficient (gamma) for a given state."""
    current_score = objective_function(solution)
    improving_moves = 0
    non_improving_moves = 0
    
    for neighbor in get_neighbors(solution):
        if objective_function(neighbor) > current_score:
            improving_moves += 1
        else:
            non_improving_moves += 1
            
    if non_improving_moves == 0: return np.inf
    return improving_moves / non_improving_moves

def calculate_delta_for_hill_climbing(solution, objective_function):
    """
    Calculates the EXPLORATION-EXPLOITATION coefficient (delta) for the Hill Climbing policy.
    
    For Hill Climbing, the policy is to always choose an improving move if one exists.
    Therefore, P(choose exploration) is 0 unless at a local optimum.
    This makes the algorithm purely exploitation-oriented.
    """
    current_score = objective_function(solution)
    has_improving_move = any(objective_function(n) > current_score for n in get_neighbors(solution))
    
    if has_improving_move:
        # P(choose explore) = 0, P(choose exploit) = 1.
        # delta = 0 / 1 = 0
        return 0.0
    else:
        # At a local optimum. No exploitation moves are possible.
        # P(choose exploit) = 0. The algorithm terminates.
        # Following the paper's classification, delta remains 0.
        return 0.0

# --- 3. Implement the Hill Climbing Algorithm Run ---
def hill_climbing_run_for_delta(string_length, max_iterations=100):
    """
    Performs a single run of hill-climbing, tracking both delta and gamma
    coefficients and the objective value.
    """
    current_solution = np.random.randint(0, 2, string_length)
    
    history = {
        'time': [],
        'delta': [],
        'gamma': [],
        'score': []
    }
    
    for t in range(max_iterations):
        current_score = objective_function(current_solution)
        
        # Calculate coefficients for the current state
        delta = calculate_delta_for_hill_climbing(current_solution, objective_function)
        gamma = calculate_gamma_coefficient(current_solution, objective_function)
        
        # Record history
        history['time'].append(t)
        history['delta'].append(delta)
        history['gamma'].append(gamma)
        history['score'].append(current_score)
        
        # Find the best neighbor
        neighbors = get_neighbors(current_solution)
        neighbor_scores = [objective_function(n) for n in neighbors]
        best_neighbor_score = np.max(neighbor_scores)
        
        # Check for convergence (local/global optimum)
        if best_neighbor_score <= current_score:
            print(f"Convergence reached at iteration {t}. Final score: {current_score}")
            # Add final state if not already there
            if history['gamma'][-1] != 0:
                 history['time'].append(t+1)
                 history['delta'].append(0)
                 history['gamma'].append(0)
                 history['score'].append(current_score)
            break
            
        # Move to the best neighbor
        best_neighbor_index = np.argmax(neighbor_scores)
        current_solution = neighbors[best_neighbor_index]
        
    return history

# --- 4. Run the Experiment and Visualize Results ---
if __name__ == '__main__':
    STRING_LENGTH = 30
    
    # Perform a run
    experiment_history = hill_climbing_run_for_delta(STRING_LENGTH)
    
    # Create the visualization
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Delta Coefficient
    color = 'tab:green'
    ax1.set_xlabel('Time (Iterations)', fontsize=14)
    ax1.set_ylabel('δ (Exploration-Exploitation Coefficient)', color=color, fontsize=14)
    ax1.plot(experiment_history['time'], experiment_history['delta'], color=color, marker='s', linestyle='-', markersize=6, label='δ Coefficient')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.1, 1.1) # Set y-axis for delta for clarity

    # Create a second y-axis to show the gamma coefficient
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('γ (Convergence Coefficient)', color=color, fontsize=14)
    ax2.plot(experiment_history['time'], experiment_history['gamma'], color=color, marker='o', linestyle='--', alpha=0.7, label='γ Coefficient')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.1, 1.1) # Set y-axis for delta for clarity

    # Create a third y-axis to show the objective function score
    # ax3 = ax2.twinx()
    # color = 'tab:blue'
    # ax3.set_ylabel('Objective Score (Number of 1s)', color=color, fontsize=14)
    # ax3.plot(experiment_history['time'], experiment_history['score'], color=color, marker='x', linestyle='--', label='Objective Score')
    # ax3.tick_params(axis='y', labelcolor=color)
    
    # Final touches
    # plt.title(f'δ and γ Coefficients for Hill Climbing on OneMax (n={STRING_LENGTH})', fontsize=16)
    fig.tight_layout()
    # Add a single legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.show()