import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Optimization Problem (OneMax) ---
def objective_function(solution):
    """Calculates the fitness of a solution (number of 1s)."""
    return np.sum(solution)

# --- 2. Define Neighborhood and Gamma Coefficient Calculation ---
def get_neighbors(solution):
    """Generates all neighbors by flipping one bit at a time."""
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution.copy()
        neighbor[i] = 1 - neighbor[i] # Flip the bit
        neighbors.append(neighbor)
    return neighbors

def calculate_gamma_coefficient(solution, objective_function):
    """
    Calculates the convergence coefficient (gamma) for a given solution.
    γ = |M+| / |M-|
    |M+|: Number of neighbors that improve the objective function.
    |M-|: Number of neighbors that do not improve the objective function.
    """
    current_score = objective_function(solution)
    neighbors = get_neighbors(solution)
    
    improving_moves = 0
    non_improving_moves = 0
    
    for neighbor in neighbors:
        if objective_function(neighbor) > current_score:
            improving_moves += 1
        else:
            non_improving_moves += 1
            
    # Avoid division by zero when no non-improving moves exist
    if non_improving_moves == 0:
        # This is a rare case for OneMax unless at the start,
        # implies all moves are improving. Result is technically infinite.
        return np.inf 
        
    return improving_moves / non_improving_moves

# --- 3. Implement the Hill Climbing Algorithm ---
def hill_climbing_run(string_length, max_iterations=100):
    """
    Performs a single run of the hill-climbing algorithm, tracking
    the gamma coefficient and objective value at each step.
    """
    # Start with a random binary string
    current_solution = np.random.randint(0, 2, string_length)
    current_score = objective_function(current_solution)
    
    # History tracking
    history = {
        'time': [],
        'gamma': [],
        'score': []
    }
    
    for t in range(max_iterations):
        # First, calculate and record metrics for the current state
        gamma = calculate_gamma_coefficient(current_solution, objective_function)
        history['time'].append(t)
        history['gamma'].append(gamma)
        history['score'].append(current_score)
        
        # Find the best neighbor
        neighbors = get_neighbors(current_solution)
        neighbor_scores = [objective_function(n) for n in neighbors]
        best_neighbor_score = np.max(neighbor_scores)
        best_neighbor_index = np.argmax(neighbor_scores)
        
        # Check for convergence (local/global optimum)
        if best_neighbor_score <= current_score:
            print(f"Convergence reached at iteration {t}. Final score: {current_score}")
            # Add final state with gamma=0
            if gamma != 0:
                 history['time'].append(t+1)
                 history['gamma'].append(0)
                 history['score'].append(current_score)
            break
            
        # Move to the best neighbor
        current_solution = neighbors[best_neighbor_index]
        current_score = best_neighbor_score
        
    return history

# --- 4. Run the Experiment and Visualize Results ---
if __name__ == '__main__':
    STRING_LENGTH = 30
    
    # Perform a run
    experiment_history = hill_climbing_run(STRING_LENGTH)
    
    # Create the visualization
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Gamma Coefficient
    color = 'tab:red'
    ax1.set_xlabel('Time (Iterations)', fontsize=14)
    ax1.set_ylabel('γ (Convergence Coefficient)', color=color, fontsize=14)
    ax1.plot(experiment_history['time'], experiment_history['gamma'], color=color, marker='o', linestyle='-', label='γ Coefficient')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Create a second y-axis to show the objective function score
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Objective Score (Number of 1s)', color=color, fontsize=14)
    ax2.plot(experiment_history['time'], experiment_history['score'], color=color, marker='x', linestyle='--', label='Objective Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Final touches
    # plt.title(f'Convergence Coefficient (γ) for Hill Climbing on OneMax (n={STRING_LENGTH})', fontsize=16)
    fig.tight_layout()
    plt.show()
