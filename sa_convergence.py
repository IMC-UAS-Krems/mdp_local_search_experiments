import numpy as np
import matplotlib.pyplot as plt
import math

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
            
    # Avoid division by zero
    if non_improving_moves == 0:
        return np.inf 
        
    return improving_moves / non_improving_moves

# --- 3. Implement the Simulated Annealing Algorithm ---
def simulated_annealing_run(string_length, max_iterations=1000, initial_temp=10.0, cooling_rate=0.995):
    """
    Performs a single run of the simulated annealing algorithm, tracking
    the gamma coefficient and objective value at each step.
    """
    # Start with a random binary string
    current_solution = np.random.randint(0, 2, string_length)
    current_score = objective_function(current_solution)
    
    temperature = initial_temp
    
    # History tracking
    history = {
        'time': [],
        'gamma': [],
        'score': [],
        'temperature': []
    }
    
    for t in range(max_iterations):
        # Calculate and record metrics for the current state
        gamma = calculate_gamma_coefficient(current_solution, objective_function)
        history['time'].append(t)
        history['gamma'].append(gamma)
        history['score'].append(current_score)
        history['temperature'].append(temperature)
        
        # Check for convergence
        if gamma == 0:
            print(f"Global optimum reached at iteration {t}. Final score: {current_score}")
            break

        # Pick a random neighbor
        neighbors = get_neighbors(current_solution)
        random_neighbor = neighbors[np.random.randint(0, len(neighbors))]
        neighbor_score = objective_function(random_neighbor)
        
        # Calculate the change in 'energy' (fitness)
        # We use f(new) - f(current) because we are maximizing
        delta_score = neighbor_score - current_score
        
        # Decide whether to move to the new solution
        if delta_score > 0:
            # Always accept a better solution
            current_solution = random_neighbor
            current_score = neighbor_score
        else:
            # Accept a worse solution with a certain probability
            # Avoid math domain error with large negative delta_score
            if temperature > 1e-6:
                acceptance_probability = math.exp(delta_score / temperature)
                if np.random.rand() < acceptance_probability:
                    current_solution = random_neighbor
                    current_score = neighbor_score
        
        # Cool the temperature
        temperature *= cooling_rate
        
    return history

# --- 4. Run the Experiment and Visualize Results ---
if __name__ == '__main__':
    STRING_LENGTH = 30
    
    # Perform a run
    experiment_history = simulated_annealing_run(STRING_LENGTH)
    
    # Create the visualization
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Gamma Coefficient
    color = 'tab:red'
    ax1.set_xlabel('Time (Iterations)', fontsize=14)
    ax1.set_ylabel('γ (Convergence Coefficient)', color=color, fontsize=14)
    ax1.plot(experiment_history['time'], experiment_history['gamma'], color=color, marker='o', linestyle='-', markersize=4, label='γ Coefficient')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Create a second y-axis to show the objective function score
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Objective Score', color=color, fontsize=14)
    ax2.plot(experiment_history['time'], experiment_history['score'], color=color, alpha=0.6, linestyle='--', label='Objective Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Final touches
    # plt.title(f'Convergence Coefficient (γ) for Simulated Annealing on OneMax (n={STRING_LENGTH})', fontsize=16)
    fig.tight_layout()
    plt.show()