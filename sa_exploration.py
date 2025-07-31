import numpy as np
import matplotlib.pyplot as plt
import math

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

def calculate_delta_for_simulated_annealing(solution, objective_function, temperature):
    """
    Calculates the EXPLORATION-EXPLOITATION coefficient (delta) for the SA policy.
    δ = Sum of exploration probabilities / Number of improving moves
    """
    current_score = objective_function(solution)
    neighbors = get_neighbors(solution)
    
    num_improving_moves = 0
    sum_of_exploration_probabilities = 0.0
    
    # Avoid division by zero when temperature is extremely low
    if temperature < 1e-9:
        return 0.0

    for neighbor in neighbors:
        neighbor_score = objective_function(neighbor)
        delta_score = neighbor_score - current_score
        
        if delta_score > 0:
            num_improving_moves += 1
        else:
            # This is an exploration move, add its acceptance probability
            p = math.exp(delta_score / temperature)
            sum_of_exploration_probabilities += p / (1-p) # math.exp(delta_score / temperature)
            
    # Handle the case of being at a local/global optimum where P(Exploit) = 0
    if num_improving_moves == 0:
        # The ratio technically diverges to infinity, indicating pure exploration to escape.
        # We can cap it at a large number for plotting purposes.
        return np.inf
        
    num_neighbors = len(neighbors)
    return sum_of_exploration_probabilities / num_neighbors

# --- 3. Implement the Simulated Annealing Algorithm Run ---
def simulated_annealing_run_for_delta(string_length, max_iterations=1000, initial_temp=10.0, cooling_rate=0.995):
    """
    Performs a single run of SA, tracking the delta coefficient and temperature.
    """
    current_solution = np.random.randint(0, 2, string_length)
    temperature = initial_temp
    
    history = {
        'time': [],
        'delta': [],
        'score': [],
        'temperature': []
    }
    
    for t in range(max_iterations):
        # Calculate coefficients for the current state and temperature
        delta = calculate_delta_for_simulated_annealing(current_solution, objective_function, temperature)
        
        # Record history
        history['time'].append(t)
        history['delta'].append(delta)
        history['score'].append(objective_function(current_solution))
        history['temperature'].append(temperature)

        # Pick a random neighbor
        random_neighbor = get_neighbors(current_solution)[np.random.randint(0, string_length)]
        neighbor_score = objective_function(random_neighbor)
        
        # Decide whether to move
        delta_score = neighbor_score - objective_function(current_solution)
        if delta_score > 0 or np.random.rand() < math.exp(delta_score / temperature):
            current_solution = random_neighbor

        # Cool the temperature
        temperature *= cooling_rate

        # Stop if converged and temperature is negligible
        if temperature < 1e-5 and delta == 0:
            print(f"Convergence at iteration {t}.")
            break
            
    return history

# --- 4. Run the Experiment and Visualize Results ---
if __name__ == '__main__':
    STRING_LENGTH = 30
    
    # Perform a run
    experiment_history = simulated_annealing_run_for_delta(STRING_LENGTH)
    
    # Clean up infinite values for plotting
    delta_values = np.array(experiment_history['delta'])
    delta_values[np.isinf(delta_values)] = np.nan # Replace inf with NaN for plotting
    
    # Create the visualization
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Delta Coefficient on a log scale
    color = 'tab:green'
    ax1.set_xlabel('Time (Iterations)', fontsize=14)
    ax1.set_ylabel('δ (Exploration-Exploitation Coefficient)', color=color, fontsize=14)
    ax1.plot(experiment_history['time'], delta_values, color=color, marker='.', linestyle='-', markersize=5, label='δ Coefficient')
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_yscale('log') # Log scale is crucial to see the dramatic change
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a second y-axis to show the temperature
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Temperature', color=color, fontsize=14)
    ax2.plot(experiment_history['time'], experiment_history['temperature'], color=color, linestyle='--', label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Final touches
    # plt.title(f'δ Coefficient and Temperature for Simulated Annealing (n={STRING_LENGTH})', fontsize=16)
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.show()