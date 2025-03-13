import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from functools import partial

# Constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_ROUNDS = 400
MEMORY_SIZE = 3

# Payoff matrix for Prisoner's Dilemma
PAYOFF_MATRIX = {
    ('C', 'C'): (3, 3),  # Both cooperate
    ('C', 'D'): (0, 5),  # Row cooperates, column defects
    ('D', 'C'): (5, 0),  # Row defects, column cooperates
    ('D', 'D'): (1, 1)  # Both defect
}


# Fixed strategies (same as before)
def always_cooperate(history, opponent_history):
    return 'C'


def always_defect(history, opponent_history):
    return 'D'


def tit_for_tat(history, opponent_history):
    if not opponent_history:
        return 'C'
    return opponent_history[-1]

def tit_for_two_tats(history, opponent_history):
    if len(opponent_history) < 2:
        return 'C'
    if opponent_history[-1] == 'D' and opponent_history[-2] == 'D':
        return 'D'
    return 'C'


def pavlov(history, opponent_history):
    if not history or not opponent_history:
        return 'C'

    my_last_move = history[-1]
    opponent_last_move = opponent_history[-1]
    my_payoff = PAYOFF_MATRIX[(my_last_move, opponent_last_move)][0]

    if my_payoff > 1:
        return my_last_move
    else:
        return 'D' if my_last_move == 'C' else 'C'


# Dictionary of fixed strategies
FIXED_STRATEGIES = {
    'AlwaysC': always_cooperate,
    'AlwaysD': always_defect,
    'TitForTat': tit_for_tat,
    'TitForTwoTats': tit_for_two_tats,
    'Pavlov': pavlov
}


# Strategy creation functions (same as before)
def create_strategy_table():
    table = {}
    for i in range(2 ** (2 * MEMORY_SIZE)):
        history_key = format(i, f'0{2 * MEMORY_SIZE}b')
        my_history = tuple(('C' if h == '0' else 'D') for h in history_key[:MEMORY_SIZE])
        opp_history = tuple(('C' if h == '0' else 'D') for h in history_key[MEMORY_SIZE:])
        table[(my_history, opp_history)] = 'C' if random.random() < 0.5 else 'D'
    return table


def lookup_table_strategy(lookup_table, history, opponent_history):
    if len(history) < MEMORY_SIZE:
        padded_history = ('C',) * (MEMORY_SIZE - len(history)) + tuple(history[-MEMORY_SIZE:] if history else ())
        padded_opp_history = ('C',) * (MEMORY_SIZE - len(opponent_history)) + tuple(
            opponent_history[-MEMORY_SIZE:] if opponent_history else ())
    else:
        padded_history = tuple(history[-MEMORY_SIZE:])
        padded_opp_history = tuple(opponent_history[-MEMORY_SIZE:])

    return lookup_table[(padded_history, padded_opp_history)]

def apply_noise(move, noise_level):
    #Apply noise to a move by randomly flipping it.
    if random.random() < noise_level:
        return 'D' if move == 'C' else 'C'
    return move


def play_game_with_noise(strategy1, strategy2, noise_level=0.05, num_rounds=NUM_ROUNDS):
    history1 = []  # Actual move history for player 1
    history2 = []  # Actual move history for player 2
    perceived_history1 = []  # What player 2 perceives of player 1's moves
    perceived_history2 = []  # What player 1 perceives of player 2's moves

    score1 = 0
    score2 = 0

    for _ in range(num_rounds):
        # Players decide based on what they perceive, not necessarily what actually happened
        if isinstance(strategy1, dict):
            move1 = lookup_table_strategy(strategy1, history1, perceived_history2)
        else:
            move1 = strategy1(history1, perceived_history2)

        if isinstance(strategy2, dict):
            move2 = lookup_table_strategy(strategy2, history2, perceived_history1)
        else:
            move2 = strategy2(history2, perceived_history1)

        # Execute the moves with possible execution errors
        actual_move1 = apply_noise(move1, noise_level)
        actual_move2 = apply_noise(move2, noise_level)

        # Calculate scores based on what actually happened
        payoff = PAYOFF_MATRIX[(actual_move1, actual_move2)]
        score1 += payoff[0]
        score2 += payoff[1]

        # Update actual histories
        history1.append(actual_move1)
        history2.append(actual_move2)

        # Update perceived histories with possible perception errors
        perceived_move1 = apply_noise(actual_move1, noise_level)
        perceived_move2 = apply_noise(actual_move2, noise_level)

        perceived_history1.append(perceived_move1)
        perceived_history2.append(perceived_move2)

    return score1, score2


def calculate_fitness_with_noise(strategy, noise_level=0.05):
    #Calculate fitness under noisy conditions
    total_score = 0

    for fixed_strategy_name, fixed_strategy_func in FIXED_STRATEGIES.items():
        score, _ = play_game_with_noise(
            lambda h1, h2: lookup_table_strategy(strategy, h1, h2),
            fixed_strategy_func,
            noise_level=noise_level
        )
        total_score += score

    return total_score


def run_genetic_algorithm_with_noise(noise_level=0.05):
    #Run the genetic algorithm with noise
    population = [create_strategy_table() for _ in range(POPULATION_SIZE)]

    avg_fitness_history = []
    max_fitness_history = []
    best_strategy = None
    best_fitness = 0

    # Create a fitness function with the specified noise level
    fitness_func = partial(calculate_fitness_with_noise, noise_level=noise_level)

    for generation in (range(NUM_GENERATIONS)):
        # Evaluate fitness
        fitnesses = [fitness_func(strategy) for strategy in population]

        # Track statistics
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        avg_fitness_history.append(avg_fitness)
        max_fitness_history.append(max_fitness)

        # Store the best strategy
        best_idx = fitnesses.index(max_fitness)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_strategy = population[best_idx].copy()

        # Create new population
        new_population = []

        # Elitism: keep the best strategy
        new_population.append(population[best_idx].copy())

        # Create the rest of the population
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child)

            # Add to new population
            new_population.append(child)

        # Replace old population
        population = new_population

        # Print progress
        if (generation + 1) % 10 == 0:
            print(f"Generation {generation + 1}: Avg Fitness = {avg_fitness:.2f}, Max Fitness = {max_fitness:.2f}")

    return best_strategy, avg_fitness_history, max_fitness_history


def analyze_noise_impact():
    #Analyze the impact of different noise levels on strategy effectiveness
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    best_strategies = []
    fitness_histories = []

    for noise_level in noise_levels:
        print(f"\nRunning with noise level: {noise_level}")
        best_strategy, avg_fitness_history, max_fitness_history = run_genetic_algorithm_with_noise(noise_level)
        best_strategies.append(best_strategy)
        fitness_histories.append((avg_fitness_history, max_fitness_history))

    # Compare strategies
    print("\nComparing strategies evolved under different noise levels:")

    # Create a matrix to store cross-comparison results
    comparison_matrix = np.zeros((len(noise_levels), len(noise_levels)))

    # Cross-evaluate each strategy against each noise level
    for i, evolved_strategy in enumerate(best_strategies):
        for j, test_noise in enumerate(noise_levels):
            fitness = calculate_fitness_with_noise(evolved_strategy, noise_level=test_noise)
            comparison_matrix[i, j] = fitness

    # Visualize the comparison matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(comparison_matrix, annot=True, fmt=".1f",
                xticklabels=[f"Noise {n}" for n in noise_levels],
                yticklabels=[f"Evolved at {n}" for n in noise_levels],
                cmap="YlGnBu")
    plt.xlabel("Test Noise Level")
    plt.ylabel("Training Noise Level")
    plt.title("Performance of Strategies Evolved at Different Noise Levels")
    plt.tight_layout()
    plt.show()

    # Plot fitness histories
    plt.figure(figsize=(12, 8))
    for i, (avg_history, max_history) in enumerate(fitness_histories):
        plt.plot(max_history, label=f"Noise {noise_levels[i]}")

    plt.xlabel("Generation")
    plt.ylabel("Max Fitness")
    plt.title("Fitness Progression at Different Noise Levels")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_strategies, fitness_histories




def calculate_multi_objective_fitness(strategy):
    # Calculate multiple fitness objectives:
    # 1. Total score (primary objective)
    # 2. Stability (consistency of strategy)
    # 3. Robustness (performance variation across opponents)
    scores = []
    stability_values = []

    for fixed_strategy_name, fixed_strategy_func in FIXED_STRATEGIES.items():
        # Play the game
        history1 = []
        history2 = []

        for _ in range(NUM_ROUNDS):
            move1 = lookup_table_strategy(strategy, history1, history2)
            move2 = fixed_strategy_func(history2, history1)

            history1.append(move1)
            history2.append(move2)

        # Calculate score
        score, _ = play_game(lambda h1, h2: lookup_table_strategy(strategy, h1, h2), fixed_strategy_func)
        scores.append(score)

        # Calculate stability (lack of transitions between C and D)
        transitions = sum(1 for i in range(1, len(history1)) if history1[i] != history1[i - 1])
        stability = 1 - (transitions / (NUM_ROUNDS - 1))  # Higher is more stable
        stability_values.append(stability)

    # Calculate objectives
    total_score = sum(scores)
    avg_stability = sum(stability_values) / len(stability_values)
    score_variance = np.var(scores)  # Lower variance means more robust

    # Normalize robustness (invert variance so higher is better)
    max_variance = NUM_ROUNDS * NUM_ROUNDS * len(FIXED_STRATEGIES)  # Theoretical max variance
    robustness = 1 - (score_variance / max_variance)

    return (total_score, avg_stability, robustness)


def play_game(strategy1, strategy2, num_rounds=NUM_ROUNDS):
    #Play a standard IPD game without noise
    history1 = []
    history2 = []
    score1 = 0
    score2 = 0

    for _ in range(num_rounds):
        move1 = strategy1(history1, history2)
        move2 = strategy2(history2, history1)

        payoff = PAYOFF_MATRIX[(move1, move2)]
        score1 += payoff[0]
        score2 += payoff[1]

        history1.append(move1)
        history2.append(move2)

    return score1, score2


def tournament_selection(population, fitnesses, tournament_size):
    #Select a strategy using tournament selection
    tournament_indices = random.sample(range(len(population)), tournament_size)

    best_idx = tournament_indices[0]
    best_fitness = fitnesses[best_idx]

    for idx in tournament_indices[1:]:
        if fitnesses[idx] > best_fitness:
            best_idx = idx
            best_fitness = fitnesses[idx]

    return population[best_idx]


def tournament_selection_multi_objective(population, fitness_list, tournament_size):
    #Tournament selection for multi-objective optimisation using Pareto dominance
    tournament_indices = random.sample(range(len(population)), tournament_size)

    # Randomly select the first candidate as initial winner
    winner_idx = tournament_indices[0]

    for idx in tournament_indices[1:]:
        if pareto_dominates(fitness_list[idx], fitness_list[winner_idx]):
            winner_idx = idx
        elif not pareto_dominates(fitness_list[winner_idx], fitness_list[idx]):
            # Neither dominates the other, randomly choose one with 50% probability
            if random.random() < 0.5:
                winner_idx = idx

    return population[winner_idx]


def pareto_dominates(fitness_a, fitness_b):

    # Returns True if fitness_a Pareto-dominates fitness_b

    better_in_at_least_one = False
    for a, b in zip(fitness_a, fitness_b):
        if a < b:  # If a is worse in any objective, it doesn't dominate
            return False
        if a > b:  # If a is better in any objective, note it
            better_in_at_least_one = True

    return better_in_at_least_one


def crossover(parent1, parent2):
    #Perform crossover between two parent strategies
    if random.random() > CROSSOVER_RATE:
        return parent1.copy()

    child = {}
    keys = list(parent1.keys())
    crossover_point = random.randint(1, len(keys) - 1)

    for i, key in enumerate(keys):
        if i < crossover_point:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]

    return child


def mutate(strategy):
    #Mutate a strategy by randomly flipping some moves
    mutated_strategy = strategy.copy()

    for key in mutated_strategy:
        if random.random() < MUTATION_RATE:
            mutated_strategy[key] = 'D' if mutated_strategy[key] == 'C' else 'C'

    return mutated_strategy


def run_multi_objective_ga():
    #Run the genetic algorithm with multi-objective optimization

    # Initialize population
    population = [create_strategy_table() for _ in range(POPULATION_SIZE)]

    # Track fitness history for each objective
    history_score = []
    history_stability = []
    history_robustness = []

    # Track Pareto front
    pareto_front = []

    # Main evolutionary loop
    for generation in (range(NUM_GENERATIONS)):
        # Evaluate fitness for all objectives
        fitness_list = [calculate_multi_objective_fitness(strategy) for strategy in population]

        # Track statistics
        avg_score = sum(f[0] for f in fitness_list) / len(fitness_list)
        avg_stability = sum(f[1] for f in fitness_list) / len(fitness_list)
        avg_robustness = sum(f[2] for f in fitness_list) / len(fitness_list)

        history_score.append(avg_score)
        history_stability.append(avg_stability)
        history_robustness.append(avg_robustness)

        # Update Pareto front
        pareto_front = []
        for i, fitness in enumerate(fitness_list):
            is_dominated = False
            for other_fitness in fitness_list:
                if pareto_dominates(other_fitness, fitness) and other_fitness != fitness:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append((population[i].copy(), fitness))

        # Create new population
        new_population = []

        # Add some members of the Pareto front (elitism)
        num_elite = min(10, len(pareto_front))
        elite_indices = random.sample(range(len(pareto_front)), num_elite)
        for idx in elite_indices:
            new_population.append(pareto_front[idx][0].copy())

        # Create the rest of the population
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection based on Pareto dominance
            parent1 = tournament_selection_multi_objective(population, fitness_list, TOURNAMENT_SIZE)
            parent2 = tournament_selection_multi_objective(population, fitness_list, TOURNAMENT_SIZE)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child)

            # Add to new population
            new_population.append(child)

        # Replace old population
        population = new_population

        # Print progress
        if (generation + 1) % 10 == 0:
            print(f"Generation {generation + 1}: Avg Score = {avg_score:.2f}, " +
                  f"Avg Stability = {avg_stability:.2f}, Avg Robustness = {avg_robustness:.2f}")

    return pareto_front, history_score, history_stability, history_robustness


def visualize_pareto_front(pareto_front):
    #Visualize the Pareto front in 3D space
    if not pareto_front:
        print("No Pareto front to visualize.")
        return

    # Extract fitness values
    scores = [p[1][0] for p in pareto_front]
    stabilities = [p[1][1] for p in pareto_front]
    robustness = [p[1][2] for p in pareto_front]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot
    scatter = ax.scatter(scores, stabilities, robustness, c=scores, cmap='viridis', s=100, alpha=0.6)

    # Add labels
    ax.set_xlabel('Total Score')
    ax.set_ylabel('Stability')
    ax.set_zlabel('Robustness')
    ax.set_title('Pareto Front of Evolved Strategies')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Score')

    plt.tight_layout()
    plt.show()


def visualize_objective_histories(history_score, history_stability, history_robustness):
    #Visualize the progression of each objective over generations
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(history_score)
    plt.ylabel('Avg Score')
    plt.title('Multi-Objective Optimization Progress')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(history_stability)
    plt.ylabel('Avg Stability')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(history_robustness)
    plt.xlabel('Generation')
    plt.ylabel('Avg Robustness')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_pareto_strategies(pareto_front):
    # Analyze the strategies in the Pareto front
    if not pareto_front:
        print("No Pareto front to analyze.")
        return

    # Select diverse strategies to analyze
    scores = [p[1][0] for p in pareto_front]
    stabilities = [p[1][1] for p in pareto_front]

    # Find strategies with interesting properties
    max_score_idx = scores.index(max(scores))
    max_stability_idx = stabilities.index(max(stabilities))

    # Find a balanced strategy (closest to normalized centroid)
    normalized_scores = [(s - min(scores)) / (max(scores) - min(scores)) for s in scores]
    normalized_stabilities = [(s - min(stabilities)) / (max(stabilities) - min(stabilities)) for s in stabilities]

    distances = [(normalized_scores[i] - 0.5) ** 2 + (normalized_stabilities[i] - 0.5) ** 2
                 for i in range(len(pareto_front))]
    balanced_idx = distances.index(min(distances))

    strategies_to_analyze = {
        "Highest Score": pareto_front[max_score_idx][0],
        "Highest Stability": pareto_front[max_stability_idx][0],
        "Balanced": pareto_front[balanced_idx][0]
    }

    # Analyze each selected strategy
    for name, strategy in strategies_to_analyze.items():
        print(f"\nAnalyzing {name} strategy:")

        # Get fitness values
        fitness = calculate_multi_objective_fitness(strategy)
        print(f"  Score: {fitness[0]:.2f}")
        print(f"  Stability: {fitness[1]:.2f}")
        print(f"  Robustness: {fitness[2]:.2f}")

        # Analyze cooperation rates
        results = {}
        for fixed_strategy_name, fixed_strategy_func in FIXED_STRATEGIES.items():
            history1 = []
            history2 = []

            for _ in range(NUM_ROUNDS):
                move1 = lookup_table_strategy(strategy, history1, history2)
                move2 = fixed_strategy_func(history2, history1)

                history1.append(move1)
                history2.append(move2)

            coop_rate = history1.count('C') / len(history1)
            results[fixed_strategy_name] = coop_rate

        print("  Cooperation rates:")
        for opponent, rate in results.items():
            print(f"    vs {opponent}: {rate:.2f}")


def noise_and_multi_objective_demo():
    # Run a demonstration of both noise and multi-objective optimization3
    # 1. First, analyze the impact of noise
    print("=== ANALYZING IMPACT OF NOISE ===")
    best_noise_strategies, noise_fitness_histories = analyze_noise_impact()

    # 2. Then, run multi-objective optimization
    print("\n=== RUNNING MULTI-OBJECTIVE OPTIMIZATION ===")
    pareto_front, history_score, history_stability, history_robustness = run_multi_objective_ga()

    # 3. Visualize results
    print("\n=== VISUALIZING MULTI-OBJECTIVE RESULTS ===")
    visualize_pareto_front(pareto_front)
    visualize_objective_histories(history_score, history_stability, history_robustness)

    # 4. Analyze Pareto front strategies
    print("\n=== ANALYZING PARETO FRONT STRATEGIES ===")
    analyze_pareto_strategies(pareto_front)

    return {
        "noise_strategies": best_noise_strategies,
        "pareto_front": pareto_front
    }


if __name__ == "__main__":
    # Run the full demonstration
    results = noise_and_multi_objective_demo()