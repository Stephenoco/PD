import random

# Constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_ROUNDS = 400  # Number of rounds in each IPD game
MEMORY_SIZE = 3  # How many previous moves to remember

# Payoff matrix for Prisoner's Dilemma
# (row player's payoff, column player's payoff)

PAYOFF_MATRIX = {
    ('C', 'C'): (3, 3),  # Both cooperate
    ('C', 'D'): (0, 5),  # Row cooperates, column defects
    ('D', 'C'): (5, 0),  # Row defects, column cooperates
    ('D', 'D'): (1, 1)  # Both defect
}


# Fixed strategies
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

    # Get last round's moves
    my_last_move = history[-1]
    opponent_last_move = opponent_history[-1]

    # Determine last round's payoff
    my_payoff = PAYOFF_MATRIX[(my_last_move, opponent_last_move)][0]

    # If I got a good payoff (3 or 5), stick with the same move
    if my_payoff > 1:
        return my_last_move
    # Otherwise, change my move
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

def create_strategy_table():
    table = {}

    # Generate all possible history combinations
    for i in range(2 ** (2 * MEMORY_SIZE)):
        history_key = format(i, f'0{2 * MEMORY_SIZE}b')
        # Convert binary string to a tuple of move histories
        my_history = tuple(('C' if h == '0' else 'D') for h in history_key[:MEMORY_SIZE])
        opp_history = tuple(('C' if h == '0' else 'D') for h in history_key[MEMORY_SIZE:])

        # Randomly assign a move (C or D) with 50% probability
        table[(my_history, opp_history)] = 'C' if random.random() < 0.5 else 'D'

    return table


def lookup_table_strategy(lookup_table, history, opponent_history):
    # For the first few rounds when history is limited
    if len(history) < MEMORY_SIZE:
        # Pad with 'C's for missing history
        padded_history = ('C',) * (MEMORY_SIZE - len(history)) + tuple(history[-MEMORY_SIZE:] if history else ())
        padded_opp_history = ('C',) * (MEMORY_SIZE - len(opponent_history)) + tuple(
            opponent_history[-MEMORY_SIZE:] if opponent_history else ())
    else:
        # Use the last MEMORY_SIZE moves
        padded_history = tuple(history[-MEMORY_SIZE:])
        padded_opp_history = tuple(opponent_history[-MEMORY_SIZE:])

    # Return the move from the lookup table
    return lookup_table[(padded_history, padded_opp_history)]


def play_game(strategy1, strategy2, num_rounds=NUM_ROUNDS):
    history1 = []
    history2 = []
    score1 = 0
    score2 = 0

    for x in range(num_rounds):
        # Determine moves based on history
        if isinstance(strategy1, dict):
            move1 = lookup_table_strategy(strategy1, history1, history2)
        else:
            move1 = strategy1(history1, history2)

        if isinstance(strategy2, dict):
            move2 = lookup_table_strategy(strategy2, history2, history1)
        else:
            move2 = strategy2(history2, history1)

        # Update scores based on payoff matrix
        payoff = PAYOFF_MATRIX[(move1, move2)]
        score1 += payoff[0]
        score2 += payoff[1]

        # Update histories
        history1.append(move1)
        history2.append(move2)

    return score1, score2


def calculate_fitness(strategy):
    total_score = 0

    for fixed_strategy_name, fixed_strategy_func in FIXED_STRATEGIES.items():
        # Play against each fixed strategy
        score, _ = play_game(lambda h1, h2: lookup_table_strategy(strategy, h1, h2), fixed_strategy_func)
        total_score += score

    return total_score


def tournament_selection(population, fitnesses, tournament_size):
    # Randomly select tournament_size individuals
    tournament_indices = random.sample(range(len(population)), tournament_size)

    # Find the best individual in the tournament
    best_idx = tournament_indices[0]
    best_fitness = fitnesses[best_idx]

    for idx in tournament_indices[1:]:
        if fitnesses[idx] > best_fitness:
            best_idx = idx
            best_fitness = fitnesses[idx]

    return population[best_idx]


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy()

    child = {}

    # Get all keys (history combinations)
    keys = list(parent1.keys())

    # Choose a random crossover point
    crossover_point = random.randint(1, len(keys) - 1)

    # Combine parents
    for i, key in enumerate(keys):
        if i < crossover_point:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]

    return child


def mutate(strategy):
    mutated_strategy = strategy.copy()

    for key in mutated_strategy:
        if random.random() < MUTATION_RATE:
            # Flip the move
            mutated_strategy[key] = 'D' if mutated_strategy[key] == 'C' else 'C'

    return mutated_strategy