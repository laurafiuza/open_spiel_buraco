import os
import buraco
from open_spiel.python import rl_environment
import copy
import numpy as np
import time
import json

EXPLORATION_CONSTANT = 2
RANDOM_BOT = 0
HARDCODED_BOT = 1
_WILDCARD = 2

def pick_best_action_given_hardcoded_logic(state):
    legal_actions = state.legal_actions()
    player_id = state.current_player()
    hand = state.hands[player_id]
    melds = state.melds[player_id]

    if legal_actions == [1, 0]:
        for card in state.discard_pile:
            # Check if the card can be added to an existing meld
            for meld in melds:
                if state._can_add_card_to_meld(card, meld):
                    return state.all_actions.index("take_discard_pile")
            # Check if the card can form a sequence with two other cards in hand
            for other_card1 in hand:
                for other_card2 in hand:
                    if other_card1 != card and other_card2 != card and other_card1 != other_card2:
                        if state._check_valid_new_meld([card, other_card1, other_card2]):
                            return state.all_actions.index("take_discard_pile")
        return state.all_actions.index("draw_card")

    # 1. Add to existing melds if the card added is not a wildcard
    for action in legal_actions:
        if "add_to_meld" in state.all_actions[action]:
            card = state.all_actions[action].split("_")[3]
            if int(card[:-1]) != _WILDCARD:
                return action

    # 2. Create a new meld if it's not a sequence that is one or two cards away from an existing sequence AND it's not a wildcard
    for action in legal_actions:
        if "create_meld" in state.all_actions[action]:
            cards = state.all_actions[action].split("_")[2:]
            use_wildcard = any(int(card[:-1]) == _WILDCARD for card in cards)
            if use_wildcard:
                continue
            return action

    remaining_actions = []

    # Prepare remaining legal actions
    for action in legal_actions:
        if "discard_" in state.all_actions[action]:
            card = state.all_actions[action].split("_")[1]
            if int(card[:-1]) != _WILDCARD:
                remaining_actions.append(action)
        else:
            remaining_actions.append(action)

    # Remove wildcard discards
    remaining_actions = [action for action in remaining_actions if "discard_" not in state.all_actions[action] or int(state.all_actions[action].split("_")[1][:-1]) != _WILDCARD]

    if not remaining_actions:
        return legal_actions[0]

    # Choose discard action with 80% probability
    if np.random.rand() < 0.8:
        discard_actions = [action for action in remaining_actions if "discard_" in state.all_actions[action]]
        if discard_actions:
            # Easy and hard discards
            easy_discards = ["3", "13", "1", "4", "12", "5"]
            hard_discards = ["11", "6", "10", "7", "9", "8"]

            easy_discard_actions = [action for action in discard_actions if state.all_actions[action].split("_")[1][:-1] in easy_discards]
            hard_discard_actions = [action for action in discard_actions if state.all_actions[action].split("_")[1][:-1] in hard_discards]

            if np.random.rand() < 0.9 and easy_discard_actions:
                return np.random.choice(easy_discard_actions)
            elif hard_discard_actions:
                return np.random.choice(hard_discard_actions)
    else:
        non_discard_actions = [action for action in remaining_actions if "discard_" not in state.all_actions[action]]
        if non_discard_actions:
            return np.random.choice(non_discard_actions)

    # Default to the first legal action if no other rule applies
    return legal_actions[0]

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.points_accrued = 0
        self.explored_actions = set()
        self.action = action

def mcts(root_state, num_simulations, bot_type):
    root_node = Node(root_state)

    for sim in range(num_simulations):
        node = root_node
        state = copy.deepcopy(root_state)

        # Selection
        while not state.is_terminal():
            if len(node.children) < len(state.legal_actions()):
                # Expansion
                unexplored_actions = [action for action in state.legal_actions() if action not in node.explored_actions]
                if unexplored_actions:
                    action = np.random.choice(unexplored_actions)
                    next_state = state.clone().apply_action(action)
                    child_node = Node(next_state, parent=node, action=action)
                    node.children.append(child_node)
                    node.explored_actions.add(action)
                    state = next_state
                    break

            # UCB1 selection
            best_child = max(node.children, key=lambda c: (c.wins / c.visits if c.visits > 0 else float('inf')) + EXPLORATION_CONSTANT * np.sqrt(2 * np.log(node.visits) / c.visits if c.visits > 0 else float('inf')))
            action = best_child.action
            state.apply_action(action)
            node = best_child
        
        # Simulation
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            if not legal_actions:
                print("Legal actions is empty")
                break
            action = pick_best_action_given_hardcoded_logic(state) if bot_type == HARDCODED_BOT else np.random.choice(legal_actions)
            state.apply_action(action)

        # Backpropagation
        learning_player = root_state.current_player()
        while node is not None:
            node.visits += 1
            points_0, points_1 = state.returns()
            if points_0 > points_1:
                if learning_player == 0:
                    node.wins += 1
            if points_1 > points_0:
                if learning_player == 1:
                    node.wins += 1
            node.points_accrued += points_0 if learning_player == 0 else points_1
            node = node.parent
    
    best_child = max(root_node.children, key=lambda c: c.wins / c.visits if c.visits > 0 else 0)
    best_action = best_child.action
    return best_action

results = []

def play_against_mcts(env, num_simulations, bot_type):
    env.reset()
    while True:
        print("Current state:")
        print(env._state)
        if env._state.is_terminal():
            print("Game over.")
            points_random, points_mcts = env._state.returns()
            print(f"Points: Random Agent: {points_random}, MCTS Agent: {points_mcts}")
            results.append([points_random, points_mcts])
            break

        
        player = env._state.current_player()
        legal_actions = env._state.legal_actions()
        print("Possible actions")
        for action in legal_actions:
            print("\t", action, env._state.all_actions[action])
        if player == 0:
            # while loop with check if number is valid and if it's in legal actions
            while True:
                action = input("Enter action: ")
                if not len(action):
                    print("Invalid action. Try again.")
                    continue
                action = int(action)
                if action in legal_actions:
                    break
                print("Invalid action. Try again.")
        else:
            action = mcts(env._state, num_simulations, bot_type)
            print("MCTS Agent Picked action", action, env._state.all_actions[action])

        time_step = env.step([action])
        if time_step.last():
            print("Game over.")
            points_random, points_mcts = env._state.returns()
            print(f"Points: Random Agent: {points_random}, MCTS Agent: {points_mcts}")
            results.append([points_random, points_mcts])
            break

# Example usage
game_instance = buraco.BuracoGame()
env = rl_environment.Environment(game_instance)
num_simulations = 5

play_against_mcts(env, num_simulations, HARDCODED_BOT)