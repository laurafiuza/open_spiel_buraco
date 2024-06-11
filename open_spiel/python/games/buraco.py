# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel
_GAME_TYPE = pyspiel.GameType(
    short_name="buraco",
    long_name="Buraco",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=True,
    parameter_specification={},
    default_loadable=True,
    provides_factored_observation_string=False
)

_WILDCARD = 2
_ACE = 1
_KING = 13
_QUEEN = 12
_CLEAN_FULL_SEQUENCE = 200
_DIRTY_FULL_SEQUENCE = 100
_MIN_LEN_FULL_SEQUENCE = 7
_ACE_POINTS = 15
_HIGH_POINTS = 10
_LOW_POINTS = 5
_POINT_DEDUCTION_FROM_NOT_GETTING_EXTRA_PILE = 100
_POINTS_FOR_ENDING_GAME = 100

class BuracoGame(pyspiel.Game):
    def __init__(self, params=None):
        game_type = _GAME_TYPE
        game_info = pyspiel.GameInfo(
            num_distinct_actions=2498,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=1000,  # Estimate of maximum moves
        )
        super().__init__(game_type, game_info, params or dict())
        self.deck = [f"{rank}{suit}" for rank in range(1, 14) for suit in 'SHDC']
        self.deck += self.deck

    def new_initial_state(self):
        return BuracoState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return BuracoObserver()

class BuracoState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._cur_player = 0
        self._is_terminal = False
        self.deck = np.random.permutation(self.get_game().deck).tolist()
        self.hands = [self.deck[i*11:(i+1)*11] for i in range(2)]
        self.extra_piles = [self.deck[i*11:(i+1)*11] for i in range(2, 4)]
        self.got_extra_pile = [False, False]
        self.draw_pile = self.deck[45:]
        self.discard_pile = [self.deck[44]]
        self.melds = [[], []]
        self.has_drawn = False
        self.all_actions = ["take_discard_pile", "draw_card"]
        self.cannot_discard = None
        all_ranks = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
        all_suits = ["S", "H", "D", "C"]
        for rank in all_ranks:
            for suit in all_suits:
                self.all_actions.append(f"discard_{rank}{suit}")
        for meld_id in range(35):
            for rank in all_ranks:
                for suit in all_suits:
                    self.all_actions.append(f"add_to_meld_{rank}{suit}_{meld_id}")
        for index, first_rank in enumerate(all_ranks[:-1]):
            second_rank = all_ranks[index + 1]
            third_rank = all_ranks[index + 2] if first_rank != "12" else "1"
            for suit in all_suits:
                self.all_actions.append(f"create_meld_{first_rank}{suit}_{second_rank}{suit}_{third_rank}{suit}")
                for wildcard_suit in all_suits:
                    self.all_actions.append(f"create_meld_2{wildcard_suit}_{second_rank}{suit}_{third_rank}{suit}")
                    self.all_actions.append(f"create_meld_{first_rank}{suit}_2{wildcard_suit}_{third_rank}{suit}")
                    self.all_actions.append(f"create_meld_{first_rank}{suit}_{second_rank}{suit}_2{wildcard_suit}")

    def observation_tensor(self, player_id):
        game = self.get_game()
        state = game.new_initial_state()
        observer = game.make_py_observer()
        observer.set_from(state, player_id)
        return observer.observation_tensor

    def _card_to_index(self, card):
        """Converts a card string to its corresponding index in the observation tensor."""
        # Assuming a standard deck of cards with ranks from 1 to 13 and suits 'S', 'H', 'D', 'C'
        # Convert rank and suit to indices
        rank_index = int(card[:-1]) - 1
        suit_index = {'S': 0, 'H': 1, 'D': 2, 'C': 3}[card[-1]]
        # Calculate the index in the observation tensor
        return rank_index * 4 + suit_index

    def _check_valid_new_meld(self, cards):
        for permutation in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
            action = f"create_meld_{cards[permutation[0]]}_{cards[permutation[1]]}_{cards[permutation[2]]}"
            if action in self.all_actions:
                return True
        return False
    
    def _meld_has_wildcard(self, meld):
        values = [int(card[:-1]) for card in meld["cards"]]
        if _WILDCARD not in values:
            return False
        if values.count(_WILDCARD) == 2:
            return True
        index_of_wildcard = values.index(_WILDCARD)
        if meld["cards"][index_of_wildcard][-1] != meld["suit"]:
            return True
        # Arrives here if there is a card with value 2
        # but it's the same suit as the meld suit, so
        # now we need to check if it's "acting" as a 2
        # or as a wildcard.
        values = sorted(values)
        if values[0] == _ACE and values[-1] == _KING:
            values.remove(_ACE)
            values.append(_ACE)
        index_of_wildcard_when_meld_is_sorted = values.index(_WILDCARD)
        if index_of_wildcard_when_meld_is_sorted - 1 >= 0:
            if values[index_of_wildcard_when_meld_is_sorted - 1] == 1:
                return False
            else:
                return True
        if index_of_wildcard_when_meld_is_sorted + 1 < len(values):
            if values[index_of_wildcard_when_meld_is_sorted + 1] == 3:
                return False
            else:
                return True

    def _can_add_card_to_meld(self, card, meld):
        card_value = int(card[:-1])
        card_suit = card[-1]
        if card_value != _WILDCARD and card_suit != meld["suit"]:
            return False

        values = [int(card[:-1]) for card in meld["cards"]]

        """Check if card can be appended or prepended to meld."""
        # Place in left edge of meld, when the wildcard isn't leftmost
        if values[0] != _WILDCARD and card_value == values[0] - 1:
            return True
        
        # Place in left edge of the meld, when the wildcard is leftmost
        if values[0] == _WILDCARD and card_value == values[1] - 2:
            return True

        # Place in right edge of meld, when the wildcard isn't rightmost
        if values[-1] != _WILDCARD and card_value == values[-1] + 1:
            return True
        
        # Place in the right edge of the meld, when the wildcard is rightmost
        if values[-1] == _WILDCARD and card_value == values[-2] + 2:
            return True
        
        # Place in the right edge of the meld, when the wildcard is replacing King
        # and second to last card is Queen
        if card_value == _ACE and values[0] == _WILDCARD and values[-1] == _QUEEN:
            return True
        
        # Place in the right edge of the meld, when the wildcard is replacing Ace
        if card_value == _ACE and values[-1] == _KING:
            return True
        
        if values[0] == _WILDCARD and card_value == values[-1] + 2:
            return True

        """Check if wildcard is in the middle of meld and can be replaced by current
        non-wildcard card."""
        wildcard = None
        wildcard_index = None
        for i, value in enumerate(meld["cards"]):
            if int(value[:-1]) == _WILDCARD:
                if not wildcard:
                    wildcard = meld["cards"][i]
                    wildcard_index = i
                elif value[-1] != meld["suit"]:
                    wildcard = meld["cards"][i]
                    wildcard_index = i
                    break
        if not wildcard:
            return card_value == _WILDCARD
        if card_value == _WILDCARD and card_suit != meld["suit"] and wildcard[-1] != meld["suit"]:
            return False
        if wildcard_index - 1 >= 0 and int(meld["cards"][wildcard_index - 1][:-1]) == card_value - 1:
            return True
        if wildcard_index + 1 < len(meld["cards"]) and int(meld["cards"][wildcard_index + 1][:-1]) == card_value + 1:
            return True
        return False

    def _map_legal_actions_to_ints(self, legal_actions):
        indices = []
        for action in legal_actions:
            if "create_meld" not in action and action not in self.all_actions:
                raise ValueError(f"Invalid action: {action}")
            if "create_meld" in action:
                cards = action.split("_")[2:]
                if len(cards) != 3:
                    raise ValueError("Invalid meld creation, len != 3")
                found = False
                for permutation in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
                    action = f"create_meld_{cards[permutation[0]]}_{cards[permutation[1]]}_{cards[permutation[2]]}"
                    if action in self.all_actions:
                        found = True
                        break
                if not found:
                    raise ValueError("Invalid meld creation, couldn't find in all_actions", action)
            index = self.all_actions.index(action)
            indices.append(index)
        return indices

    def _legal_actions(self, player=None):
        if self.is_terminal():
            return []

        if player and player != self._cur_player:
            print("Alert: Not player's turn")
            return []
        
        actions = []

        # Drawing phase: Can draw if hasn't drawn yet
        if not self.has_drawn:
            if self.draw_pile:
                actions.append("draw_card")
            if self.discard_pile:
                actions.append("take_discard_pile")
        else:
            # Adding discard actions
            for card in self.hands[self._cur_player]:
                if card == self.cannot_discard:
                    continue
                actions.append(f"discard_{card}")

            # Generating unique melds
            num_cards = len(self.hands[self._cur_player])
            sorted_cards = sorted(self.hands[self._cur_player])

            # Generate all possible combinations of three cards
            for i in range(num_cards):
                for j in range(i + 1, num_cards):
                    for k in range(j + 1, num_cards):
                        meld_candidates = [sorted_cards[i], sorted_cards[j], sorted_cards[k]]
                        action_id = f"create_meld_{'_'.join(meld_candidates)}"
                        if action_id not in actions and self._check_valid_new_meld(meld_candidates):
                            actions.append(action_id)

            # Adding to existing melds
            if num_cards > 2 or not self.got_extra_pile[self._cur_player] or any(len(meld["cards"]) >= 7 for meld in self.melds[self._cur_player]):
                for card in sorted_cards:
                    for i, meld in enumerate(self.melds[self._cur_player]):
                        # Ensure that the card addition is valid before adding it to actions
                        if self._can_add_card_to_meld(card, meld):
                            actions.append(f"add_to_meld_{card}_{i}")
        
        action_indices = self._map_legal_actions_to_ints(actions)
        return action_indices

    def apply_action(self, action):
        action_index = action if isinstance(action, int) or isinstance(action, np.int64) else action.action
        if action_index >= len(self.all_actions):
            raise ValueError("Invalid action: {}".format(action.action))
        action_string = self.all_actions[action_index]
        if action_string in ["draw_card", "take_discard_pile"]:
            self._handle_draw_action(action_string)
            return self
        elif "discard_" in action_string:
            self._handle_discard_action(action_string)
            return self
        elif "create_meld_" in action_string:
            self._handle_create_meld_action(action_string)
            return self
        elif "add_to_meld_" in action_string:
            self._handle_add_to_meld_action(action_string)
            return self
        else:
            raise ValueError("Unknown action: {}".format(action_string))
    
    def _handle_draw_action(self, action):
        if action == "draw_card" and self.draw_pile:
            card = self.draw_pile.pop(0)
            self.hands[self._cur_player].append(card)
        elif action == "take_discard_pile":
            if not self.discard_pile:
                raise ValueError("Discard pile is empty")
            if len(self.discard_pile) == 1:
                self.cannot_discard = self.discard_pile[0]
            self.hands[self._cur_player].extend(self.discard_pile)
            self.discard_pile.clear()
        else:
            raise ValueError("Draw action not possible or invalid")
        self.has_drawn = True  # Set flag that a card has been drawn this turn

    def _handle_discard_action(self, action):
        self.cannot_discard = None
        card = action.split("_")[1]
        if card not in self.hands[self._cur_player]:
            raise ValueError("Card not in hand")
        self.hands[self._cur_player].remove(card)
        self.discard_pile.append(card)
        should_terminate = self._check_should_pop_extra_pile_or_terminate_game()
        if should_terminate:
            return
        # Go to next player
        self._cur_player = 1 - self._cur_player
        self.has_drawn = False
        if not self.draw_pile:
            if self.extra_piles:
                self.draw_pile = self.extra_piles.pop(0)
            else:
                self._is_terminal = True
    
    def _check_should_pop_extra_pile_or_terminate_game(self):
        if not self.hands[self._cur_player]:
            if self.extra_piles and not self.got_extra_pile[self._cur_player]:
                self.hands[self._cur_player] = self.extra_piles.pop(0)
                self.got_extra_pile[self._cur_player] = True
                return False
            else:
                self._is_terminal = True
                return True
        return False

    def _sort_meld(self, suit_of_meld, cards):
        # Sorts by rank and suit but keeps wildcard in its right place.
        # Ace is in it's correct place, whether it's in the beginning or end of sequence.
        wildcard = None
        wildcard_index = None
        for i, card in enumerate(cards):
            if int(card[:-1]) == _WILDCARD:
                if not wildcard:
                    wildcard = cards[i]
                    wildcard_index = i
                elif card[-1] != suit_of_meld:
                    wildcard = cards[i]
                    wildcard_index = i
                    break
        if not wildcard:
            values = [int(card[:-1]) for card in cards]
            values = sorted(values)
            if values[0] == _ACE and values[-1] == _KING:
                values.remove(_ACE)
                values.append(_ACE)
            return [f"{value}{suit_of_meld}" for value in values]
        else:
            cards.remove(wildcard)
            values = [int(card[:-1]) for card in cards]
            values = sorted(values)
            if values[0] == _ACE and values[-1] == _KING:
                values.remove(_ACE)
                values.append(_ACE)
            for i, value in enumerate(values):
                if value == _KING and values[-1] == _ACE:
                    continue
                if i < len(values) - 1 and values[i + 1] != value + 1:
                    return [f"{value}{suit_of_meld}" for value in values[:i + 1]] + [wildcard] + [f"{value}{suit_of_meld}" for value in values[i + 1:]]
            return [wildcard] + [f"{value}{suit_of_meld}" for value in values]

    def _handle_create_meld_action(self, action):
        cards = action.split("_")[2:]
        if len(cards) != 3:
            raise ValueError("Invalid meld creation")
        for card in cards:
            if card not in self.hands[self._cur_player]:
                raise ValueError("Card not in hand")
            self.hands[self._cur_player].remove(card)
        
        values = [int(card[:-1]) for card in cards]
        suits = [card[-1] for card in cards if int(card[0]) != _WILDCARD]
        sorted_cards = self._sort_meld(suits[0], cards)
        self.melds[self._cur_player].append({"suit": suits[0], "cards": sorted_cards})
        self._check_should_pop_extra_pile_or_terminate_game()

    def _handle_add_to_meld_action(self, action):
        card, meld = action.split("_")[3:]
        meld = int(meld)
        if card not in self.hands[self._cur_player]:
            raise ValueError("Card not in hand")
        if meld < 0 or meld >= len(self.melds[self._cur_player]):
            raise ValueError("Meld not in melds")
        self.hands[self._cur_player].remove(card)
        self.melds[self._cur_player][meld]["cards"].append(card)
        sorted_meld = self._sort_meld(self.melds[self._cur_player][meld]["suit"], self.melds[self._cur_player][meld]["cards"])
        self.melds[self._cur_player][meld]["cards"] = sorted_meld
    
    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        return self._cur_player

    def is_terminal(self):
        return (not self.draw_pile and not self.extra_piles) or any(not hand for hand in self.hands)

    def _get_card_point(self, card_value):
        if card_value == _ACE:
            return _ACE_POINTS
        elif card_value == _WILDCARD or card_value >= 8:
            return _HIGH_POINTS
        else:
            return _LOW_POINTS

    def returns(self):
        scores = [0, 0]
        if self.draw_pile:
            scores[self._cur_player] += _POINTS_FOR_ENDING_GAME
            for i, hand in enumerate(self.hands):
                for card in hand:
                    scores[i] -= self._get_card_point(int(card[:-1]))
        for i, meld in enumerate(self.melds):
            for meld in meld:
                if len(meld["cards"]) >= _MIN_LEN_FULL_SEQUENCE:
                    scores[i] += _CLEAN_FULL_SEQUENCE
                for card in meld["cards"]:
                    scores[i] += self._get_card_point(int(card[:-1]))
        if not self.got_extra_pile[0]:
            scores[0] -= _POINT_DEDUCTION_FROM_NOT_GETTING_EXTRA_PILE
        if not self.got_extra_pile[1]:
            scores[1] -= _POINT_DEDUCTION_FROM_NOT_GETTING_EXTRA_PILE
        return scores

    def __str__(self):
        return f"Current player: {self._cur_player}, Hands: {self.hands}, Melds: {self.melds}, Discard Pile: {self.discard_pile}"

class BuracoObserver(IIGObserverForPublicInfoGame):
    def __init__(self, iig_obs_type=None, params=None):
        super().__init__(iig_obs_type, params)
        num_cards = 104  # 52 cards in two decks
        self.num_cards = num_cards
        self.max_melds = 35  # Estimate of maximum number of melds

        # Define the shape of the observation tensor
        self.tensor_shape = (
            1 +  # Opponent's hand size
            1 +  # Number of draw pile cards
            num_cards +  # Discard pile values
            self.max_melds * num_cards +  # Opponent's melds
            self.max_melds * num_cards +  # Agent's melds
            num_cards +  # Agent's hand
            1 +  # Number of extra piles left
            2  # Who got the extra pile
        )
        self.tensor_shape = (self.tensor_shape,)  # Make it a tuple
        assert self.tensor_shape[0] == 7493

    def set_from(self, state, player):
        self.observation_tensor = np.zeros(self.tensor_shape, dtype=np.float32)

        # Encode the number of cards the opponent has
        opponent_hand_size_offset = 0
        self.observation_tensor[opponent_hand_size_offset] = len(state.hands[1 - player])

        # Encode the number of cards in the draw pile
        draw_pile_count_offset = opponent_hand_size_offset + 1
        self.observation_tensor[draw_pile_count_offset] = len(state.draw_pile)

        # Encode the values of cards in the discard pile
        discard_pile_offset = draw_pile_count_offset + 1
        for card in state.discard_pile:
            card_index = state._card_to_index(card)
            self.observation_tensor[discard_pile_offset + card_index] = 1

        # Encode the opponent's melds
        opponent_melds_offset = discard_pile_offset + self.num_cards
        for meld in state.melds[1 - player]:
            for card in meld["cards"]:
                card_index = state._card_to_index(card)
                self.observation_tensor[opponent_melds_offset + card_index] = 1
            opponent_melds_offset += self.num_cards

        # Encode the agent's melds
        agent_melds_offset = discard_pile_offset + self.num_cards + self.max_melds * self.num_cards
        for meld in state.melds[player]:
            for card in meld["cards"]:
                card_index = state._card_to_index(card)
                self.observation_tensor[agent_melds_offset + card_index] = 1
            agent_melds_offset += self.num_cards

        # Encode the values of cards in the agent's hand
        agent_hand_offset = agent_melds_offset
        for card in state.hands[player]:
            card_index = state._card_to_index(card)
            self.observation_tensor[agent_hand_offset + card_index] = 1

        # Encode the number of extra piles left
        extra_piles_offset = agent_hand_offset + self.num_cards
        self.observation_tensor[extra_piles_offset] = len(state.extra_piles)

        # Encode who got the extra pile
        extra_pile_status_offset = extra_piles_offset + 1
        self.observation_tensor[extra_pile_status_offset] = state.got_extra_pile[player]
        self.observation_tensor[extra_pile_status_offset + 1] = state.got_extra_pile[1 - player]

    def string_from(self, state, player):
        return f"Player {player} observation: {self.observation_tensor}"

# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, BuracoGame)