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

"""Tests for Python Simplified Buraco."""

import difflib
import os
import pickle

from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms.get_all_states import get_all_states
import pyspiel
import buraco
from open_spiel.python.observation import make_observation

class BuracoTest(absltest.TestCase):
  def test_can_create_game_and_state(self):
    """Checks we can create the game and a state."""
    game = buraco.BuracoGame()
    state = game.new_initial_state()
    self.assertEqual(len(state.deck), 104)
    self.assertEqual(len(state.hands), 2)
    self.assertEqual(len(state.hands[0]), 11)
    self.assertEqual(len(state.hands[1]), 11)
    self.assertEqual(len(state.draw_pile), 59)
    self.assertEqual(len(state.discard_pile), 1)
    self.assertEqual(len(state.melds), 2)
    self.assertEqual(len(state.melds[0]), 0)
    self.assertEqual(len(state.melds[1]), 0)
    self.assertEqual(len(state.extra_piles[0]), 11)
    self.assertEqual(len(state.extra_piles[1]), 11)
    self.assertEqual(len(state.all_actions), 2498)

  def test_random_game(self):
    #Checks that we can play a random game.
    game = buraco.BuracoGame()
    state = game.new_initial_state()
    i = 0
    while not state.is_terminal():
      #print("Current player: ", state.current_player())
      #print("Player A: ", state.hands[0])
      #print("Player B: ", state.hands[1])
      #print("Discard pile: ", state.discard_pile)
      #print("Length of draw pile: ", len(state.draw_pile))
      #print("Player A Melds: ", state.melds[0])
      #print("Player B Melds: ", state.melds[1])
      #print("Extra piles: ", state.extra_piles)
      legal_actions = state._legal_actions()
      #print("Legal actions: ", legal_actions)
      self.assertTrue(legal_actions)
      action = np.random.choice(legal_actions)
      #print("Action: ", action)
      state.apply_action({"action": action})
      # TODO: make above line work
      if i == 0:
        if action == "draw_card":
          self.assertEqual(len(state.draw_pile), 58)
          self.assertEqual(len(state.hands[0]), 12)
          self.assertEqual(len(state.hands[1]), 11)
          self.assertEqual(len(state.discard_pile), 1)
        elif action == "take_discard_pile":
          self.assertEqual(len(state.discard_pile), 0)
          self.assertEqual(len(state.hands[0]), 12)
          self.assertEqual(len(state.hands[1]), 11)
          self.assertEqual(len(state.discard_pile), 0)
        i += 1
    print(state.returns())

  def test_create_new_meld(self):
    """Checks that we can create a new meld."""
    game = buraco.BuracoGame()
    state = game.new_initial_state()
    meld = ["1H", "2H", "3H"]
    self.assertTrue(state._check_valid_new_meld(meld))

    meld = ["1H", "2H", "4H"]
    self.assertFalse(state._check_valid_new_meld(meld))

    meld = ["1H", "2H", "13H"]
    self.assertTrue(state._check_valid_new_meld(meld))

    meld = ["3H", "2H", "5H"]
    self.assertTrue(state._check_valid_new_meld(meld))

    meld = ["2D", "3H", "4H"]
    self.assertTrue(state._check_valid_new_meld(meld))

    meld = ["2H", "4H", "5H"]
    self.assertTrue(state._check_valid_new_meld(meld))

    meld = ["3H", "2H", "10H"]
    self.assertFalse(state._check_valid_new_meld(meld))

    meld = ["1H", "2H", "13H"]
    self.assertTrue(state._check_valid_new_meld(meld))

    meld = ["1H", "2H", "3S"]
    self.assertFalse(state._check_valid_new_meld(meld))

    meld = ["11H", "12H", "13H"]
    self.assertTrue(state._check_valid_new_meld(meld))

    meld = ["12H", "13H", "1H"]
    self.assertTrue(state._check_valid_new_meld(meld))

  def test_add_card_to_meld(self):
    """Checks that we can add a card to a meld."""
    game = buraco.BuracoGame()
    state = game.new_initial_state()
    meld = {"suit": "H", "cards": ["1H", "2D", "3H"]}
    self.assertTrue(state._can_add_card_to_meld("4H", meld))
    self.assertFalse(state._can_add_card_to_meld("4S", meld))
    self.assertFalse(state._can_add_card_to_meld("5H", meld))
    self.assertTrue(state._can_add_card_to_meld("2H", meld))
    self.assertFalse(state._can_add_card_to_meld("2C", meld))

    meld = {"suit": "H", "cards": ["11H", "12H", "13H"]}
    self.assertTrue(state._can_add_card_to_meld("10H", meld))
    self.assertTrue(state._can_add_card_to_meld("1H", meld))
    self.assertFalse(state._can_add_card_to_meld("9H", meld))
    self.assertFalse(state._can_add_card_to_meld("1D", meld))
    self.assertTrue(state._can_add_card_to_meld("2D", meld))
    self.assertTrue(state._can_add_card_to_meld("2H", meld))

    meld = {"suit": "D", "cards": ["2C", "3D", "4D"]}
    self.assertTrue(state._can_add_card_to_meld("2D", meld))
    self.assertTrue(state._can_add_card_to_meld("5D", meld))
    self.assertFalse(state._can_add_card_to_meld("5H", meld))
    self.assertTrue(state._can_add_card_to_meld("6D", meld))
    self.assertTrue(state._can_add_card_to_meld("1D", meld))
    self.assertFalse(state._can_add_card_to_meld("2H", meld))

    meld = {"suit": "S", "cards": ["2S", "3S", "5S", "6S", "7S"]}
    self.assertTrue(state._can_add_card_to_meld("4S", meld))

    meld = {"suit": "S", "cards": ["2D", "10S", "11S", "12S"]}
    self.assertTrue(state._can_add_card_to_meld("1S", meld))

    meld = {"suit": "S", "cards": ['2S', '11S', '12S', '13S', '1S']}
    self.assertFalse(state._can_add_card_to_meld("3S", meld))

  def test_sort_meld(self):
    """Checks that we can sort a meld."""
    game = buraco.BuracoGame()
    state = game.new_initial_state()
    meld = ["1H", "13H", "2H"]
    self.assertEqual(state._sort_meld("H", meld), ["2H", "13H", "1H"])

    meld = ["13S", "1S", "2C"]
    self.assertEqual(state._sort_meld("S", meld), ["2C", "13S", "1S"])

    meld = ["2H", "3H", "1H"]
    self.assertEqual(state._sort_meld("H", meld), ["1H", "2H", "3H"])

    meld = ["1H", "13H", "2D"]
    self.assertEqual(state._sort_meld("H", meld), ["2D", "13H", "1H"])

    meld = ["2H", "3H", "5H"]
    self.assertEqual(state._sort_meld("H", meld), ["3H", "2H", "5H"])

    meld = ["9D", "11D", "10D"]
    self.assertEqual(state._sort_meld("D", meld), ["9D", "10D", "11D"])

if __name__ == "__main__":
    absltest.main()
