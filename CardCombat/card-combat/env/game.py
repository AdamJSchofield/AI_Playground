import numpy
from players import *
from decks import *
from cards import *

LEGAL_ACTIONS = numpy.array([[0, 0],[0, 2],[0, None],[1, None],[2, None], [None, None]])

class game:
    def __init__(self, player_names):
        self.players = {
            player_name: Player(50, 3, standard_deck(), 4) for player_name in player_names
        }
        self.player_names = player_names

    def is_player_dead(self, requested_player):
        return self.players[requested_player].health <= 0

    def apply_action_sequence(self, requested_player, action_id):
        player: Player = self.players[requested_player]
        name_index = self.player_names.index(requested_player)
        other_player: Player = self.players[self.player_names[1 - name_index]]
        action_sequence = LEGAL_ACTIONS[action_id]
        for c in action_sequence:
            if c is not None:
                card: Card = ALL_CARDS[c]
                player.play_card(card, other_player)

    def get_player_state(self, requested_player):
        player: Player = self.players[requested_player]
        name_index = self.player_names.index(requested_player)
        other_player: Player = self.players[self.player_names[1 - name_index]]
        hand_mask = get_action_mask(player.hand)
        
        observation = [player.health, other_player.health]
        observation = numpy.concatenate([observation, hand_mask], axis=0, dtype=numpy.int8)
        return {
                    "observation": observation,
                    "action_mask": hand_mask
                }
    
    def reset_player_hand(self, requested_player):
        self.players[requested_player].reset_hand()


def get_action_mask(hand: list[Card]):
    hand_values = list(map(lambda c: c.cardType.value, hand))
    return numpy.array(list(map(lambda action: is_action_valid(action, hand_values), LEGAL_ACTIONS)))

def is_action_valid(action, hand_values: list[int]):
    if (action[0] == action[1]):
        if (action[0] is None and action[1] is None):
            return numpy.int8(1)
        return numpy.int8(1) if hand_values.count(action[0]) > 1 else numpy.int8(0)
    return numpy.int8(1) if (action[0] in hand_values or action[0] is None) and (action[1] in hand_values or action[1] is None) else numpy.int8(0)



    


