import numpy
from env.players import *
from env.decks import *
from env.cards import *
from collections import OrderedDict

class game:

    HAND_SIZE = 4
    DECK_SIZE = 30

    def __init__(self, player_names, player_healths = [50, 50], player_energies = [3, 3], player_decks= [DECK_TYPE.STANDARD, DECK_TYPE.STANDARD]):
       
        self.players = {player_names[i]: Player(player_healths[i], player_energies[i], deck(player_decks[i]), self.HAND_SIZE, give_coin=(i == 1)) for i in range(2)}
        self.player_names = player_names

        actions = ALL_CARDS.copy()
        actions.append(None)
        self.ACTIONS = actions
        # Name and max value for organization. Flattened in get_observation_space to values representing the max discrete value for that space (i.e highest card type value, highest count for given card, etc...)
        self.OBSERVATION_SHAPE = flatten_observation({
            "player_health": max(player_healths) + 1, # if max health is 50, number of values could be 0-50 which is 51 possible values
            "opponent_health": max(player_healths) + 1,
            "player_energy": max(player_energies) + 2,
            "hand_counts": [self.HAND_SIZE + 1] * len(ALL_CARDS), # ex [2, 0, 1] would be 2 FIREBALL, 0 FIREBLAST, 1 HEAL 

            # Num of possible values is len + 1 to account for possible -1 padding value
            "cards_played": [len(ALL_CARDS) + 1] * (self.HAND_SIZE + 1), # ex [2, 1, None, None] would be HEAL then FIREBLAST played so far this turn. Max value is len(ALL_CARDS) + 1 to account for Coin added on start
            "opponent_cards_played": [len(ALL_CARDS) + 1] * (self.HAND_SIZE + 1) # [2, None, None, None] means the opponent played HEAL then ended the turn
        })

    def is_player_dead(self, player_name):
        return self.players[player_name].health <= 0

    def get_player_state(self, player_name):
        player: Player = self.players[player_name]
        player_index = self.player_names.index(player_name)

        other_player_name = self.player_names[1 - player_index]
        other_player: Player = self.players[other_player_name]
        
        observation = {
            "player_health": player.health,
            "opponent_health": other_player.health,
            "player_energy": player.energy,
            "hand_counts": get_hand_counts(player.hand),
            "cards_played": pad_list(player.cards_played, self.HAND_SIZE + 1, len(ALL_CARDS)), # If ALL_CARDS contains types [0, 1, 2], pad with 3's for no action
            "opponent_cards_played": pad_list(other_player.cards_played, self.HAND_SIZE + 1, len(ALL_CARDS))
        }

        return  {
                    "observation": flatten_observation(observation),
                    "action_mask": self.get_action_mask(player)
                }
    
    def take_action(self, action, player_name, opponent_name):
        if self.ACTIONS[action] is not None:
            self.players[player_name].play_card(ALL_CARDS[action], self.players[opponent_name])
            return False
        else:
            return True


    def get_action_mask(self, player):
        mask = []
        for card in ALL_CARDS:
            if card in player.hand and card.energyCost <= player.energy:
                mask.append(1)
            else:
                mask.append(0)
        mask.append(1)
        return numpy.array(mask, dtype=numpy.int8)

def flatten_observation(observation):
    flat = []
    for value in observation.values():
        if (isinstance(value, list) or isinstance(value, numpy.ndarray)):
            flat = flat + value
        else:
            flat.append(value)
    return numpy.array(flat, dtype=numpy.int8)

def get_hand_counts(hand: list[Card]):
    hand_counts = []
    for card in ALL_CARDS:
        hand_counts.append(hand.count(card))
    return hand_counts

def pad_list(list: list, length: int, pad_value: int):
    return list + [pad_value] * (length - len(list))




    


