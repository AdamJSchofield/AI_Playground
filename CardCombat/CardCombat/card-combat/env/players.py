import random
from env.decks import *
from env.cards import Card

class Player:
    def __init__(self, max_health: int, max_energy: int, starting_deck: deck, hand_size: int):
        self.MAX_HEALTH = max_health
        self.STARTING_DECK = starting_deck.cards
        self.HAND_SIZE = hand_size
        self.MAX_ENERGY = max_energy

        self.is_turn = 0
        self.health = max_health
        self.energy = max_energy
        self.damage_buff = 0
        self.drawPile = []
        self.discardPile = []
        self.hand = []
        self.cards_played = []

        self.reset()
    
    # TODO: Clean up card logic, maybe abstraction
    def play_card(self, card: Card, target_player):
        self.hand.remove(card)
        self.discardPile.append(card)
        self.cards_played.append(card.cardType.value)
        self.energy -= card.energyCost
        if card.self_cast:
            self.apply_card(card, self)
        else:
            target_player.apply_card(card, self)

    def apply_card(self, card: Card, source_player = None):
        if source_player is self:
            source_player.damage_buff = card.damage_buff
            self.health = min(self.health + card.healing, self.MAX_HEALTH)

        if source_player is not self:
            self.health = max(0, self.health - card.damage - source_player.damage_buff)
            source_player.damage_buff = 0
    
    def reset(self):
        self.discardPile.append(self.hand)
        self.hand = []
        self.cards_played = []
        self.energy = self.MAX_ENERGY
        self.damage_buff = 0
    
         # Shuffle cards if needed
        if (len(self.drawPile) < self.HAND_SIZE):
            self.drawPile = self.STARTING_DECK.copy()
            random.shuffle(self.drawPile)
            self.discardPile = []

        # Draw cards
        self.hand = random.sample(self.drawPile, self.HAND_SIZE)
        for card in self.hand:
            self.drawPile.remove(card)




