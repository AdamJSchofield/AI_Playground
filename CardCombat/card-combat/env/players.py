import random
from decks import *
from cards import Card

class Player:
    def __init__(self, max_health: int, energy: int, starting_deck: standard_deck, hand_size: int):
        self.MAX_HEALTH = max_health
        self.ENERGY = energy
        self.STARTING_DECK = starting_deck.cards
        self.HAND_SIZE = hand_size

        self.health = max_health
        self.drawPile = []
        self.discardPile = []
        self.hand = []

        self.reset_hand()
    
    def play_card(self, card: Card, other_player):
        self.hand.remove(card)
        self.discardPile.append(card)
        if card.self_cast:
            self.apply_card(card)
        else:
            other_player.apply_card(card)

    def apply_card(self, card: Card):
        self.health -= card.damage
        self.health = min(self.health + card.healing, self.MAX_HEALTH)
    
    def reset_hand(self):
        self.discardPile.append(self.hand)
        self.hand = []
    
         # Shuffle cards if needed
        if (len(self.drawPile) < self.HAND_SIZE):
            self.drawPile = self.STARTING_DECK.copy()
            random.shuffle(self.drawPile)
            self.discardPile = []

        # Draw cards
        self.hand = random.sample(self.drawPile, self.HAND_SIZE)
        for card in self.hand:
            self.drawPile.remove(card)




