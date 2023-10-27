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
        self.drawPile = []
        self.discardPile = []
        self.hand = []
        self.cards_played = []

        self.reset()
    
    def play_card(self, card: Card, other_player):
        self.hand.remove(card)
        self.discardPile.append(card)
        self.cards_played.append(card.cardType.value)
        self.energy -= card.energyCost
        if card.self_cast:
            self.apply_card(card)
        else:
            other_player.apply_card(card)

    def apply_card(self, card: Card):
        self.health -= card.damage
        self.health = min(self.health + card.healing, self.MAX_HEALTH)
    
    def reset(self):
        self.discardPile.append(self.hand)
        self.hand = []
        self.cards_played = []
        self.energy = self.MAX_ENERGY
    
         # Shuffle cards if needed
        if (len(self.drawPile) < self.HAND_SIZE):
            self.drawPile = self.STARTING_DECK.copy()
            random.shuffle(self.drawPile)
            self.discardPile = []

        # Draw cards
        self.hand = random.sample(self.drawPile, self.HAND_SIZE)
        for card in self.hand:
            self.drawPile.remove(card)




