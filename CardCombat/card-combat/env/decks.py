from cards import *

class standard_deck:
    def __init__(self):
        self.cards = []
        for i in range(10):
            self.cards.append(FireballCard())
        for i in range(3):
            self.cards.append(FireBlastCard())
        for i in range(5):
            self.cards.append(HealCard())
