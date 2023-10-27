from enum import Enum
from env.cards import *

class deck:
    def __init__(self, type):
        self.cards = []

        if (type == DECK_TYPE.STANDARD):
            for i in range(20):
                self.cards.append(FireballCard())
            for i in range(5):
                self.cards.append(FireBlastCard())
            for i in range(10):
                self.cards.append(HealCard())
        else:
            raise f'No deck type defined for {type}'

class DECK_TYPE(Enum):
    STANDARD = 0
