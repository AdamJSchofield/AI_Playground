from enum import Enum

class Card:
    def __init__(self, cardType, energyCost, self_cast):
        self.cardType = cardType
        self.energyCost = energyCost
        self.damage = 0
        self.healing = 0
        self.self_cast = self_cast
    
    def __eq__(self, obj):
        return isinstance(obj, Card) and obj.cardType.value == self.cardType.value
    
class CardType(Enum):
    FIREBALL = 0
    FIREBLAST = 1
    HEAL = 2
    
class FireballCard(Card):

    def __init__(self):
        super().__init__(CardType.FIREBALL, 1, False)
        self.damage = 3

class FireBlastCard(Card):
    def __init__(self):
        super().__init__(CardType.FIREBLAST, 3, False)
        self.damage = 10

class HealCard(Card):
    def __init__(self):
        super().__init__(CardType.HEAL, 2, True)
        self.healing = 3

ALL_CARDS = [FireballCard(), FireBlastCard(), HealCard()]