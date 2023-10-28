import env.card_combat_env as card_combat_env
from pettingzoo.test import api_test

env = card_combat_env.env(render_mode="human")
api_test(env, num_cycles=100, verbose_progress=True)