from .td3 import TD3Agent
from .sac import SACAgent
from .FA import QLearningFA, SarsaFA, SarsaLambdaFA
from .ddpg import DDPGAgent
from .vpg import VPGAgent
from .ppo import PPOAgent
from .a2c import A2CAgent

CONTINUOUS_REGISTRY = {
    "td3": TD3Agent,
    "sac": SACAgent,
    "q_fa": QLearningFA,
    "sarsa_fa": SarsaFA,
    "sarsa_lambda_fa": SarsaLambdaFA,
    "ddpg": DDPGAgent,
    "vpg": VPGAgent,
    "ppo": PPOAgent,
    "a2c": A2CAgent,
}