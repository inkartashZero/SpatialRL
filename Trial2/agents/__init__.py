from .td3 import TD3Agent
from .sac import SACAgent

CONTINUOUS_REGISTRY = {
    "td3": TD3Agent,
    "sac": SACAgent
}