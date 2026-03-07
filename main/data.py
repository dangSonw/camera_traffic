from dataclasses import dataclass
from enum import Enum

class LightState(Enum):
    ERR = 0,
    GREEN = 1,
    RED = 2

from datetime import datetime

@dataclass
class TranPkg:
    quantity: int = 0
    timeRed: str = datetime.now().isoformat()
    timeGreen: str = datetime.now().isoformat()
