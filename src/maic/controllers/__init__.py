REGISTRY = {}

from maic.controllers.basic_controller import BasicMAC
from maic.controllers.maic_controller import MAICMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['maic_mac'] = MAICMAC
