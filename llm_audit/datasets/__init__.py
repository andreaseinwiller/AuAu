from .a import A
from .aa import AA
from .act import ACT
from .apc import APC
from .asc import ASC
from .base import AgreementDataset
from .bdw import BDW
from .bfi10 import BFI10
from .csm import CSM
from .cw import CW
from .d import D
from .dw import DW
from .f import F
from .ksa3 import KSA3
from .las import LAS
from .pi import PI
from .pisd import PISD
from .rwa import RWA
from .rwa3d import RWA3D
from .sdo7 import SDO7
from .vsa import VSA

DATASETS: dict[str, type[AgreementDataset]] = {
    "A": A,
    "AA": AA,
    "ACT": ACT,
    "APC": APC,
    "ASC": ASC,
    "BDW": BDW,
    "BFI10": BFI10,
    "CSM": CSM,
    "CW": CW,
    "D": D,
    "DW": DW,
    "F": F,
    "KSA3": KSA3,
    "LAS": LAS,
    "PI": PI,
    "PISD": PISD,
    "RWA": RWA,
    "RWA3D": RWA3D,
    "SDO7": SDO7,
    "VSA": VSA,
}
