from enum import Enum


class ColorPalette(str, Enum):
    TAB20 = "tab20"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    INFERNO = "inferno"
    CIVIDIS = "cividis"
    RAINBOW = "gist_rainbow"
