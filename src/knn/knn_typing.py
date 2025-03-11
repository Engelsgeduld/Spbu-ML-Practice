from typing import Sequence

import numpy as np

PointType = Sequence | np._typing.NDArray
PointsContainer = Sequence[PointType] | np._typing.NDArray
