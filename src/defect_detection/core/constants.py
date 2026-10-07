"""Single source of truth for class names and label ids.

The positive class is ``defective`` (label 1): precision/recall/F1, PR-AUC and the operating
threshold are all defined with respect to finding defects.
"""

from typing import Final

NORMAL_LABEL: Final = 0
DEFECTIVE_LABEL: Final = 1

# Index in this tuple == label id == model output column.
CLASS_NAMES: Final[tuple[str, str]] = ("normal", "defective")
POSITIVE_CLASS: Final = CLASS_NAMES[DEFECTIVE_LABEL]
