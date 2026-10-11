"""Primary-signal zoo (SPEC §4). Importing the package registers every primary."""

from primaries import mechanism, ml_xgb, region, rules  # noqa: F401  (registration)
from primaries.base import PRIMARY_COLUMNS, REGISTRY, check_signal, make_primary, side

__all__ = ["PRIMARY_COLUMNS", "REGISTRY", "check_signal", "make_primary", "side"]
