"""Training and evaluation core for CARE-OPD.

The package intentionally does not depend on a particular agent runtime.  The
environment-facing contract lives in :mod:`care_opd.rollout`; training consumes
portable JSONL records emitted through that contract.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
