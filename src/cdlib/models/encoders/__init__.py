from .registry import ENCODER_REGISTRY

# Import submodules for their registration side-effects: each module
# registers its encoder key(s) into ENCODER_REGISTRY at import time.
from . import resnet  # noqa: F401,E402
from . import efficientnet  # noqa: F401,E402

__all__ = ["ENCODER_REGISTRY"]
