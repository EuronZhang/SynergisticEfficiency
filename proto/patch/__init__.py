# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .vpt import apply_patch as vpt
from .timm import apply_patch as timm
from .rand_drop import apply_patch as rand_drop

__all__ = ["vpt", "timm", "rand_drop"]
