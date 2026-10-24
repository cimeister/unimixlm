# --------------------------------------------------------------------------------------
# Copyright 2021 by Oculy Authors, see git history for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# --------------------------------------------------------------------------------------
"""Plotting backends used to display plots.

"""
from .backend import Backend
from .resolver import BackendResolver

__all__ = ("Backend", "BackendResolver")