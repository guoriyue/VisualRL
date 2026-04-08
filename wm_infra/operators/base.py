"""Shared operator abstractions for model invocation."""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any


class OperatorFamily(str, Enum):
    DYNAMICS = "dynamics"
    GENERATION = "generation"


class ModelOperator(ABC):
    """Base class for model invocation adapters."""

    operator_name = "model-operator"
    family: OperatorFamily

    def runtime_descriptor(self) -> dict[str, Any]:
        """Return serving-runtime metadata shared by operator call sites."""

        return {}

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.operator_name,
            "family": self.family.value,
            **self.runtime_descriptor(),
        }
