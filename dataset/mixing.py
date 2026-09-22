"""Public source-mixture policy and bounded epoch quotas."""

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext


@dataclass(frozen=True)
class MixingConfig:
    mode: str = "balanced"
    size_power: float | None = None

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("mixing must be a mapping")
        unknown = set(value) - {"mode", "size_power"}
        if unknown:
            raise ValueError(f"unknown mixing option(s): {sorted(unknown)}")
        mode = value.get("mode", "balanced")
        if not isinstance(mode, str) or mode not in {
            "balanced", "weighted", "natural", "tempered"
        }:
            raise ValueError("mixing.mode must be balanced, weighted, natural, or tempered")
        if mode != "tempered":
            if "size_power" in value:
                raise ValueError("mixing.size_power is only valid in tempered mode")
            return cls(mode)
        power = value.get("size_power")
        if isinstance(power, bool) or not isinstance(power, (int, float)):
            raise TypeError("tempered mixing requires a numeric size_power in [0, 1]")
        if not math.isfinite(power) or not 0 <= power <= 1:
            raise ValueError("mixing.size_power must be finite and in [0, 1]")
        if power == 0:
            return cls("balanced")
        if power == 1:
            return cls("natural")
        return cls(mode, float(power))

    def epoch_quotas(self, counts):
        """Return maximal no-repeat size-weighted quotas, or natural counts."""
        if not counts or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("size-based mixing requires known non-negative row counts")
        if not 0 < sum(counts) < 2**63:
            raise ValueError("size-based mixing requires between 1 and 2**63-1 rows")
        if self.mode == "natural":
            return tuple(counts)
        minimum = min(counts)
        if minimum == 0:
            raise ValueError("tempered mixing cannot maintain its ratio with an empty source")
        # Decimal arithmetic avoids flooring an exact limiting quota one row low.
        # Counts have at most 19 digits; the extra precision protects integer ties.
        with localcontext() as context:
            context.prec = 60
            power = Decimal(str(self.size_power))
            quotas = []
            for count in counts:
                if count == minimum:
                    quotas.append(minimum)
                    continue
                value = Decimal(minimum) * (Decimal(count) / minimum) ** power
                nearest = value.to_integral_value()
                if abs(value - nearest) < Decimal("1e-35"):
                    value = nearest
                quotas.append(min(count, int(value.to_integral_value(rounding=ROUND_FLOOR))))
        return tuple(quotas)
