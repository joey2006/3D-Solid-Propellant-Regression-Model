"""Unit conversion for every quantity the app displays (#154).

The engine computes in SI throughout -- metres, kilograms, seconds, pascals --
and conversion happens **only at the display edge**. That is the whole design:
no stored number is ever in inches or psi, so a unit mistake can only ever be a
display mistake, never a physics one. It also means switching units cannot
change a result, only how it is written down.

Why a registry rather than scattered constants
----------------------------------------------
The measurements panel already converted lengths by hand, the 3D view had its
own copy of the inch factor, and the phi tab had a third. Three copies of
``0.0254`` is three chances to get one wrong, and the failure is silent -- a
number that looks plausible and is off by a factor. One table, one lookup.

Temperature is the awkward one
------------------------------
Every other quantity here is a pure scale factor, so conversion is a
multiplication. Temperature has an offset as well (0 degrees Celsius is not
zero kelvin), so it cannot be expressed as a factor and is handled separately.
Rankine is included because it is what US thermochemistry tables use.

Burn rate deserves a note
-------------------------
``mm/s`` and ``in/s`` are the conventional units, and the numbers are small --
a composite runs about 5 mm/s at 1 MPa. Displaying that in m/s gives 0.005,
which reads as a rounding error rather than a rate, so the display default is
mm/s even in an otherwise metric setup.
"""

from __future__ import annotations

#: Every quantity the UI knows how to display, and the SI unit it is stored in.
CANONICAL: dict[str, str] = {
    "length": "m",
    "pressure": "Pa",
    "force": "N",
    "mass": "kg",
    "density": "kg/m3",
    "velocity": "m/s",
    "burn_rate": "m/s",
    "volume": "m3",
    "area": "m2",
    "temperature": "K",
    "impulse": "N*s",
}

#: ``{quantity: {unit: how many SI units one of it is}}``.
#: Multiply by the factor to reach SI; divide by it to display.
FACTORS: dict[str, dict[str, float]] = {
    "length": {
        "m": 1.0, "cm": 1e-2, "mm": 1e-3,
        "in": 2.54e-2, "ft": 0.3048,
    },
    "pressure": {
        "Pa": 1.0, "kPa": 1e3, "MPa": 1e6,
        "bar": 1e5, "atm": 101325.0, "psi": 6894.757293168361,
    },
    "force": {
        "N": 1.0, "kN": 1e3, "lbf": 4.4482216152605,
    },
    "mass": {
        "kg": 1.0, "g": 1e-3, "lb": 0.45359237, "oz": 0.028349523125,
    },
    "density": {
        # lb/in3 is the convention in US propellant data sheets.
        "kg/m3": 1.0, "g/cm3": 1000.0, "lb/in3": 27679.904710203125,
    },
    "velocity": {
        "m/s": 1.0, "ft/s": 0.3048, "mph": 0.44704,
    },
    "burn_rate": {
        "m/s": 1.0, "mm/s": 1e-3, "in/s": 2.54e-2,
    },
    "volume": {
        "m3": 1.0, "cm3": 1e-6, "L": 1e-3, "in3": 1.6387064e-5,
    },
    "area": {
        "m2": 1.0, "cm2": 1e-4, "mm2": 1e-6, "in2": 6.4516e-4,
    },
    "impulse": {
        "N*s": 1.0, "kN*s": 1e3, "lbf*s": 4.4482216152605,
    },
}

#: What each quantity is shown in, per unit system. Chosen for how the numbers
#: read at motor scale rather than for strict system purity -- see the module
#: docstring on burn rate, and note metric pressure is MPa because that is the
#: convention the Vieille coefficient follows.
SYSTEMS: dict[str, dict[str, str]] = {
    "metric": {
        "length": "mm", "pressure": "MPa", "force": "N", "mass": "kg",
        "density": "kg/m3", "velocity": "m/s", "burn_rate": "mm/s",
        "volume": "cm3", "area": "mm2", "temperature": "C", "impulse": "N*s",
    },
    "imperial": {
        "length": "in", "pressure": "psi", "force": "lbf", "mass": "lb",
        "density": "lb/in3", "velocity": "ft/s", "burn_rate": "in/s",
        "volume": "in3", "area": "in2", "temperature": "F", "impulse": "lbf*s",
    },
}

#: Temperature units, as ``(scale, offset)`` where ``K = value * scale + offset``.
TEMPERATURE: dict[str, tuple[float, float]] = {
    "K": (1.0, 0.0),
    "C": (1.0, 273.15),
    "F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    "R": (5.0 / 9.0, 0.0),
}


class UnitError(ValueError):
    """Raised when a unit or quantity is not one this project knows."""


def units_for(quantity: str, system: str) -> str:
    """The unit ``quantity`` is displayed in under ``system``."""
    if system not in SYSTEMS:
        raise UnitError(
            f"unknown unit system {system!r}; expected one of "
            f"{', '.join(sorted(SYSTEMS))}"
        )
    try:
        return SYSTEMS[system][quantity]
    except KeyError:
        raise UnitError(f"unknown quantity {quantity!r}") from None


def factor(quantity: str, unit: str) -> float:
    """How many SI units one ``unit`` of ``quantity`` is."""
    if quantity == "temperature":
        raise UnitError(
            "temperature has an offset as well as a scale; use to_si / "
            "from_si rather than a factor"
        )
    try:
        table = FACTORS[quantity]
    except KeyError:
        raise UnitError(f"unknown quantity {quantity!r}") from None
    try:
        return table[unit]
    except KeyError:
        known = ", ".join(sorted(table))
        raise UnitError(
            f"unknown {quantity} unit {unit!r}; known units: {known}"
        ) from None


def to_si(value: float, quantity: str, unit: str) -> float:
    """Convert a displayed value into the engine's units."""
    if quantity == "temperature":
        scale, offset = _temperature(unit)
        return value * scale + offset
    return value * factor(quantity, unit)


def from_si(value: float, quantity: str, unit: str) -> float:
    """Convert an engine value into what should be displayed."""
    if quantity == "temperature":
        scale, offset = _temperature(unit)
        return (value - offset) / scale
    return value / factor(quantity, unit)


def convert(value: float, quantity: str, source: str, target: str) -> float:
    """Convert between two display units of the same quantity."""
    return from_si(to_si(value, quantity, source), quantity, target)


def format_value(
    value: float | None,
    quantity: str,
    system: str,
    decimals: int | None = None,
    with_unit: bool = True,
) -> str:
    """Render an SI value in ``system``, with its unit.

    ``None`` renders as ``"--"`` rather than raising: a missing measurement is
    a normal state -- an unclosed mesh has no volume, a grain with no bore has
    no web -- and the display layer should say so rather than fabricate a zero.
    """
    if value is None:
        return "--"
    unit = units_for(quantity, system)
    shown = from_si(float(value), quantity, unit)
    if decimals is None:
        decimals = _default_decimals(quantity, unit)
    text = f"{shown:.{decimals}f}"
    return f"{text} {_pretty(unit)}" if with_unit else text


def _temperature(unit: str) -> tuple[float, float]:
    try:
        return TEMPERATURE[unit]
    except KeyError:
        known = ", ".join(sorted(TEMPERATURE))
        raise UnitError(
            f"unknown temperature unit {unit!r}; known units: {known}"
        ) from None


def _default_decimals(quantity: str, unit: str) -> int:
    """Enough precision to be useful, not so much it shows facet noise.

    Three decimals of an inch is 25 microns -- finer than any real tolerance,
    and the point at which a tessellated circle's faceting would start to show
    if it were not measured at vertices.
    """
    if quantity == "length":
        return 3 if unit == "in" else 2
    if quantity in ("volume", "area", "burn_rate"):
        return 3
    if quantity == "density":
        return 4 if unit == "lb/in3" else 1
    if quantity == "pressure":
        return 3 if unit == "MPa" else 1
    if quantity == "mass":
        # A hobby motor's grain is tens of grams to a few kilos, so two
        # decimals of a kilogram rounds a 43 g grain to "0.04".
        return 3
    return 2


#: Units whose ASCII spelling is not what a person would write.
_PRETTY = {
    "kg/m3": "kg/m³", "g/cm3": "g/cm³", "lb/in3": "lb/in³",
    "m3": "m³", "cm3": "cm³", "in3": "in³",
    "m2": "m²", "cm2": "cm²", "mm2": "mm²", "in2": "in²",
    "N*s": "N·s", "kN*s": "kN·s", "lbf*s": "lbf·s",
    "C": "°C", "F": "°F", "R": "°R",
}


def _pretty(unit: str) -> str:
    return _PRETTY.get(unit, unit)
