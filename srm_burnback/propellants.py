"""Named propellants, so coefficients are chosen rather than retyped (#152).

A propellant is three numbers -- the Vieille coefficient ``a``, the pressure
exponent ``n``, and density -- and typing them from memory every session is
both tedious and a good way to transpose a digit into a motor design. Naming
them makes the choice reviewable: "KNSB" is checkable at a glance in a way that
``a = 0.00513`` is not.

**These values are starting points, not characterisations of your propellant.**
Burn rate depends on the specific batch: oxidiser particle size, binder ratio,
cure, additives, even mixing. Published coefficients are what to simulate with
before you have your own data, and what to replace the moment you do. A static
test firing is the only thing that tells you how *your* propellant burns, and
the difference is not small -- it moves chamber pressure by the power
``1 / (1 - n)``, so a 10% error in ``a`` is a ~15% error in pressure at
``n = 0.35``.

Units
-----
``a`` follows the **MPa convention** used throughout this package: with
pressure supplied in MPa, ``r = a * P^n`` gives metres per second. So the
familiar "5 mm/s at 1 MPa" composite is ``a = 0.005``. See
:mod:`srm_burnback.physics.vieille`.

Why KNDX is not here
--------------------
Potassium nitrate / dextrose is common in amateur work and is deliberately
omitted, because its burn rate is not well described by one ``(a, n)`` pair:
the exponent changes sharply across pressure ranges, and over one interval it
is even *negative*. A single fitted pair would look authoritative and be wrong
across most of a real burn. Piecewise burn-rate laws are their own feature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Propellant:
    """A propellant's burn-rate law and density.

    Parameters
    ----------
    name:
        How it is shown and stored.
    a:
        Vieille coefficient, MPa convention, giving m/s.
    n:
        Pressure exponent. Must be below 1 for a stable motor.
    density:
        kg/m^3.
    source:
        Where the numbers came from, so they can be checked or challenged.
    """

    name: str
    a: float
    n: float
    density: float
    source: str = ""

    def burn_rate(self, pressure_mpa: float) -> float:
        """``r = a * P^n`` in m/s, with pressure in MPa."""
        return self.a * pressure_mpa**self.n

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Propellant":
        allowed = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**allowed)


#: Published reference propellants. Read the module docstring before trusting
#: any of them for a motor you intend to fire.
LIBRARY: tuple[Propellant, ...] = (
    Propellant(
        name="APCP (generic composite)",
        a=0.005,
        n=0.35,
        density=1750.0,
        source="Representative composite values; the package's own default.",
    ),
    Propellant(
        name="KNSB 65/35 (KNO3 / sorbitol)",
        a=0.00513,
        n=0.222,
        density=1841.0,
        source="Nakka, published sugar-propellant data. Verify by static test.",
    ),
    Propellant(
        name="KNSU 65/35 (KNO3 / sucrose)",
        a=0.00826,
        n=0.319,
        density=1889.0,
        source="Nakka, published sugar-propellant data. Verify by static test.",
    ),
)

#: Shown when nothing in the library is selected.
CUSTOM = "Custom"


def by_name(name: str) -> Propellant | None:
    """Look a propellant up in the built-in library."""
    for propellant in LIBRARY:
        if propellant.name == name:
            return propellant
    return None


def matches(propellant: Propellant, a: float, n: float, density: float) -> bool:
    """Whether the given values still correspond to ``propellant``.

    Used to decide when an edited set of coefficients has stopped being the
    named propellant and should read as "Custom" instead -- the label must
    never claim a name the numbers no longer match.
    """
    return (
        abs(propellant.a - a) < 1e-9
        and abs(propellant.n - n) < 1e-9
        and abs(propellant.density - density) < 1e-6
    )
