"""Length units: the project's canonical unit, and reading STEP's (#170).

Why this module exists
----------------------
A scale error is the classic silent simulation bug. Nothing about a grain
imported 1000x too large is *invalid* -- it burns, converges, and produces
smooth plots, with a burnout time three orders of magnitude wrong and no
symptom that says so. Unlike a malformed mesh there is no exception to catch,
so the only defence is to establish the unit at import time and never guess.

The canonical unit
------------------
**Metres.** Everything inside the package -- coordinates, grid spacing ``h``,
burn rates, the Vieille coefficient ``a`` -- is metres and seconds. There is
exactly one place inches appear, and it is the *display* layer: the
measurements panel defaults to inches because experimental motor work is
dimensioned that way. That is a formatting choice applied on the way out; no
stored number is ever in inches.

(An older ``GrainGeometry`` docstring claimed inches were the project default.
It never matched the code -- every test, example and default value is metric --
and it has been corrected.)

What STEP declares, and what the loader does with it
----------------------------------------------------
STEP files state their length unit explicitly, which makes CAD the *only*
import path where the unit is knowable rather than assumed. The declaration
lives in the entity graph rather than the header::

    #132=(GEOMETRIC_REPRESENTATION_CONTEXT(3)
          GLOBAL_UNIT_ASSIGNED_CONTEXT((#136,#140,#141)) ...);
    #136=(CONVERSION_BASED_UNIT('inch',#139) LENGTH_UNIT() NAMED_UNIT(#134));
    #139=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#137);
    #137=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));

So resolving "what unit is this file in" means following references: the
context names a length unit, which is either an SI unit with a prefix, or a
conversion-based unit defined as a multiple of one. :func:`step_length_unit`
walks exactly that chain.

**The conversion itself is not ours to do.** OpenCASCADE (via ``cascadio``)
reads the declaration and emits metres, so a STEP grain arrives already scaled
-- measured, not assumed: ``BATES.stp`` declares inches and its 1 in bore
imports as 0.0254, and the same file re-declared as millimetres imports as
0.001. This module's job is therefore to *report* the declared unit, so the UI
can show it and the user can confirm it, and to fail loudly when a file
declares nothing. ``tests/test_units.py`` pins the metres-out behaviour so a
converter upgrade that changes convention breaks a test rather than a motor.

Mesh formats declare nothing
----------------------------
STL has no unit field at all, and OBJ/PLY none that tooling honours. For those
the numbers are taken as-is and the caller may state a unit explicitly via
``load_mesh(..., assume_units=...)``. This is why a STEP export is the better
input when the choice exists.
"""

from __future__ import annotations

import re
from pathlib import Path

#: The one unit the package computes in. See the module docstring.
CANONICAL_LENGTH_UNIT = "m"

#: Length units the loader accepts by name, as metres per unit.
LENGTH_UNITS: dict[str, float] = {
    "m": 1.0,
    "metre": 1.0,
    "meter": 1.0,
    "cm": 1e-2,
    "centimetre": 1e-2,
    "centimeter": 1e-2,
    "mm": 1e-3,
    "millimetre": 1e-3,
    "millimeter": 1e-3,
    "in": 2.54e-2,
    "inch": 2.54e-2,
    "ft": 0.3048,
    "foot": 0.3048,
}

#: SI prefixes as they are spelled in STEP's ``SI_UNIT`` entity.
SI_PREFIXES: dict[str, float] = {
    "EXA": 1e18,
    "PETA": 1e15,
    "TERA": 1e12,
    "GIGA": 1e9,
    "MEGA": 1e6,
    "KILO": 1e3,
    "HECTO": 1e2,
    "DECA": 1e1,
    "DECI": 1e-1,
    "CENTI": 1e-2,
    "MILLI": 1e-3,
    "MICRO": 1e-6,
    "NANO": 1e-9,
    "PICO": 1e-12,
    "FEMTO": 1e-15,
    "ATTO": 1e-18,
}

_ENTITY = re.compile(r"#(\d+)\s*=\s*(.*)", re.DOTALL)
_REF = re.compile(r"#(\d+)")
_SI_UNIT = re.compile(r"SI_UNIT\s*\(\s*(\.(\w+)\.|\$)\s*,\s*\.(\w+)\.\s*\)")
_CONVERSION = re.compile(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']*)'\s*,\s*#(\d+)")
_MEASURE_WITH_UNIT = re.compile(
    r"LENGTH_MEASURE_WITH_UNIT\s*\(\s*LENGTH_MEASURE\s*\(\s*([-+0-9.eE]+)\s*\)\s*,"
    r"\s*#(\d+)"
)
_GLOBAL_UNITS = re.compile(r"GLOBAL_UNIT_ASSIGNED_CONTEXT\s*\(\s*\(([^)]*)\)")


class UnitError(ValueError):
    """Raised when a length unit cannot be determined or is not recognised."""


def to_metres(units: str) -> float:
    """Metres per unit for a unit named in :data:`LENGTH_UNITS`.

    Raises
    ------
    UnitError
        If the name is not one this project accepts. Rejecting an unknown name
        is the point: silently defaulting is how scale errors get in.
    """
    key = str(units).strip().lower()
    if key not in LENGTH_UNITS:
        known = ", ".join(sorted(LENGTH_UNITS))
        raise UnitError(f"unknown length unit '{units}'. Known units: {known}")
    return LENGTH_UNITS[key]


def _entities(text: str) -> dict[int, str]:
    """Map ``#id -> entity body`` for the DATA section of a STEP file.

    STEP statements end at a semicolon and may wrap across any number of lines
    -- a real file breaks a single entity mid-token -- so the split is on ``;``
    and never on newlines.
    """
    out: dict[int, str] = {}
    for statement in text.split(";"):
        match = _ENTITY.match(statement.strip())
        if match:
            out[int(match.group(1))] = " ".join(match.group(2).split())
    return out


def _resolve_length(entity_id: int, entities: dict[int, str], depth: int = 0):
    """Metres per unit for a STEP length-unit entity, following references.

    Returns ``(name, metres_per_unit)`` or ``None`` when the entity is not a
    length unit. ``depth`` guards against a reference cycle in a damaged file,
    which would otherwise recurse forever.
    """
    if depth > 8 or entity_id not in entities:
        return None

    body = entities[entity_id]

    conversion = _CONVERSION.search(body)
    if conversion:
        # e.g. CONVERSION_BASED_UNIT('inch', #139) -- #139 says how much of
        # some other unit one inch is, so the factor is that measure times the
        # unit it is measured in.
        name = conversion.group(1)
        measure_body = entities.get(int(conversion.group(2)), "")
        measure = _MEASURE_WITH_UNIT.search(measure_body)
        if not measure:
            return None
        base = _resolve_length(int(measure.group(2)), entities, depth + 1)
        if base is None:
            return None
        return name.strip().lower(), float(measure.group(1)) * base[1]

    si = _SI_UNIT.search(body)
    if si and si.group(3).upper() == "METRE":
        prefix = si.group(2)
        factor = SI_PREFIXES.get((prefix or "").upper(), 1.0) if prefix else 1.0
        name = {1.0: "m", 1e-3: "mm", 1e-2: "cm"}.get(factor, f"{factor:g} m")
        return name, factor

    return None


def step_length_unit(path: str | Path) -> tuple[str, float] | None:
    """The length unit a STEP file declares, as ``(name, metres_per_unit)``.

    Returns ``None`` when the file declares no length unit at all -- which is
    the caller's cue to refuse the file rather than assume one.

    The unit is read from the *geometric* representation context. A file can
    carry several contexts (product structure, drawing annotations), so this
    takes the first one that resolves to a length, which in practice is the
    context the solid geometry itself is expressed in.

    This is a text scan, not a full STEP parser. That is deliberate: the unit
    declaration is a small, stable, well-specified corner of AP203/AP214, and
    reading it needs neither OpenCASCADE loaded nor the geometry tessellated --
    it can run before an expensive import to tell the user what they are about
    to get.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    entities = _entities(text)

    for body in entities.values():
        context = _GLOBAL_UNITS.search(body)
        if not context:
            continue
        for ref in _REF.findall(context.group(1)):
            resolved = _resolve_length(int(ref), entities)
            if resolved is not None:
                return resolved

    # Some exporters attach the units to a bare NAMED_UNIT set with no
    # representation context. Fall back to any length unit in the file.
    for entity_id in sorted(entities):
        if "LENGTH_UNIT" in entities[entity_id]:
            resolved = _resolve_length(entity_id, entities)
            if resolved is not None:
                return resolved

    return None
