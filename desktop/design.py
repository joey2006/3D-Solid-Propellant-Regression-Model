"""Saving and reloading a motor design (#155).

A design is everything the user chose: which grain file, how it is labelled,
what propellant, what grid, what units. It is *not* results -- those are
reproducible from the design and would bloat the file enormously (a 256³ field
is 67 million floats). Exporting results is #149, and deliberately separate.

Why JSON and not pickle
-----------------------
A design file is something a person may want to read, diff, email to a
teammate, or check into a repository next to the CAD it refers to. Pickle is
none of those things, and it executes arbitrary code on load, which is an
unreasonable thing to ask of someone opening a motor design they were sent.

Why the mesh is referenced, not embedded
----------------------------------------
The file stores a *path* to the geometry, not the geometry itself. A STEP file
can be megabytes, and embedding it would mean a design silently diverging from
the CAD it was drawn from -- edit the part, and the design would keep simulating
the old shape with no indication. A path stays honest: the design always refers
to whatever the file says now, and says so plainly when the file has moved.

Both an absolute and a repo-relative path are stored, and load tries the
relative one first, so a design travels with a project folder that has been
moved or shared.

Versioning
----------
``FORMAT_VERSION`` is written into every file. Nothing reads it yet, but a
design saved today will outlive several changes to the panels, and a file that
cannot say which schema it follows can only be guessed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Bumped when the schema changes in a way an older reader could misinterpret.
FORMAT_VERSION = 1

#: Suffix for design files, and what the file dialogs filter on.
SUFFIX = ".srmd"


class DesignError(RuntimeError):
    """Raised when a design file cannot be read or is not a design file."""


@dataclass
class Design:
    """Everything a user chose, in one serialisable object.

    Defaults match the application's own defaults, so a design file that omits
    a field -- one written by an older version, say -- loads as though the user
    had left that control alone rather than failing.
    """

    geometry_path: str | None = None
    geometry_relative: str | None = None
    ends: str = "inhibited"

    resolution: int = 64
    margin: float = 0.05
    device: str = "cuda"

    propellant_name: str = "APCP (generic composite)"
    burn_coefficient: float = 0.005
    pressure_exponent: float = 0.35
    density: float = 1750.0

    max_time: float = 10.0
    cfl: float = 0.4
    reinit_every: int = 5

    units: str = "imperial"
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> str:
        payload = {
            "format": "srm-burnback-design",
            "version": FORMAT_VERSION,
            **{k: v for k, v in self.__dict__.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "Design":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DesignError(f"this is not a valid design file: {exc}") from exc
        if not isinstance(payload, dict):
            raise DesignError("a design file must contain a JSON object")
        if payload.get("format") != "srm-burnback-design":
            raise DesignError(
                "this JSON file is not a motor design — it has no "
                "'srm-burnback-design' marker."
            )
        # Unknown keys are dropped rather than raising: a file written by a
        # newer version should still open here, minus whatever is new.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(SUFFIX)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Design":
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DesignError(f"could not read '{path.name}': {exc}") from exc
        return cls.from_json(text)

    # -- geometry ----------------------------------------------------------

    def set_geometry(self, path: str | Path | None, base: Path | None = None) -> None:
        """Record the grain file, both absolutely and relative to ``base``."""
        if path is None:
            self.geometry_path = None
            self.geometry_relative = None
            return
        resolved = Path(path).resolve()
        self.geometry_path = str(resolved)
        self.geometry_relative = None
        if base is not None:
            try:
                self.geometry_relative = str(
                    resolved.relative_to(Path(base).resolve())
                )
            except ValueError:
                # Not under `base` at all; the absolute path is all there is.
                self.geometry_relative = None

    def resolve_geometry(self, base: Path | None = None) -> Path | None:
        """The grain file, preferring a relative path so folders can move.

        Returns ``None`` when nothing was recorded or the file is gone. The
        caller decides what to say about it -- a design whose geometry has been
        moved is still a perfectly good design, and should open with everything
        else intact rather than refusing.
        """
        if base is not None and self.geometry_relative:
            candidate = Path(base) / self.geometry_relative
            if candidate.is_file():
                return candidate
        if self.geometry_path:
            candidate = Path(self.geometry_path)
            if candidate.is_file():
                return candidate
        return None
