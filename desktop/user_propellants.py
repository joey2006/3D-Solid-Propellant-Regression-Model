"""Propellants the user measured themselves, saved between sessions (#152).

The built-in ``LIBRARY`` in :mod:`srm_burnback.propellants` is published
reference data: fixed, shipped with the app, and explicitly *not* anyone's
actual batch. The whole point of the warning attached to it is that a real
propellant has to be characterised by static test -- which means the numbers
that matter most to a user are precisely the ones the app had nowhere to put.
Editing the coefficients showed "Custom" and lost them at the next selection.

So user propellants live here, in a small JSON file beside the app's other
settings. They are kept deliberately separate from the built-in library rather
than merged into it:

* Published values must stay recognisable as published. If a user's "KNSB" and
  Nakka's "KNSB" were the same list, a name would no longer say where its
  numbers came from.
* A user file is user data. It is never rewritten by an upgrade, and shipping
  a new built-in propellant can never silently overwrite a measured one.

The file is a JSON list of objects, readable and hand-editable, for the same
reasons ``.srmd`` designs are (see :mod:`desktop.design`). A corrupt or
unreadable file is treated as an empty library rather than an error: losing
saved propellants is bad, but refusing to start the app is worse, and the file
is recoverable by hand precisely because it is JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from srm_burnback.propellants import LIBRARY, Propellant

#: Name of the store inside the app's data directory.
FILENAME = "propellants.json"


def store_path() -> Path:
    """Where the user's propellants are kept.

    ``AppDataLocation`` is the per-user, per-application directory Qt resolves
    for the platform -- ``%APPDATA%`` on Windows, ``~/.local/share`` on Linux.
    Using it rather than a path beside the code means the store survives
    reinstalling or moving the application, and never lands somewhere the user
    lacks permission to write.
    """
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    return Path(base) / FILENAME


def load() -> list[Propellant]:
    """Every propellant the user has saved, oldest first.

    Returns an empty list when the store is missing, unreadable or malformed.
    Entries that are individually broken are skipped rather than discarding the
    whole file, so one bad record cannot cost the user the rest.
    """
    path = store_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []

    propellants = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        try:
            propellants.append(Propellant.from_dict(entry))
        except (TypeError, ValueError):
            continue
    return propellants


def save_all(propellants: list[Propellant]) -> Path:
    """Write the whole store, creating the directory if it is not there yet."""
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [p.as_dict() for p in propellants]
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    return path


def save(propellant: Propellant) -> list[Propellant]:
    """Add or update one propellant by name, and return the new store.

    Saving under an existing user name replaces that entry rather than adding a
    duplicate -- re-saving after a second static test is the expected way to
    use this, and two entries with one name would make the combo box a lottery.
    """
    propellants = [p for p in load() if p.name != propellant.name]
    propellants.append(propellant)
    save_all(propellants)
    return propellants


def delete(name: str) -> list[Propellant]:
    """Remove a saved propellant by name, and return the new store."""
    propellants = [p for p in load() if p.name != name]
    save_all(propellants)
    return propellants


def is_builtin(name: str) -> bool:
    """Whether ``name`` belongs to the shipped reference library.

    Used to refuse saving over a published name, which would leave the user
    with an entry claiming to be Nakka's data while holding their own.
    """
    return any(p.name == name for p in LIBRARY)
