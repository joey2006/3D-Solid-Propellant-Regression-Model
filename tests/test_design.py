"""Saved designs (#155) and the propellant library (#152).

Both are about a number surviving a round trip without being quietly altered:
a design written today must reopen as the same motor, and picking "KNSB" must
put KNSB's coefficients in rather than something adjacent to them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from desktop.design import SUFFIX, Design, DesignError
from srm_burnback.propellants import (
    CUSTOM,
    LIBRARY,
    Propellant,
    by_name,
    matches,
)


class TestPropellantLibrary:
    def test_every_entry_is_physically_sensible(self):
        for propellant in LIBRARY:
            # n >= 1 is unstable: burn rate would rise faster than pressure can
            # relieve it, and the motor runs away.
            assert 0.0 < propellant.n < 1.0, propellant.name
            assert propellant.a > 0.0, propellant.name
            # Solid propellants sit well above water and well below metal.
            assert 1200.0 < propellant.density < 2200.0, propellant.name

    def test_every_entry_says_where_it_came_from(self):
        # A burn-rate coefficient with no provenance cannot be checked, and
        # these are numbers people build motors from.
        for propellant in LIBRARY:
            assert propellant.source.strip(), propellant.name

    def test_names_are_unique(self):
        names = [p.name for p in LIBRARY]
        assert len(names) == len(set(names))

    def test_burn_rate_follows_the_mpa_convention(self):
        """``a`` is the rate at 1 MPa, in m/s."""
        composite = by_name("APCP (generic composite)")
        assert composite.burn_rate(1.0) == pytest.approx(composite.a)
        # The familiar 5 mm/s at 1 MPa.
        assert composite.burn_rate(1.0) == pytest.approx(0.005)
        # Rate rises with pressure, but sub-linearly since n < 1.
        assert composite.burn_rate(5.0) > composite.burn_rate(1.0)
        assert composite.burn_rate(5.0) < 5 * composite.burn_rate(1.0)

    def test_matching_detects_an_edited_coefficient(self):
        propellant = by_name("KNSB 65/35 (KNO3 / sorbitol)")
        assert matches(
            propellant, propellant.a, propellant.n, propellant.density
        )
        # One digit changed is no longer that propellant, and the UI must stop
        # calling it by name.
        assert not matches(
            propellant, propellant.a * 1.01, propellant.n, propellant.density
        )

    def test_unknown_names_resolve_to_nothing(self):
        assert by_name("Unobtainium") is None
        assert by_name(CUSTOM) is None

    def test_round_trips_through_a_dict(self):
        original = LIBRARY[1]
        assert Propellant.from_dict(original.as_dict()) == original


class TestDesignRoundTrip:
    @staticmethod
    def _design():
        return Design(
            resolution=128,
            margin=0.08,
            ends="inhibited",
            propellant_name="KNSU 65/35 (KNO3 / sucrose)",
            burn_coefficient=0.00826,
            pressure_exponent=0.319,
            density=1889.0,
            max_time=7.5,
            cfl=0.35,
            units="metric",
        )

    def test_a_saved_design_reopens_identical(self, tmp_path):
        original = self._design()
        written = original.save(tmp_path / "motor")
        assert written.suffix == SUFFIX
        assert Design.load(written) == original

    def test_the_file_is_readable_json(self, tmp_path):
        """A design should be diffable and emailable, hence JSON not pickle."""
        written = self._design().save(tmp_path / "motor")
        payload = json.loads(written.read_text(encoding="utf-8"))
        assert payload["format"] == "srm-burnback-design"
        assert payload["version"] >= 1
        assert payload["resolution"] == 128

    def test_a_file_from_a_newer_version_still_opens(self, tmp_path):
        """Unknown keys are dropped rather than fatal.

        A design written by a later build should lose only what is new, not
        refuse to open at all.
        """
        written = self._design().save(tmp_path / "motor")
        payload = json.loads(written.read_text(encoding="utf-8"))
        payload["nozzle_throat_area"] = 1.5
        (tmp_path / "future.srmd").write_text(json.dumps(payload), encoding="utf-8")

        loaded = Design.load(tmp_path / "future.srmd")
        assert loaded.resolution == 128

    def test_a_missing_field_falls_back_to_the_default(self, tmp_path):
        payload = {"format": "srm-burnback-design", "version": 1, "resolution": 96}
        (tmp_path / "sparse.srmd").write_text(json.dumps(payload), encoding="utf-8")

        loaded = Design.load(tmp_path / "sparse.srmd")
        assert loaded.resolution == 96
        assert loaded.density == 1750.0     # untouched default

    def test_arbitrary_json_is_refused(self, tmp_path):
        (tmp_path / "notes.srmd").write_text('{"shopping": ["eggs"]}', encoding="utf-8")
        with pytest.raises(DesignError, match="not a motor design"):
            Design.load(tmp_path / "notes.srmd")

    def test_malformed_json_is_refused_clearly(self, tmp_path):
        (tmp_path / "broken.srmd").write_text("{oh no", encoding="utf-8")
        with pytest.raises(DesignError, match="not a valid design file"):
            Design.load(tmp_path / "broken.srmd")

    def test_a_missing_file_is_refused_clearly(self, tmp_path):
        with pytest.raises(DesignError, match="could not read"):
            Design.load(tmp_path / "nothing-here.srmd")


class TestGeometryReference:
    """The grain file is referenced, not embedded, so it must survive moving."""

    def test_a_relative_path_is_recorded_alongside_the_absolute_one(self, tmp_path):
        grain = tmp_path / "parts" / "grain.stl"
        grain.parent.mkdir()
        grain.write_text("solid\nendsolid\n", encoding="utf-8")

        design = Design()
        design.set_geometry(grain, base=tmp_path)
        assert design.geometry_relative == str(Path("parts") / "grain.stl")
        assert Path(design.geometry_path).is_absolute()

    def test_a_moved_project_folder_still_finds_its_grain(self, tmp_path):
        """The point of storing a relative path.

        The absolute path is stale after a move -- or on someone else's
        machine -- so resolution prefers the relative one.
        """
        original, moved = tmp_path / "before", tmp_path / "after"
        (original / "parts").mkdir(parents=True)
        (original / "parts" / "grain.stl").write_text("solid\n", encoding="utf-8")

        design = Design()
        design.set_geometry(original / "parts" / "grain.stl", base=original)
        original.rename(moved)

        assert design.resolve_geometry(base=original) is None
        assert design.resolve_geometry(base=moved) == moved / "parts" / "grain.stl"

    def test_a_grain_outside_the_design_folder_keeps_its_absolute_path(
        self, tmp_path
    ):
        elsewhere = tmp_path / "elsewhere" / "grain.stl"
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_text("solid\n", encoding="utf-8")
        base = tmp_path / "designs"
        base.mkdir()

        design = Design()
        design.set_geometry(elsewhere, base=base)
        assert design.geometry_relative is None
        assert design.resolve_geometry(base=base) == elsewhere

    def test_a_deleted_grain_resolves_to_nothing_rather_than_raising(self, tmp_path):
        """A design whose geometry is gone is still a good design.

        Everything else should restore, and the caller reports the missing
        file, rather than the whole design refusing to open.
        """
        design = Design()
        design.set_geometry(tmp_path / "vanished.stl", base=tmp_path)
        assert design.resolve_geometry(base=tmp_path) is None

    def test_no_geometry_is_a_valid_design(self):
        design = Design()
        design.set_geometry(None)
        assert design.geometry_path is None
        assert design.resolve_geometry() is None
