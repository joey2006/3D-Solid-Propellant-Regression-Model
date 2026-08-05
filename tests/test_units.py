"""Length-unit handling on import (#170).

A scale error produces no exception, no warning and no visibly wrong picture --
only a burn time off by orders of magnitude. There is nothing to notice at run
time, so the invariant has to be pinned by tests instead.

Two things are pinned here:

* our own STEP unit reader, against a real Inventor export; and
* **the converter's behaviour**, which we depend on but do not control.
  OpenCASCADE currently emits metres whatever the file declares. Nothing in our
  code would break visibly if a ``cascadio`` upgrade changed that convention --
  imports would simply come in 1000x off -- so the round-trip below exists to
  turn that into a failing test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from srm_burnback.geometry.import_mesh import MeshImportError, load_mesh, mesh_stats
from srm_burnback.geometry.units import (
    CANONICAL_LENGTH_UNIT,
    UnitError,
    step_length_unit,
    to_metres,
)

trimesh = pytest.importorskip("trimesh")

#: A real Inventor STEP export, tracked in the repo. It declares inches via a
#: CONVERSION_BASED_UNIT and measures 1 in across by 2 in long.
BATES_STEP = Path(__file__).resolve().parents[1] / "BATES.stp"
BATES_STEP_DIAMETER_M = 0.0254
BATES_STEP_LENGTH_M = 0.0508

cascadio = pytest.importorskip("cascadio")


def _redeclare_units(source: Path, target: Path, from_ref: str, to_ref: str) -> Path:
    """Copy a STEP file with its global length unit re-pointed.

    Changing only the unit *reference* leaves every coordinate untouched, which
    isolates the thing under test: identical geometry, different declaration,
    so any difference in the imported result is unit handling and nothing else.
    """
    text = source.read_text()
    marker = f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({from_ref},"
    assert marker in text, "fixture no longer declares units the expected way"
    target.write_text(text.replace(marker, f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({to_ref},"))
    return target


class TestUnitNames:
    def test_known_units_convert(self):
        assert to_metres("mm") == pytest.approx(1e-3)
        assert to_metres("in") == pytest.approx(0.0254)
        assert to_metres("m") == 1.0

    def test_case_and_whitespace_tolerated(self):
        assert to_metres(" MM ") == pytest.approx(1e-3)

    def test_unknown_unit_is_rejected_not_defaulted(self):
        # Defaulting silently is the failure mode this whole module exists to
        # prevent, so an unrecognised name must raise rather than pass through.
        with pytest.raises(UnitError):
            to_metres("furlong")


class TestStepUnitReader:
    def test_reads_conversion_based_inch(self):
        assert step_length_unit(BATES_STEP) == ("inch", pytest.approx(0.0254))

    def test_reads_si_millimetre(self, tmp_path):
        # #136 is the inch unit, #137 the millimetre one.
        mm = _redeclare_units(BATES_STEP, tmp_path / "mm.stp", "#136", "#137")
        assert step_length_unit(mm) == ("mm", pytest.approx(1e-3))

    def test_reads_si_centimetre(self, tmp_path):
        cm = _redeclare_units(BATES_STEP, tmp_path / "cm.stp", "#136", "#138")
        assert step_length_unit(cm) == ("cm", pytest.approx(1e-2))

    def test_file_with_no_length_unit_returns_none(self, tmp_path):
        bare = tmp_path / "bare.stp"
        bare.write_text(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\nEND-ISO-10303-21;\n"
        )
        assert step_length_unit(bare) is None

    def test_survives_entities_wrapped_across_lines(self):
        # Real exporters break a single entity mid-token at ~72 columns. A
        # line-oriented scan would miss the unit entirely; this fixture has
        # wrapped entities in it already, so passing means the split on ';'
        # is doing its job.
        assert step_length_unit(BATES_STEP) is not None


class TestStepImportScale:
    """The converter's metres-out convention, which we rely on.

    These are the tests that fail if ``cascadio`` ever changes its output
    convention -- the point being that they fail here, loudly, rather than in a
    motor simulation that merely reports the wrong answer.
    """

    def test_inch_declared_file_imports_at_true_physical_size(self):
        mesh = load_mesh(BATES_STEP)
        # 1 in x 2 in, in metres. The tolerance is tessellation slack on the
        # curved wall, not unit slack.
        assert max(mesh.extents) == pytest.approx(BATES_STEP_LENGTH_M, rel=1e-3)
        across = sorted(mesh.extents)[:2]
        for extent in across:
            assert extent == pytest.approx(BATES_STEP_DIAMETER_M, rel=1e-2)

    def test_same_geometry_declared_mm_imports_25_4x_smaller(self, tmp_path):
        # The strongest available check: identical coordinates, two different
        # declarations. If the declaration were being ignored the ratio would
        # be 1.0 rather than the inch/mm ratio.
        inch = load_mesh(BATES_STEP)
        mm = load_mesh(_redeclare_units(BATES_STEP, tmp_path / "mm.stp", "#136", "#137"))
        assert max(inch.extents) / max(mm.extents) == pytest.approx(25.4, rel=1e-6)

    def test_declared_unit_is_reported_to_the_caller(self):
        stats = mesh_stats(load_mesh(BATES_STEP))
        assert stats["source_units"] == "inch"
        assert stats["units_origin"] == "declared"

    def test_undeclared_step_is_refused_rather_than_guessed(self, tmp_path):
        bare = tmp_path / "bare.stp"
        bare.write_text(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\nEND-ISO-10303-21;\n"
        )
        with pytest.raises(MeshImportError, match="declares no length unit"):
            load_mesh(bare)

    def test_step_import_is_watertight_and_has_a_volume(self):
        # glTF (the converter's output format) stores vertices per-face, so a
        # STEP grain arrives with coincident-but-separate vertices and reads as
        # open unless they are merged. That took volume, mass and port fraction
        # off the measurements panel for every CAD import.
        mesh = load_mesh(BATES_STEP)
        assert mesh.is_watertight
        assert mesh.volume > 0.0

        # 1 in outer diameter with a 0.2 in bore, 2 in long -- the part as
        # drawn. Checked in metres, which is what the import must produce.
        bore, outer, length = 0.1 * 0.0254, 0.5 * 0.0254, BATES_STEP_LENGTH_M
        expected = np.pi * (outer**2 - bore**2) * length
        assert mesh.volume == pytest.approx(expected, rel=2e-2)

    def test_assume_units_refused_for_cad(self):
        # The file already declares a unit; layering an assumption on top would
        # double-convert. Better to refuse than to quietly pick one.
        with pytest.raises(MeshImportError, match="declares its own length unit"):
            load_mesh(BATES_STEP, assume_units="mm")


class TestMeshFormatUnits:
    """STL and friends declare nothing, so the caller may state a unit."""

    @staticmethod
    def _write_mm_grain(tmp_path) -> Path:
        """A grain modelled in millimetres: 50 mm across, 120 mm long."""
        grain = trimesh.creation.annulus(r_min=10.0, r_max=25.0, height=120.0)
        path = tmp_path / "grain_mm.stl"
        grain.export(path)
        return path

    def test_taken_as_is_when_nothing_is_stated(self, tmp_path):
        mesh = load_mesh(self._write_mm_grain(tmp_path))
        assert max(mesh.extents) == pytest.approx(120.0, rel=1e-3)
        assert mesh_stats(mesh)["units_origin"] == "unknown"

    def test_assume_units_scales_to_metres(self, tmp_path):
        mesh = load_mesh(self._write_mm_grain(tmp_path), assume_units="mm")
        assert max(mesh.extents) == pytest.approx(0.120, rel=1e-3)
        assert mesh.extents.max() < 1.0

    def test_assumed_unit_is_reported_and_flagged_as_assumed(self, tmp_path):
        stats = mesh_stats(load_mesh(self._write_mm_grain(tmp_path), assume_units="mm"))
        # "assumed" rather than "declared" is the distinction the UI shows: the
        # number is only as good as the user's word.
        assert (stats["source_units"], stats["units_origin"]) == ("mm", "assumed")

    def test_unknown_assumed_unit_is_rejected(self, tmp_path):
        with pytest.raises(UnitError):
            load_mesh(self._write_mm_grain(tmp_path), assume_units="cubits")

    def test_volume_scales_with_the_cube_of_the_length_factor(self, tmp_path):
        path = self._write_mm_grain(tmp_path)
        raw = load_mesh(path).volume
        scaled = load_mesh(path, assume_units="mm").volume
        # This is the specific arithmetic behind a 10^9 volume error: getting
        # length wrong by 1000 gets volume wrong by a billion.
        assert raw / scaled == pytest.approx(1e9, rel=1e-6)

    def test_canonical_unit_recorded_on_the_mesh(self, tmp_path):
        mesh = load_mesh(self._write_mm_grain(tmp_path), assume_units="mm")
        assert mesh.metadata["units"] == CANONICAL_LENGTH_UNIT == "m"
