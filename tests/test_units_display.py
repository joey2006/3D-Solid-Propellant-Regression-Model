"""The display-edge unit system (#154).

The engine computes in SI and converts only where a number reaches a person.
That is what makes a unit mistake impossible to turn into a physics mistake --
but only if the conversions themselves are right, and a wrong factor produces a
plausible-looking number rather than an error. So every factor is checked
against an independently known value rather than against itself.
"""

from __future__ import annotations

import pytest

from srm_burnback.units import (
    SYSTEMS,
    UnitError,
    convert,
    format_value,
    from_si,
    to_si,
    units_for,
)


class TestKnownConversions:
    """Each expectation is a value you can look up, not one this code produced."""

    @pytest.mark.parametrize(
        "quantity, value, source, target, expected",
        [
            ("length", 1.0, "in", "mm", 25.4),
            ("length", 1.0, "ft", "in", 12.0),
            ("pressure", 1.0, "MPa", "psi", 145.0377),
            ("pressure", 1.0, "atm", "kPa", 101.325),
            ("pressure", 1.0, "bar", "psi", 14.503774),
            ("force", 1.0, "lbf", "N", 4.4482216),
            ("mass", 1.0, "kg", "lb", 2.2046226),
            ("mass", 1.0, "lb", "oz", 16.0),
            ("density", 1750.0, "kg/m3", "lb/in3", 0.06322276),
            ("velocity", 1.0, "mph", "ft/s", 1.4666667),
            ("burn_rate", 5.0, "mm/s", "in/s", 0.19685039),
            ("volume", 1.0, "in3", "cm3", 16.387064),
            ("area", 1.0, "in2", "mm2", 645.16),
            ("impulse", 1.0, "lbf*s", "N*s", 4.4482216),
        ],
    )
    def test_factor(self, quantity, value, source, target, expected):
        assert convert(value, quantity, source, target) == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.parametrize(
        "value, source, target, expected",
        [
            (0.0, "C", "F", 32.0),
            (100.0, "C", "F", 212.0),
            (-40.0, "C", "F", -40.0),      # the one place the scales cross
            (0.0, "C", "K", 273.15),
            (491.67, "R", "F", 32.0),
            (0.0, "K", "R", 0.0),
        ],
    )
    def test_temperature_has_an_offset_not_just_a_scale(
        self, value, source, target, expected
    ):
        assert convert(value, "temperature", source, target) == pytest.approx(
            expected, abs=1e-6
        )

    def test_a_factor_is_refused_for_temperature(self):
        from srm_burnback.units import factor

        # Asking for a factor would silently drop the offset, so it raises.
        with pytest.raises(UnitError, match="offset"):
            factor("temperature", "C")


class TestRoundTrip:
    """Switching systems must never change the stored value."""

    @pytest.mark.parametrize("quantity", sorted(SYSTEMS["metric"]))
    def test_si_survives_a_round_trip_through_both_systems(self, quantity):
        original = 123.456
        for system in ("metric", "imperial"):
            unit = units_for(quantity, system)
            assert to_si(from_si(original, quantity, unit), quantity, unit) == (
                pytest.approx(original, rel=1e-12)
            )

    def test_every_quantity_is_defined_in_both_systems(self):
        # A quantity present in one system and missing from the other would
        # raise only when a user happened to switch, which is the worst time.
        assert set(SYSTEMS["metric"]) == set(SYSTEMS["imperial"])


class TestFormatting:
    def test_missing_values_render_as_a_dash(self):
        # An unclosed mesh has no volume; the display says so rather than
        # inventing a zero.
        assert format_value(None, "volume", "metric") == "--"

    def test_lengths_carry_their_unit(self):
        assert format_value(0.0254, "length", "imperial") == "1.000 in"
        assert format_value(0.0254, "length", "metric") == "25.40 mm"

    def test_a_real_grain_reads_sensibly_in_both_systems(self):
        # 43 g of propellant: two decimals of a kilogram would round it to 0.04.
        assert format_value(0.0432, "mass", "metric") == "0.043 kg"
        assert format_value(0.0432, "mass", "imperial") == "0.095 lb"

    def test_pressure_uses_the_convention_the_burn_model_expects(self):
        # Vieille's a ~ 0.005 is the MPa convention, so metric shows MPa.
        assert units_for("pressure", "metric") == "MPa"
        assert format_value(5e6, "pressure", "metric") == "5.000 MPa"

    def test_unknown_units_and_systems_are_rejected(self):
        with pytest.raises(UnitError):
            units_for("length", "furlongs")
        with pytest.raises(UnitError):
            to_si(1.0, "length", "cubits")
        with pytest.raises(UnitError):
            to_si(1.0, "sparkliness", "m")
