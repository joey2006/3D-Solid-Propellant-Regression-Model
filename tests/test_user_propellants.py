"""The user's own saved propellants (#152).

These cover the store itself rather than the panel, so they need no Qt widgets
-- only ``QStandardPaths``, which is redirected to a temporary directory so a
test run can never touch the real one.
"""

from __future__ import annotations

import json

import pytest

from srm_burnback.propellants import LIBRARY, Propellant


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The propellant store, pointed at a throwaway directory."""
    from desktop import user_propellants

    monkeypatch.setattr(
        user_propellants, "store_path", lambda: tmp_path / "propellants.json"
    )
    return user_propellants


def a_propellant(name="My KNSB batch 3", a=0.0061, n=0.24, density=1820.0):
    return Propellant(name=name, a=a, n=n, density=density, source="static test")


class TestRoundTrip:
    def test_an_empty_store_reads_as_no_propellants(self, store):
        assert store.load() == []

    def test_a_saved_propellant_comes_back_intact(self, store):
        store.save(a_propellant())
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].name == "My KNSB batch 3"
        assert loaded[0].a == pytest.approx(0.0061)
        assert loaded[0].n == pytest.approx(0.24)
        assert loaded[0].density == pytest.approx(1820.0)

    def test_several_propellants_keep_their_order(self, store):
        for i in range(3):
            store.save(a_propellant(name=f"batch {i}"))
        assert [p.name for p in store.load()] == ["batch 0", "batch 1", "batch 2"]

    def test_the_file_is_readable_json(self, store):
        store.save(a_propellant())
        payload = json.loads(store.store_path().read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert payload[0]["name"] == "My KNSB batch 3"

    def test_the_directory_is_created_if_absent(self, store, tmp_path):
        nested = tmp_path / "deep" / "deeper" / "propellants.json"
        store.store_path = lambda: nested
        store.save_all([a_propellant()])
        assert nested.exists()


class TestUpdatingAndDeleting:
    def test_saving_an_existing_name_replaces_it(self, store):
        store.save(a_propellant(a=0.0061))
        store.save(a_propellant(a=0.0075))
        loaded = store.load()
        assert len(loaded) == 1, "re-saving must update, not duplicate"
        assert loaded[0].a == pytest.approx(0.0075)

    def test_deleting_removes_only_the_named_one(self, store):
        store.save(a_propellant(name="keep"))
        store.save(a_propellant(name="drop"))
        store.delete("drop")
        assert [p.name for p in store.load()] == ["keep"]

    def test_deleting_something_absent_is_harmless(self, store):
        store.save(a_propellant(name="keep"))
        store.delete("never existed")
        assert [p.name for p in store.load()] == ["keep"]


class TestBuiltinsAreProtected:
    @pytest.mark.parametrize("name", [p.name for p in LIBRARY])
    def test_every_published_name_is_recognised(self, store, name):
        assert store.is_builtin(name)

    def test_a_users_own_name_is_not_builtin(self, store):
        assert not store.is_builtin("My KNSB batch 3")


class TestDamagedFiles:
    """Losing saved propellants is bad; refusing to start the app is worse."""

    def test_unparseable_json_reads_as_empty(self, store):
        store.store_path().parent.mkdir(parents=True, exist_ok=True)
        store.store_path().write_text("{not json at all", encoding="utf-8")
        assert store.load() == []

    def test_a_json_object_instead_of_a_list_reads_as_empty(self, store):
        store.store_path().parent.mkdir(parents=True, exist_ok=True)
        store.store_path().write_text('{"name": "x"}', encoding="utf-8")
        assert store.load() == []

    def test_one_broken_entry_does_not_discard_the_others(self, store):
        store.store_path().parent.mkdir(parents=True, exist_ok=True)
        store.store_path().write_text(
            json.dumps(
                [
                    {"name": "good", "a": 0.005, "n": 0.35, "density": 1750.0},
                    "not an object",
                    {"missing": "everything"},
                ]
            ),
            encoding="utf-8",
        )
        loaded = store.load()
        assert [p.name for p in loaded] == ["good"]
