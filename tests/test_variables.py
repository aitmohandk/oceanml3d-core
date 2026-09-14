import pytest

from oceanml3d.variables import Role, VariableSet, VariableSpec


def test_roles_and_indices(variables):
    assert [s.name for s in variables.inputs] == ["ssh", "ugos", "vgos", "u10", "v10", "lat", "bathy"]
    assert [s.name for s in variables.targets] == ["u_drifter", "v_drifter"]
    assert variables.target_indices == [7, 8]
    assert variables.n_input_channels(11) == 77
    assert variables["lat"].is_static and variables["lat"].role is Role.STATIC


def test_needs_target():
    with pytest.raises(ValueError):
        VariableSet([VariableSpec("a", "x", role="input")])


def test_duplicate_names():
    with pytest.raises(ValueError):
        VariableSet([VariableSpec("a", "x", role="target"), VariableSpec("a", "y")])
