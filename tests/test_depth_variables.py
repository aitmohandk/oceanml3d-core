from oceanml3d.variables import VariableSet


def test_depth_expansion_and_groups():
    vs = VariableSet.from_config({
        "ssh_obs": {"source": "p", "role": "input"},
        "zos": {"source": "s", "role": "target", "group": "ssh"},
        "thetao": {"source": "m", "role": "target", "depth_indices": [0, 2, 25], "group": "temperature", "depth_values": {0: 0.49}},
        "argo": {"source": "a", "var_name": "thetao", "role": "input", "depth_indices": [0, 2]},
    })
    assert [s.name for s in vs.targets] == ["zos", "thetao_d00", "thetao_d02", "thetao_d25"]
    assert vs["thetao_d25"].depth_index == 25 and vs["thetao_d25"].var_name == "thetao"
    assert vs["thetao_d00"].depth_m == 0.49 and vs["thetao_d02"].base_name == "thetao"
    assert vs.target_groups == {"ssh": ["zos"], "temperature": ["thetao_d00", "thetao_d02", "thetao_d25"]}
    assert vs["argo_d02"].group == "argo" and vs["argo_d02"].is_input
