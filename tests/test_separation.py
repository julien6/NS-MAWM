from pathlib import Path


def test_ns_mawm_is_environment_and_algorithm_agnostic():
    root = Path(__file__).resolve().parents[1]
    ns_files = list((root / "ns_mawm").glob("*.py"))
    forbidden = ("gridcraft", "overcooked", "smac", "predator", "mappo", "qmix", "sac", "rssm", "transformer")
    for path in ns_files:
        text = path.read_text().lower()
        for token in forbidden:
            assert token not in text, f"{path} contains environment/algorithm token {token}"


def test_install_scripts_left_at_root():
    root = Path(__file__).resolve().parents[1]
    assert (root / "install.sh").exists()
    assert (root / "install_smac_pettingzoo.sh").exists()
