ENVIRONMENT_DYNAMICS_VERSION = "gridcraft_dynamics_v2_armed_combat"
REWARD_SCHEMA_VERSION = "gridcraft_reward_v2_team_milestones"


def version_provenance() -> dict[str, str]:
    return {
        "environment_dynamics_version": ENVIRONMENT_DYNAMICS_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
    }


def validate_version_provenance(provenance: dict) -> tuple[bool, str]:
    actual_dynamics = provenance.get("environment_dynamics_version")
    if actual_dynamics != ENVIRONMENT_DYNAMICS_VERSION:
        return (
            False,
            f"environment dynamics version {actual_dynamics!r} does not match "
            f"{ENVIRONMENT_DYNAMICS_VERSION!r}",
        )
    actual_reward = provenance.get("reward_schema_version")
    if actual_reward != REWARD_SCHEMA_VERSION:
        return (
            False,
            f"reward schema version {actual_reward!r} does not match "
            f"{REWARD_SCHEMA_VERSION!r}",
        )
    return True, "valid"
