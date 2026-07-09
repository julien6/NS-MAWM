from run_benchmarl_marl_gridcraft import route_benchmarl_metrics


def test_mambpo_imagination_metrics_route_to_marl_training():
    routed = route_benchmarl_metrics(
        {
            "mambpo/world_model_loss": 1.0,
            "mambpo/training_imagined_reward": 2.0,
            "mambpo/imagined_ratio": 0.9,
            "train/loss_qvalue": 3.0,
            "eval/reward": 4.0,
        }
    )
    assert routed["MARL Training/imagination_world_model_loss"] == 1.0
    assert routed["MARL Training/training_imagined_reward"] == 2.0
    assert routed["MARL Training/imagined_ratio"] == 0.9
    assert routed["MARL Training/train_loss_qvalue"] == 3.0
    assert routed["MARL Evaluation/eval_reward"] == 4.0


def test_hierarchy_metrics_get_dedicated_wandb_section():
    routed = route_benchmarl_metrics(
        {
            "hierarchy/training_event_count_mob_kill": 3.0,
            "hierarchy/evaluation_task_level_max": 8.0,
        }
    )
    assert routed["Reward hierarchy diagnosis/training_event_count_mob_kill"] == 3.0
    assert routed["Reward hierarchy diagnosis/evaluation_task_level_max"] == 8.0
