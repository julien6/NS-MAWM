from run_benchmarl_mappo_gridcraft import route_benchmarl_metrics


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
