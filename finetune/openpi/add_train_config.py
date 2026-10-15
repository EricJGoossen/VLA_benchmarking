"""Add the pi05_droid_lora_finetune TrainConfig to an openpi checkout.

Inserts it right after openpi's own pi05_droid_finetune config, which it mirrors except:
  - LoRA (gemma_2b_lora / gemma_300m_lora + matching freeze filter, no EMA), so it fits on a
    single 32 GB GPU. Full fine-tuning needs well over 70 GB.
  - 5k steps with a checkpoint kept every 1k (a 35-demo dataset is ~15k frames, so openpi's
    default 20k steps at batch 32 would be ~40 epochs).
  - Reads the local LeRobot dataset written by convert_teleop_to_lerobot.py.
  - wandb off.

Usage:  python add_train_config.py <openpi_dir> [--repo-id eric/teleop_droid]
"""
import argparse
import pathlib

CONFIG = '''    TrainConfig(
        # Added by VLA_benchmarking finetune/openpi/add_train_config.py
        name="pi05_droid_lora_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotDROIDDataConfig(
            repo_id="{repo_id}",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Reuse the original DROID norm stats, as openpi's pi05_droid_finetune does
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=5_000,
        batch_size=32,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
    ),
'''

ANCHOR = '        name="pi05_droid_finetune",'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("openpi_dir")
    parser.add_argument("--repo-id", default="eric/teleop_droid")
    args = parser.parse_args()

    path = pathlib.Path(args.openpi_dir) / "src" / "openpi" / "training" / "config.py"
    text = path.read_text()
    if 'name="pi05_droid_lora_finetune"' in text:
        print(f"pi05_droid_lora_finetune already present in {path}")
        return

    # Insert after the TrainConfig(...) block that contains the anchor
    start = text.index(ANCHOR)
    end = text.index("\n    ),\n", start) + len("\n    ),\n")
    path.write_text(text[:end] + CONFIG.format(repo_id=args.repo_id) + text[end:])
    print(f"Added pi05_droid_lora_finetune (repo_id={args.repo_id}) to {path}")


if __name__ == "__main__":
    main()
