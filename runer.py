import sys
from omegaconf import OmegaConf

from exp_saliency_general import ShapeXPipline
from txai.synth_data.synth_data_base import SynthTrainDataset


SHAPEX_BETA_ROOT = "/home/gbsguest/Research/boson/TS/XTS/ShapeX-beta"


def build_config(dataset_name: str, num_classes: int = 4, seq_len: int = 500, proto_len: int = 30):
    """
    Construct a config object compatible with exp_saliency_general.get_args.
    The dataset-specific defaults can be overridden by CLI flags that
    exp_saliency_general.get_args will parse from sys.argv.
    """
    # Each dataset must be an attribute on the config with (num_classes, seq_len, proto_len)
    cfg = {
        "base": {"root_dir": SHAPEX_BETA_ROOT},
        "dataset": {"name": dataset_name, "meta_dataset": "default"},
        dataset_name: {
            "num_classes": num_classes,
            "seq_len": seq_len,
            "proto_len": proto_len,
        },
    }
    return OmegaConf.create(cfg)


def main():
    # Load run params from YAML (only three keys)
    run_cfg = OmegaConf.load(f"{SHAPEX_BETA_ROOT}/configs/run_configs.yaml").defaults
    datasets = str(run_cfg.datasets)
    do_train = bool(run_cfg.train)
    do_test = bool(run_cfg.test)

    dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]

    for ds in dataset_list:
        print(f"===== Running dataset: {ds} =====")
        config = build_config(ds)
        # prevent downstream arg parse from reading CLI
        saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        pipeline = ShapeXPipline(config)
        sys.argv = saved_argv

        # Train
        if do_train:
            # get_args() inside pipeline will read is_training from CLI; ensure it is set
            # If not provided by user, training flag here implies is_training=1
            # Users can still override via CLI (e.g., --is_training 0)
            pipeline.args.is_training = 1
            pipeline.train_shapex()

        # Evaluate
        if do_test:
            pipeline.args.saliency = True
            pipeline.eval_shapex()


if __name__ == "__main__":
    main()


