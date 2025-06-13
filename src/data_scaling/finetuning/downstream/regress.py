import hydra
from omegaconf import DictConfig
from datasets import load_from_disk
from data_scaling.evaluation.linear_probe import run_loocv_linear_probe

@hydra.main(config_path='../../../configs', config_name='base.yaml')
def main(cfg: DictConfig):
    # Load dataset (assume HF dataset path in config)
    dataset_path = getattr(cfg.data, 'hf_dataset_path', None)
    if dataset_path is None:
        from data_scaling.paths import PROJECT_DIR
        dataset_path = PROJECT_DIR / 'hf'
    dataset = load_from_disk(str(dataset_path))

    # Get test samples for LOOCV from config
    test_samples = cfg.data.multimodal.test
    # Filter dataset to only test samples
    dataset = [row for row in dataset if row['name'] in test_samples]

    # Path to fusion model checkpoint (should be in config)
    model_ckpt_path = cfg.models.checkpoint_path

    # Run LOOCV linear probe regression
    metrics_list, agg_metrics = run_loocv_linear_probe(
        cfg, dataset, model_ckpt_path,
        task_type='regression',
        target_key='cell_type_ratio'
    )
    print('Aggregated LOOCV regression results:', agg_metrics)

if __name__ == "__main__":
    main()
