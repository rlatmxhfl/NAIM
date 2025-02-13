import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import logging
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
from .routines import initialize_experiment
from CMC_utils import metrics
from CMC_utils import cross_validation as cv
from CMC_utils.datasets import SupervisedTabularDatasetTorch
from CMC_utils.models import test_torch_model

log = logging.getLogger(__name__)

__all__ = ["supervised_learning_main"]

def create_tsne_dataset(dataset: SupervisedTabularDatasetTorch, samples_per_class: int = 20) -> SupervisedTabularDatasetTorch:
    """
    Creates a balanced subset of the dataset for t-SNE visualization.

    Args:
        dataset: The original SupervisedTabularDatasetTorch.
        samples_per_class: Number of samples to select per class.

    Returns:
        A new SupervisedTabularDatasetTorch containing a balanced subset.
    """
    class_labels = np.argmax(dataset.labels, axis=1) # Assuming one-hot encoded labels
    unique_classes = np.unique(class_labels)
    balanced_indices = []

    for cls in unique_classes:
        class_indices = np.where(class_labels == cls)[0]
        sampled_indices = np.random.choice(class_indices, size=min(samples_per_class, len(class_indices)), replace=False)
        balanced_indices.extend(sampled_indices)

    balanced_data = dataset.data[balanced_indices]
    balanced_labels = dataset.labels[balanced_indices]
    balanced_ids = [dataset.ID[i] for i in balanced_indices]
    balanced_df = pd.DataFrame(balanced_data, columns=dataset.columns) #Reconstruct DataFrame to keep columns
    balanced_df.index = balanced_ids
    balanced_labels_df = pd.DataFrame(balanced_labels, index=balanced_ids)


    tsne_dataset = SupervisedTabularDatasetTorch(
        data=balanced_df,
        labels=balanced_labels_df,
        set_name="tsne",
        preprocessing_params=dataset.preprocessing_params,
        preprocessing_paths=dataset.preprocessing_paths,
        test_fold=0,
        val_fold=0
    )
    return tsne_dataset

def supervised_learning_main(cfg: DictConfig) -> None:
    log.info(f"Supervised main started")

    initialize_experiment(cfg)

    dataset = instantiate(cfg.db, model_label_types=cfg.model.label_types, model_framework=cfg.model.framework, preprocessing_params=cfg.preprocessing, _recursive_=False)

    cv.set_cross_validation(dataset.info_for_cv, cfg.paths.cv, test_params=cfg.test_cv, val_params=cfg.val_cv)

    for test_fold, val_fold, train, val, test, last_val_fold in cv.get_cross_validation(cfg.paths.cv, "train", "val", "test"):
        train_data, train_labels, val_data, val_labels = cv.get_sets_with_idx(dataset.data, train, val, labels=dataset.labels_for_model)

        train_set = instantiate(cfg.db.dataset_class, train_data, train_labels, "train", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold, augmentation=cfg.model.name == "naim")
        val_set = instantiate(cfg.db.dataset_class, val_data, val_labels, "val", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold)

        model_params = call(cfg.model.set_params_function, OmegaConf.to_object(cfg.model), preprocessing_params=dataset.preprocessing_params, train_set=train_set, val_set=val_set)
        model = instantiate(model_params["init_params"], _recursive_=False)

        train_params = OmegaConf.to_object(cfg.train)
        train_params["set_metrics"] = metrics.set_metrics_params(train_params.get("set_metrics", {}), preprocessing_params=dataset.preprocessing_params)
        call(model_params["train_function"], model, train_set, model_params, cfg.paths.model, val_set=val_set, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

        call(model_params["test_function"], train_set, val_set, model_params=model_params, model_path=cfg.paths.model, prediction_path=cfg.paths.predictions, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

        test_data, test_labels = cv.get_sets_with_idx(dataset.data, test, labels=dataset.labels_for_model)
        test_set = instantiate(cfg.db.dataset_class, test_data, test_labels, "test", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold)

        call(model_params["test_function"], test_set, model_params=model_params, model_path=cfg.paths.model, prediction_path=cfg.paths.predictions, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

        # Create balanced dataset for t-SNE
        tsne_set = create_tsne_dataset(val_set, samples_per_class=20) # Use val_set for visualization, can change to train_set or test_set
        # tsne_dataloader = DataLoader(tsne_set, batch_size=train_params["dl_params"]["batch_size"], shuffle=False)

        # Extract features for t-SNE
        features, labels_tsne = test_torch_model(tsne_set, model_params=model_params, model_path=cfg.paths.model, prediction_path=cfg.paths.predictions, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, extractor=True)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        tsne_embeddings = tsne.fit_transform(features)

        # Plot t-SNE
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(np.argmax(labels_tsne, axis=1)) # Assuming one-hot encoded labels
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            indices = np.where(np.argmax(labels_tsne, axis=1) == label)
            plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], color=color, label=f'Class {label}') # Or use dataset.classes[label] if class names are available

        plt.title('t-SNE visualization of NAIM embeddings')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.legend()
        plt.savefig(os.path.join(cfg.paths.results, f'tsne_fold_{test_fold}_{val_fold}.png')) # Save t-SNE plot
        plt.close()

        del train_data, train_labels, val_data, val_labels, test_data, test_labels, train_set, val_set, test_set, model_params, train_params

    performance_metrics = metrics.set_metrics_params(cfg.performance_metrics, preprocessing_params=dataset.preprocessing_params)
    metrics.compute_performance(dataset.classes, cfg.paths.predictions, cfg.paths.results, performance_metrics)

    log.info(f"Job finished")


if __name__ == "__main__":
    pass
