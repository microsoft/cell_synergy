import torch
from torch.utils.data import DataLoader, Dataset


class ExpressionDataset(Dataset):
    def __init__(self, dataset):
        """
        Dataset for gene expression data.

        Args:
            dataset: Hugging Face Dataset object.
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        gexp = torch.as_tensor(row["gexp"], dtype=torch.float32)

        if "mask" in row and row["mask"] is not None:
            mask = torch.as_tensor(row["mask"], dtype=torch.bool)
        else:
            # Create a default all-True mask if not provided
            mask = torch.ones(gexp.shape[:-1], dtype=torch.bool)

        return {
            "index": idx,
            "gexp": gexp,
            "mask": mask,
        }


class ExpressionDataLoader:
    def __init__(self, dataset):
        """
        Initialize the ExpressionDataLoader with a specific dataset.

        Args:
            dataset: Hugging Face Dataset object.
        """
        self.train_dataset = ExpressionDataset(dataset)

    def get_loader(self, batch_size, num_workers) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )
