from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class BasicDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        **kwargs
    ):
        """ Basic dataset
            Need to set 
                __init__()
                __len__() : 
                __getitem__() : output must be  {'inputs':inputs, 'targets':targets} dictionary for Basic model
        """
        super().__init__()
        self.data_dir = data_dir

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int):
        pass
        # return {'inputs':inputs, 'targets':targets}

class Datamodule(pl.LightningDataModule):
    def __init__(self, 
                data_path: str,
                batch_size: int,
                num_workers: int,
                shuffle = True,
                **kwargs
                ):
        """ Basic datamodule
            Need to set 
                __init__()
                prepare_data() : load data to data_path to prepare data
                setup() : set dataset to train, validation, test dataset
        """

        super().__init__()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage='fit'):
        # -- dataset building
        if stage == "fit":
            self.train_dataset = BasicDataset(
                self.data_path
            )
            self.valid_dataset = BasicDataset(
                self.data_path
            )

        if stage == "test":
            self.test_dataset = BasicDataset(
                self.data_path
            )
        
    def train_dataloader(self):
        return DataLoader(
                    self.train_dataset, 
                    batch_size = self.batch_size, 
                    num_workers = self.num_workers,
                    shuffle=self.shuffle
                )

    def val_dataloader(self):
        return DataLoader(
                    self.valid_dataset, 
                    batch_size = self.batch_size, 
                    num_workers = self.num_workers,
                    shuffle=False
                )


    def test_dataloader(self):
        return DataLoader(
                    self.test_dataset, 
                    batch_size = self.batch_size, 
                    num_workers = self.num_workers,
                    shuffle=False
                )
