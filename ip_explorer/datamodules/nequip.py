from .base import PLDataModuleWrapper

import os

from nequip.train import Trainer
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.data.dataloader import DataLoader


class NequIPDataModule(PLDataModuleWrapper):

    def __init__(self, stage, **kwargs):
        """
        Arguments:

            stage (str):
                The working directory used for training a NequIP model. Should
                contain the following files:

                    * `config.yaml`: the full configuration file, specifying data 
                        splits. Note that the configuration file should have the
                        'dataset', 'test_dataset', and 'validation_dataset'
                        keys.
        """
        super().__init__(stage=stage, **kwargs)


    def setup(self, stage, **kwargs):
        """
        Arguments:

            stage (str):
                The working directory used for training a NequIP model. Should
                contain the following files:

                    * `config.yaml`: the full configuration file, specifying data 
                        splits. Note that the configuration file should have the
                        'dataset', 'test_dataset', and 'validation_dataset'
                        keys.
        """
        _, model_config = Trainer.load_model_from_training_session(traindir=stage)

        dataset_config = Config.from_file(
            os.path.join(stage, 'config.yaml'),
            defaults={"r_max": model_config["r_max"]}
        )


        self.train_dataset  = dataset_from_config(dataset_config, prefix="dataset")
        self.test_dataset   = dataset_from_config(dataset_config, prefix="test_dataset")
        self.val_dataset    = dataset_from_config(dataset_config, prefix="validation_dataset")

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
