from .base import PLDataModuleWrapper

import schnetpack.transform as tform
from schnetpack.data import AtomsLoader, AtomsDataFormat
from schnetpack.data.datamodule import AtomsDataModule


class SchNetDataModule(PLDataModuleWrapper):
    def __init__(self, stage, batch_size, collate_fn=None):
        """
        Arguments:

            stage (str):
                Path to a directory containing the following files:

                    * `full.db`: the formatted schnetpack.data.ASEAtomsData database
                    * `split.npz`: the file specifying the train/val/test split indices

            batch_size (int):
                The batch size, to be passed directly to the data loaders

            collate_fn (callable):
                A function for collating batch samples. See PyTorch
                documentation regarding `collate_fn` for more details.
        """
        super().__init__(stage=stage, batch_size=batch_size, collate_fn=None)


    def setup(self, stage, **kwargs):
        """
        Populates the `self.train_dataset`, `self.test_dataset`, and
        `self.val_dataset` class attributes. Will be called automatically in
        __init__()
        """

        datamodule = AtomsDataModule(
            datapath=os.path.join(data_dir, 'full.db'),
            split_file=os.path.join(data_dir, 'split.npz'),
            format=AtomsDataFormat.ASE,
            load_properties=['energy', 'forces'],
            batch_size=batch_size,
            num_workers=int(np.floor(int(os.environ['LSB_MAX_NUM_PROCESSORS'])/int(os.environ['GPUS_PER_NODE']))),
            shuffle_train=False,
            transforms=[
                tform.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False),
                tform.MatScipyNeighborList(cutoff=cutoff),
            ],
        )

        datamodule.setup()

        self.train_dataset  = datamodule.train_dataset
        self.test_dataset   = datamodule.test_dataset
        self.val_dataset    = datamodule.val_dataset


    def get_dataloader(self, dataset):
        return AtomsLoader(
                dataaset,
                batch_size=self.batch_size,
                # num_workers=self.num_workers,
                # shuffle=True,
                # pin_memory=self._pin_memory,
            )





