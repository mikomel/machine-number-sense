import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.functional import nll_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from mns.dataset import MNSDataset
from mns.model import ConvMLP, ConvLSTM, ConvNALU
from mns.model.scl import SCL


class MNSModule(pl.LightningModule):
    def __init__(self, hparams):
        super(MNSModule, self).__init__()
        self.hparams = hparams
        self.model = self.build_model()

    def build_model(self):
        if self.hparams.model == 'mlp':
            return ConvMLP(image_size=self.hparams.image_size)
        elif self.hparams.model == 'lstm':
            return ConvLSTM(image_size=self.hparams.image_size)
        elif self.hparams.model == 'nalu':
            return ConvNALU(image_size=self.hparams.image_size)
        elif self.hparams.model == 'scl':
            return SCL(image_size=self.hparams.image_size)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.reduce_lr_on_plateau_factor,
            patience=self.hparams.reduce_lr_on_plateau_patience,
            verbose=True
        )
        return [optimizer], [scheduler]

    def step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y_hat = logits.log_softmax(dim=-1)
        loss = nll_loss(y_hat, y)
        acc = y_hat.argmax(dim=1).eq(y).sum() / float(len(y))
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        return {
            'loss': loss,
            'progress_bar': {'acc': acc},
            'log': {'train_loss': loss, 'train_acc': acc}
        }

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        return {
            'val_loss': loss,
            'val_acc': acc
        }

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        acc = torch.stack([o['val_acc'] for o in outputs]).mean()
        return {
            'progress_bar': {'val_loss': loss, 'val_acc': acc},
            'log': {'val_loss': loss, 'val_acc': acc, 'step': self.current_epoch}
        }

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        return {
            'test_loss': loss,
            'test_acc': acc
        }

    def test_epoch_end(self, outputs):
        loss = torch.stack([o['test_loss'] for o in outputs]).mean()
        acc = torch.stack([o['test_acc'] for o in outputs]).mean()
        return {
            'progress_bar': {'test_loss': loss, 'test_acc': acc},
            'log': {'test_loss': loss, 'test_acc': acc, 'step': self.current_epoch}
        }

    def train_dataloader(self):
        data_dir = os.path.join(self.hparams.dataset_dir, 'train_set')
        dataset = MNSDataset(data_dir, self.hparams.image_size)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        data_dir = os.path.join(self.hparams.dataset_dir, 'val_set')
        dataset = MNSDataset(data_dir, self.hparams.image_size)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        data_dir = os.path.join(self.hparams.dataset_dir, 'test_set')
        dataset = MNSDataset(data_dir, self.hparams.image_size)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    @staticmethod
    def build_parser():
        parser = ArgumentParser()
        parser.add_argument('--batch-size', default=128, type=int)
        parser.add_argument('--dataset-dir', default='/app/datasets', type=str)
        parser.add_argument('--disable-early-stopping', dest='disable_early_stopping', action='store_true')
        parser.add_argument('--disable-model-checkpoint', dest='disable_model_checkpoint', action='store_true')
        parser.add_argument('--early-stopping-patience', default=10, type=int)
        parser.add_argument('--gpu', default='0', type=str)
        parser.add_argument('--half-precision', dest='half_precision', action='store_true')
        parser.add_argument('--image-size', default=160, type=int)
        parser.add_argument('--learning-rate', default=3e-4, type=float)
        parser.add_argument('--log-dir', default='/app/logs', type=str)
        parser.add_argument('--model', default='mlp', type=str)
        parser.add_argument('--num-epochs', default=50, type=int)
        parser.add_argument('--num-workers', default=16, type=int)
        parser.add_argument('--reduce-lr-on-plateau-factor', default=0.1, type=float)
        parser.add_argument('--reduce-lr-on-plateau-patience', default=4, type=int)
        parser.add_argument('--resume-from-checkpoint', default=None, type=str)
        parser.add_argument('--seed', default=-1, type=int)
        return parser

    def build_trainer(self, **kwargs):
        logger = TensorBoardLogger(self.hparams.log_dir, name=self.hparams.model)
        distributed_backend = 'ddp' if len(self.hparams.gpu.split(',')) > 1 else None
        if self.hparams.seed != -1:
            print(f"Using fixed seed: {self.hparams.seed}")
            pl.seed_everything(self.hparams.seed)
        if distributed_backend == 'ddp' and self.hparams.seed == -1:
            print('Warning: Using ddp distributed backend but seed was not provided.')
            print('Warning: Setting fixed seed as 42. To use a different one, use --seed <seed>')
            pl.seed_everything(42)
        early_stop_callback = False if self.hparams.disable_early_stopping else EarlyStopping(
            patience=self.hparams.early_stopping_patience,
            verbose=True
        )
        checkpoint_callback = False if self.hparams.disable_model_checkpoint else ModelCheckpoint(
            filepath=logger.log_dir,
            verbose=True
        )
        precision = 16 if self.hparams.half_precision else 32
        amp_level = 'O2' if self.hparams.half_precision else 'O1'
        return Trainer(
            logger=logger,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            max_epochs=self.hparams.num_epochs,
            gpus=self.hparams.gpu,
            auto_select_gpus=True,
            distributed_backend=distributed_backend,
            progress_bar_refresh_rate=5,
            num_sanity_val_steps=2,
            precision=precision,
            amp_level=amp_level,
            resume_from_checkpoint=self.hparams.resume_from_checkpoint,
            **kwargs
        )


if __name__ == '__main__':
    parser = MNSModule.build_parser()
    args = parser.parse_args()
    print(args)
    module = MNSModule(args)
    trainer = module.build_trainer()
    trainer.fit(module)
    print('Testing model from last epoch')
    trainer.test(module)
    path = trainer.checkpoint_callback.best_model_path
    module = module.load_from_checkpoint(path)
    print(f"Testing model from best epoch - path: {path}, score: {trainer.checkpoint_callback.best_model_score}")
    trainer.test(module)
