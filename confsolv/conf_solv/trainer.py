import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from .model.model import ConfSolv
import numpy as np


class LitConfSolvModule(pl.LightningModule):
    def __init__(self, config):
        super(LitConfSolvModule, self).__init__()

        self.save_hyperparameters()
        self.model = ConfSolv(config)
        self.config = config
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.relative_loss = config["relative_loss"]
        self.relative_model = config["relative_model"]
        self.max_confs = config["max_confs"]
        self.debug = config["debug"]
        self.max_confs = config["max_confs"]

    def forward(self, data, max_confs):
        return self.model(data, max_confs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                               patience=5, min_lr=self.lr / 100)
        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}

    def _step(self, data, max_confs, batch_idx, mode):
        out = self(data, max_confs)

        try:
            scaler = self.trainer.datamodule.scaler.to(self.device)
        except AttributeError:
            scaler = self.scaler.to(self.device)

        if self.relative_loss:
            y = data.y.view(-1, max_confs).unsqueeze(-1) - \
                data.y.view(-1, max_confs).unsqueeze(-2)
            mask = data.solute_mask.view(-1, max_confs).unsqueeze(-1) * \
                   data.solute_mask.view(-1, max_confs).unsqueeze(-2)
            mask = torch.stack([m.fill_diagonal_(False) for m in mask])

            if not self.relative_model:
                out = out.view(-1, max_confs).unsqueeze(-1) - \
                      out.view(-1, max_confs).unsqueeze(-2)

            # divide by all non-zero entries (non-diagonal true entries in mask)
            normalized_coeff = mask.sum()

        else:
            y = data.y
            mask = data.solute_mask
            normalized_coeff = data.solute_mask.sum()

        y = y * mask
        scaled_y = scaler.transform(y) * mask
        out = out * mask
        pred = scaler.inverse_transform(out) * mask

        loss = F.mse_loss(out, scaled_y,  reduction="sum") / normalized_coeff
        batch_size = len(data.mol_id)
        rmse = torch.sqrt(torch.sum((pred-y)**2) / normalized_coeff)
        mae = torch.abs(pred-y).sum() / normalized_coeff

        # logs
        if mode != 'predict':
            # Logging is disabled in the predict hooks
            self.log(f'{mode}_loss', loss, batch_size=batch_size)
            self.log(f'{mode}_rmse', rmse, batch_size=batch_size)
            self.log(f'{mode}_mae', mae, batch_size=batch_size)

        return {'loss': loss, 'preds': pred.detach(), 'target': y.detach()}

    def training_step(self, data, batch_idx):
        loss_dict = self._step(data, self.max_confs, batch_idx, mode="train")
        return loss_dict

    def validation_step(self, data, batch_idx):
        loss_dict = self._step(data, self.max_confs, batch_idx, mode="val")
        return loss_dict

    def test_step(self, data, batch_idx):
        loss_dict = self._step(data, len(data.y), batch_idx, mode="test")
        return loss_dict

    def predict_step(self, data, batch_idx):
        loss_dict = self._step(data, len(data.y), batch_idx, mode="predict")
        return loss_dict
    
    def on_save_checkpoint(self, checkpoint) -> None:
        "Objects to include in checkpoint file"
        checkpoint["scaler"] = self.trainer.datamodule.scaler

    def on_load_checkpoint(self, checkpoint) -> None:
        "Objects to retrieve from checkpoint file"
        self.scaler = checkpoint["scaler"]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("models")

        # DataLoader args
        parser.add_argument('--solvent', type=str, default=None)
        parser.add_argument('--exclude_solvent', action='store_true', default=False)

        # solvent args
        parser.add_argument('--solvent_depth', type=int, default=3)
        parser.add_argument('--solvent_hidden_dim', type=int, default=200)
        parser.add_argument('--solvent_n_layers', type=int, default=2)
        parser.add_argument('--solvent_gnn_type', type=str, default='dmpnn')
        parser.add_argument('--solvent_dropout', type=float, default=0.0)

        # 2D solute args
        parser.add_argument('--solute_depth', type=int, default=3)
        parser.add_argument('--solute_hidden_dim', type=int, default=200)
        parser.add_argument('--solute_n_layers', type=int, default=2)
        parser.add_argument('--solute_gnn_type', type=str, default='dmpnn')
        parser.add_argument('--solute_dropout', type=float, default=0.0)

        # 3D model options
        parser.add_argument('--solute_model', type=str, default='DimeNet')
        parser.add_argument('--use_scaling', action='store_true', default=False)
        parser.add_argument('--diff_type', type=str, default='atom_diff')

        # 3D model general args
        parser.add_argument('--num_spherical', type=int, default=6)
        parser.add_argument('--num_radial', type=int, default=6)
        parser.add_argument('--hidden_channels', type=int, default=128)
        parser.add_argument('--ffn_hidden_dim', type=float, default=500)
        parser.add_argument('--ffn_n_layers', type=int, default=4)
        parser.add_argument('--num_blocks', type=int, default=2)

        # 3D DimeNet solute args
        parser.add_argument('--dn_num_bilinear', type=int, default=8)

        # 3D SchNet model args
        parser.add_argument('--sn_num_filters', type=int, default=128)
        parser.add_argument('--sn_num_gaussians', type=int, default=50)
        parser.add_argument('--sn_num_neighbours', type=int, default=32)
        parser.add_argument('--sn_cutoff', type=float, default=10.0)
        parser.add_argument('--sn_readout', type=str, default='add')
        parser.add_argument('--sn_bool', type=bool, default=False)

        # 3D SphereNet model args
        parser.add_argument('--cutoff', type=float, default=5.0)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--int_emb_size', type=int, default=64)
        parser.add_argument('--basis_emb_size_dist', type=int, default=8)
        parser.add_argument('--basis_emb_size_angle', type=int, default=8)
        parser.add_argument('--basis_emb_size_torsion', type=int, default=8)
        parser.add_argument('--out_emb_channels', type=int, default=256)
        return parent_parser

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("training")
        parser.add_argument("--gpus", type=int, default=0)
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--warmup_epochs', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--max_confs', type=int, default=10)
        parser.add_argument('--relative_model', action='store_true', default=False)
        parser.add_argument('--relative_loss', action='store_true', default=False)
        parser.add_argument('--n_training_points', type=int, default=None)
        parser.add_argument('--threshold', type=int, default=100)
        parser.add_argument('--scaler_type', type=str, default=None)
        parser.add_argument('--no_ionic_solvents', action='store_true', default=False)
        parser.add_argument('--param_name', type=str, default='dG(solution)')

        # debugging
        parser.add_argument('--profile', action='store_true', default=False)
        parser.add_argument('--tune', action='store_true', default=False)

        return parent_parser

    @staticmethod
    def add_program_args(parent_parser):
        parser = parent_parser.add_argument_group("program")
        parser.add_argument('--log_dir', type=str, default='./test')
        parser.add_argument('--coords_path', type=str, default='data/debug/coords.pkl.gz')
        parser.add_argument('--energies_path', type=str, default='data/debug/free_energy.pkl.gz')
        parser.add_argument('--split_path', type=str, default='data/debug/split_0.npy')
        parser.add_argument('--restart_path', type=str, default=None,
                            help="Path to latest model .ckpt file")
        parser.add_argument('--checkpoint', type=str, default=None,
                            help="Path to the model .ckpt file for loading and overwriting weights")
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--verbose', action='store_true', default=False)
        parser.add_argument('--debug', action='store_true', default=False)
        return parent_parser

    @classmethod
    def add_args(cls, parent_parser):
        parser = cls.add_program_args(parent_parser)  # program options
        parser = cls.add_argparse_args(parser)  # trainer options
        parser = cls.add_model_specific_args(parser)  # models specific args
        return parser
