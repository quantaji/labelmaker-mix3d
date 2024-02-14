import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import hydra
import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch
from mix3d.models.metrics import IoU
from omegaconf.listconfig import ListConfig
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch import nn
from torch.utils.data import DataLoader


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        self.idx2ds = []
        self.idx2dsidx = []

        for ds_idx, ds in enumerate(self.datasets):
            self.idx2ds += [ds_idx] * len(ds)
            self.idx2dsidx += list(range(len(ds)))

    def __getitem__(self, i):
        return self.datasets[self.idx2ds[i]][self.idx2dsidx[i]]

    def __len__(self):
        return len(self.idx2dsidx)


class SemanticSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label
        self.criterion = hydra.utils.instantiate(config.loss)
        # metrics
        # self.confusion = hydra.utils.instantiate(config.metrics)
        self.iou = IoU()
        # misc
        self.labels_info = dict()

    def forward(self, x):
        with self.optional_freeze():
            x = self.model(x)
        x = self.model.final(x)
        return x

    def training_step(self, batch, batch_idx, *args, **kwargs):
        garbage_collection_cuda()

        data, target = batch
        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )
        output = self.forward(data)
        loss = self.criterion(output.F, target).unsqueeze(0)

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        garbage_collection_cuda()

        data, target = batch

        inverse_maps = data.inverse_maps
        original_labels = data.original_labels
        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        output = self.forward(data)
        loss = self.criterion(output.F, target).unsqueeze(0)

        # getting original labels
        ordered_output = []
        for i in range(len(inverse_maps)):
            # https://github.com/NVIDIA/MinkowskiEngine/issues/119
            ordered_output.append(output.F[output.C[:, 0] == i])
        output = ordered_output
        for i, (out, inverse_map) in enumerate(zip(output, inverse_maps)):
            out = out.max(1)[1].view(-1).detach().cpu()
            output[i] = out[inverse_map].numpy()

        self.confusions[dataloader_idx].add(np.hstack(output), np.hstack(original_labels))

        return {
            "val_loss": loss,
        }

    def training_epoch_end(self, outputs):
        train_loss = torch.cat([out["loss"] for out in outputs], dim=0).mean()
        results = {"train_loss": train_loss}
        self.log_dict(results)

        garbage_collection_cuda()

    def validation_epoch_end(self, outputs):

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        val_losses = [torch.cat([out["val_loss"] for out in output], dim=0).mean() for output in outputs]
        results = {}
        for i, val_loss in enumerate(val_losses):
            results[f"val_loss_{i}"] = val_loss

        num_val_loaders = len(outputs)
        for j in range(num_val_loaders):
            confusion_matrix = self.confusions[j].value()
            results_iou = self.iou.value(confusion_matrix)
            for i, k in enumerate(self.labels_info.keys()):
                metric_name = self.labels_info[k]["name"]
                results[f"val_IoU_{j}_" + metric_name] = results_iou[i]
            results[f"val_IoU_{j}"] = np.nanmean(results_iou)

            self.confusions[j].reset()

        self.log_dict(results)

        garbage_collection_cuda()

    def test_step(self, batch, batch_idx, *args, **kwargs):
        data, target = batch
        inverse_maps = data.inverse_maps
        original_labels = data.original_labels
        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )
        data
        output = self.forward(data)
        loss = 0
        if original_labels[0].size > 0:
            loss = self.criterion(output.F, target).unsqueeze(0)
            target = target.detach().cpu()
        original_predicted = []
        for i in range(len(inverse_maps)):
            # https://github.com/NVIDIA/MinkowskiEngine/issues/119
            out = output.F[output.C[:, 0] == i]
            out = out.max(1)[1].view(-1).detach().cpu()
            original_predicted.append(out[inverse_maps[i]].numpy())

        return {
            "test_loss": loss,
            "metrics": {
                "original_output": original_predicted,
                "original_label": original_labels,
            },
        }

    def test_epoch_end(self, outputs):
        outs = []
        labels = []
        results = {}

        results = dict()
        for out in outputs:
            outs.extend(out["metrics"]["original_output"])
            labels.extend(out["metrics"]["original_label"])
        if labels[0].size > 0:
            self.confusion.reset()
            self.confusion.add(np.hstack(outs), np.hstack(labels))
            confusion_matrix = self.confusion.value()
            results_iou = self.iou.value(confusion_matrix)
            for i, k in enumerate(self.labels_info.keys()):
                metric_name = self.labels_info[k]["name"]
                results["test_IoU_" + metric_name] = results_iou[i]
            results["test_IoU"] = results_iou.mean()

        if self.test_dataset.data[0].get("scene"):
            results = scannet_submission(
                self.trainer.weights_save_path,
                outs,
                results,
                self.test_dataset.data,
                self.test_dataset._remap_model_output,
            )
        self.log_dict(results)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.parameters())
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(self.train_dataloader())
        lr_scheduler = hydra.utils.instantiate(self.config.scheduler.scheduler, optimizer=optimizer)
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        train_dataset_list = [hydra.utils.instantiate(conf) for conf in self.config.data.train_dataset]
        self.train_dataset = ConcatDataset(train_dataset_list)

        self.validation_dataset = [hydra.utils.instantiate(conf) for conf in self.config.data.validation_dataset]

        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)

        length_max = max(len(self.train_dataset), len(self.validation_dataset))

        self.confusions = [hydra.utils.instantiate(self.config.metrics) for _ in range(length_max)]
        self.confusion = hydra.utils.instantiate(self.config.metrics)

        self.labels_info = train_dataset_list[0].label_info

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return [
            hydra.utils.instantiate(
                self.config.data.validation_dataloader,
                dataset,
                collate_fn=c_fn,
            )
            for dataset in self.validation_dataset
        ]

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )


def scannet_submission(path, outputs, results, test_filebase, remap_function):
    # checking if we are submitting scannet
    filepaths = []
    for out, info in zip(outputs, test_filebase):
        save_path = Path(path).parent / "submission" / f"scene{info['scene']:04}_{info['sub_scene']:02}.txt"
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        out = remap_function(out)
        np.savetxt(save_path, out, fmt="%d")
        filepaths.append(save_path)

    zip_path = save_path.parent.parent / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as myzip:
        for file in filepaths:
            myzip.write(file)
    results["submission_path"] = str(zip_path)
    return results
