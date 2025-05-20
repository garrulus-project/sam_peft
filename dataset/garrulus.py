import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
import kornia.augmentation as K
from gdl.datasets.benchmark import get_field_D_grid_split
from gdl.datasets.polygon import PolygonSplitter
from gdl.samplers.single import RandomAoiGeoSampler, GridAoiSampler
from gdl.datasets.geo import GarrulusSegmentationDataset


class GarrulusSegmentationDatasetNew(GarrulusSegmentationDataset):
    def plot(
        self,
        sample,
        show_mask=True,
        show_prediction=True,
        show_titles=True,
        suptitle=None,
        prediction_title="Predictions",
    ):
        """Plot a sample from the dataset.

        Args:
            sample (dict): A sample returned by RasterDataset.__getitem__.
            show_titles (bool): Flag indicating whether to show titles above each panel.
            suptitle (str | None): Optional string to use as a suptitle.

        Returns:
            fig (matplotlib.figure.Figure): A matplotlib Figure with the rendered sample.
        """
        if sample["image"].shape[1] > 3:
            rgb_indices = []
            for band in self.image.rgb_bands:
                if band in self.image.bands:
                    rgb_indices.append(self.image.bands.index(band))
                else:
                    raise ValueError("RGB band does not include all RGB bands")

            image = sample["image"][rgb_indices].permute(1, 2, 0)
        else:
            image = sample["image"].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        if show_mask:
            mask = sample["mask"].numpy().astype("uint8").squeeze()

        num_panels = 2 if show_mask else 1

        if show_prediction and "prediction" in sample:
            predictions = sample["prediction"]
            if not isinstance(predictions, np.ndarray):
                predictions = predictions.numpy().astype("uint8").squeeze()
            pred_panel_idx = num_panels
            num_panels += 1

        if "combined_gt_and_pred" in sample:
            combined_gt_pred_panel_idx = num_panels
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")

        if show_mask:
            axs[1].imshow(mask, vmin=0, vmax=self._cmap.N - 1, cmap=self._cmap)
            axs[1].axis("off")

        # Show legend
        # if show_mask or show_prediction:
        if show_mask:
            legend_elements = [
                Patch(
                    facecolor=[
                        self.cmap[i][0] / 255.0,
                        self.cmap[i][1] / 255.0,
                        self.cmap[i][2] / 255.0,
                    ],
                    edgecolor="none",
                    label=self.label_names[i],
                )
                for i in self.cmap
            ]
            axs[1].legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=2,
            )

        if show_titles:
            axs[0].set_title("Image")
            if show_mask:
                axs[1].set_title("Groundtruth Mask")

        if show_prediction and "prediction" in sample:
            # panel_idx = num_panels - 2
            # panel_idx -= 1
            axs[pred_panel_idx].imshow(
                predictions, vmin=0, vmax=self._cmap.N - 1, cmap=self._cmap
            )
            # axs[pred_panel_idx].imshow(image)
            # axs[pred_panel_idx].imshow(predictions)
            axs[pred_panel_idx].axis("off")
            legend_elements = [
                Patch(
                    facecolor=[
                        self.cmap[i][0] / 255.0,
                        self.cmap[i][1] / 255.0,
                        self.cmap[i][2] / 255.0,
                    ],
                    edgecolor="none",
                    label=self.label_names[i],
                )
                for i in self.cmap
            ]
            axs[pred_panel_idx].legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=2,
            )

            if show_titles:
                axs[pred_panel_idx].set_title(f"{prediction_title}")

        if "combined_gt_and_pred" in sample:
            # panel_idx = num_panels - 1
            axs[combined_gt_pred_panel_idx].imshow(sample["combined_gt_and_pred"])
            axs[combined_gt_pred_panel_idx].axis("off")
            if show_titles:
                axs[combined_gt_pred_panel_idx].set_title("True positive masks")

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig


def get_train_val(
    batch_size=1,
    img_size=1024,
    worker_init_fn=None,
    seed=2025,
    min_window_res=None,
    save_sampled_dataset=False,
):
    grid_path = "/home/jovyan/garrulus/field-D/grid-10m-squares/grid-10m-squares.shp"
    fenced_area_path = "/home/jovyan/garrulus/field-D/boundary-shape/boundary-shape.shp"
    raster_image_root_path = "/home/jovyan/garrulus/field-D"
    mask_root_path = "/home/jovyan/garrulus/field-D/d-RGB-9mm-mask"

    # create dataset split
    ps = PolygonSplitter(grid_path, fenced_area_path)
    train_indices, validation_indices, test_indices = get_field_D_grid_split()

    train_polygon = ps.get_polygon_by_indices(grid_indices=train_indices)
    validation_polygon = ps.get_polygon_by_indices(grid_indices=validation_indices)
    test_polygon = ps.get_polygon_by_indices(grid_indices=test_indices)

    train_polygon = train_polygon + validation_polygon

    # set the same value for min and max to produce consisten window size
    size_lims = (img_size, img_size)
    if min_window_res is not None:
        size_lims = (min_window_res, img_size)

    transforms = K.AugmentationSequential(
        K.Resize(img_size),
        K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
        data_keys=None,
    )

    ## ToDo: update gdl to handle different transform
    # train_transform = K.AugmentationSequential(
    #     K.Resize(img_size),
    #     K.RandomHorizontalFlip(p=1.0),
    #     K.RandomVerticalFlip(p=1.0),
    #     K.RandomRotation(degrees=90, p=1.0),
    #     K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
    #     data_keys=None,
    # )
    # val_transform = K.AugmentationSequential(
    #     K.Resize(img_size),
    #     K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
    #     data_keys=None,
    # )

    # create Garrulus segmentation dataset
    gsd = GarrulusSegmentationDatasetNew(
        raster_image_paths=raster_image_root_path,
        mask_paths=mask_root_path,
        grid_shape_path=grid_path,
        transforms=transforms,
    )

    # create RandomAoiSampler, this method takes the samples only within the train_polygon
    random_train_sampler = RandomAoiGeoSampler(
        gsd,
        size_lims=size_lims,
        polygons=train_polygon,
        length=1000,
        outer_boundary_shape=fenced_area_path,
    )
    # create PyTorch dataloader
    if worker_init_fn is not None:
        train_loader = DataLoader(
            gsd,
            sampler=random_train_sampler,
            collate_fn=stack_samples,
            batch_size=batch_size,
            num_workers=8,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed=seed),
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            gsd,
            sampler=random_train_sampler,
            collate_fn=stack_samples,
            num_workers=8,
            shuffle=False,
            batch_size=batch_size if not save_sampled_dataset else 1,
            drop_last=True if not save_sampled_dataset else False,
        )

    grid_val_sampler = GridAoiSampler(
        gsd,
        size=img_size,
        polygons=test_polygon,
        outer_boundary_shape=fenced_area_path,
        polygon_intersection=0.5,
        window_overlap=0.25,
    )
    # create PyTorch dataloader
    # use batch size 1 for val loader to avoid OOM on 24GB gpu
    val_loader = DataLoader(
        gsd,
        sampler=grid_val_sampler,
        collate_fn=stack_samples,
        batch_size=1,
        shuffle=False,
        drop_last=True if not save_sampled_dataset else False,
    )

    if save_sampled_dataset:
        sampled_train_data = {
            "crs": [],
            "bbox": [],
            "image": [],
            "mask": [],
        }
        for batch in train_loader:
            for key, value in batch.items():
                sampled_train_data[key].append(value)

        torch.save(sampled_train_data, "./sampled_train_data.pt")

        # test data
        sampled_test_data = {
            "crs": [],
            "bbox": [],
            "image": [],
            "mask": [],
        }
        for batch in val_loader:
            for key, value in batch.items():
                sampled_test_data[key].append(value)

        torch.save(sampled_test_data, "./sampled_test_data.pt")

    return gsd, train_loader, val_loader


class GarrulusDatasetICRA:
    """
    This will load pre-generated and sampled Garrulus Field-D
    dataset used for ICRA2025.
    """

    label_names = {
        0: "BACKGROUND",
        1: "CWD",
        2: "MISC",
        3: "STUMP",
        4: "VEGETATION",
        # 5: 'CUT',
    }

    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (100, 100, 78, 255),
        3: (200, 115, 0, 255),
        4: (76, 230, 0, 255),
        # 5: (163, 255, 115, 255),
    }

    def __init__(self, sampled_dataset_path=""):
        self.data_dict = torch.load(sampled_dataset_path, weights_only=False)
        self.keys = list(self.data_dict.keys())
        self.length = len(self.data_dict[self.keys[0]])
        self.colors = [
            (self.cmap[i][0] / 255.0, self.cmap[i][1] / 255.0, self.cmap[i][2] / 255.0)
            for i in range(len(self.cmap))
        ]
        self.listed_cmap = ListedColormap(self.colors)

    def __getitem__(self, idx):
        return {
            key: (
                self.data_dict[key][idx].squeeze()
                if key in ["image", "mask"]
                else self.data_dict[key][idx]
            )
            for key in self.keys
        }

    def __len__(self):
        return self.length

    def plot_sample(
        self,
        sample,
        show_mask=True,
        show_prediction=True,
        show_titles=True,
        suptitle=None,
        prediction_title="Predictions",
    ):
        """Plot a sample from the dataset.

        Args:
            sample (dict): A sample returned by RasterDataset.__getitem__.
            show_titles (bool): Flag indicating whether to show titles above each panel.
            suptitle (str | None): Optional string to use as a suptitle.

        Returns:
            fig (matplotlib.figure.Figure): A matplotlib Figure with the rendered sample.
        """

        image = sample["image"].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        if show_mask:
            mask = sample["mask"].numpy().astype("uint8").squeeze()

        num_panels = 2 if show_mask else 1

        if show_prediction and "prediction" in sample:
            predictions = sample["prediction"]
            if not isinstance(predictions, np.ndarray):
                predictions = predictions.numpy().astype("uint8").squeeze()
            pred_panel_idx = num_panels
            num_panels += 1

        if "combined_gt_and_pred" in sample:
            combined_gt_pred_panel_idx = num_panels
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")

        if show_mask:
            axs[1].imshow(
                mask, vmin=0, vmax=self.listed_cmap.N - 1, cmap=self.listed_cmap
            )
            axs[1].axis("off")

        # Show legend
        # if show_mask or show_prediction:
        if show_mask:
            legend_elements = [
                Patch(
                    facecolor=[
                        self.cmap[i][0] / 255.0,
                        self.cmap[i][1] / 255.0,
                        self.cmap[i][2] / 255.0,
                    ],
                    edgecolor="none",
                    label=self.label_names[i],
                )
                for i in self.cmap
            ]
            axs[1].legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=2,
            )

        if show_titles:
            axs[0].set_title("Image")
            if show_mask:
                axs[1].set_title("Groundtruth Mask")

        if show_prediction and "prediction" in sample:
            # panel_idx = num_panels - 2
            # panel_idx -= 1
            axs[pred_panel_idx].imshow(
                predictions, vmin=0, vmax=self.listed_cmap.N - 1, cmap=self.listed_cmap
            )
            # axs[pred_panel_idx].imshow(image)
            # axs[pred_panel_idx].imshow(predictions)
            axs[pred_panel_idx].axis("off")
            legend_elements = [
                Patch(
                    facecolor=[
                        self.cmap[i][0] / 255.0,
                        self.cmap[i][1] / 255.0,
                        self.cmap[i][2] / 255.0,
                    ],
                    edgecolor="none",
                    label=self.label_names[i],
                )
                for i in self.cmap
            ]
            axs[pred_panel_idx].legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=2,
            )

            if show_titles:
                axs[pred_panel_idx].set_title(f"{prediction_title}")

        if "combined_gt_and_pred" in sample:
            # panel_idx = num_panels - 1
            axs[combined_gt_pred_panel_idx].imshow(sample["combined_gt_and_pred"])
            axs[combined_gt_pred_panel_idx].axis("off")
            if show_titles:
                axs[combined_gt_pred_panel_idx].set_title("True positive masks")

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig
