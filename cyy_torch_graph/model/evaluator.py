from typing import Any

import torch
import torch_geometric.nn
import torch_geometric.utils
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import (DatasetCollection, MachineLearningPhase,
                               ModelEvaluator)
from cyy_torch_toolbox.tensor import tensor_to


class GraphModelEvaluator(ModelEvaluator):
    def __init__(self, dataset_collection: DatasetCollection, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dc = dataset_collection
        self.__masks: dict = {}
        get_logger().debug("use neighbour_hop %s", self.neighbour_hop)

    @property
    def neighbour_hop(self):
        return torch_geometric.utils.get_num_hops(self.model)

    def __call__(self, **kwargs: Any) -> dict:
        assert "batch_node_indices" not in kwargs
        return self.__from_neighbor_loader(**kwargs)

    def __from_neighbor_loader(self, **kwargs: Any) -> dict:
        n_id = kwargs.pop("n_id")
        kwargs["n_id"] = n_id
        inputs = {"edge_index": kwargs["edge_index"], "x": kwargs["x"], "n_id": n_id}
        if kwargs["y"].shape[0] != kwargs["x"].shape[0]:
            assert (
                kwargs["y"].shape[0]
                == self.__dc.get_dataset_util(phase=kwargs["phase"])
                .get_original_graph(0)
                .y.shape[0]
            )
            kwargs["y"] = kwargs["y"].index_select(0, n_id)
        batch_mask = self.get_mask(phase=kwargs["phase"], device=kwargs["device"])[n_id]

        y = tensor_to(
            kwargs["y"], device=kwargs["device"], non_blocking=kwargs["non_blocking"]
        )
        kwargs["targets"] = y[batch_mask]
        kwargs["batch_mask"] = batch_mask
        return super().__call__(inputs=inputs, **kwargs)

    def get_mask(self, phase: MachineLearningPhase, device) -> torch.Tensor:
        mask = self.__masks.get(phase, None)
        if mask is None:
            mask = self.__dc.get_dataset_util(phase=phase).get_mask()[0]
            self.__masks[phase] = mask.to(device=device, non_blocking=True)
        assert mask is not None
        return mask

    def _compute_loss(self, output: torch.Tensor, **kwargs: Any) -> dict:
        extra_res = {}
        n_id = kwargs.pop("n_id")
        batch_mask = kwargs.pop("batch_mask")
        if kwargs.pop("need_sample_indices", False):
            extra_res = {"sample_indices": n_id[batch_mask].tolist()}

        return super()._compute_loss(output=output[batch_mask], **kwargs) | extra_res
