import math

import torch
import torch.utils.data
import torch_geometric
import torch_geometric.utils
from cyy_torch_toolbox import (DatasetCollection, DatasetType,
                               MachineLearningPhase, TransformType)
from cyy_torch_toolbox.data_pipeline import global_data_transform_factory
from cyy_torch_toolbox.data_pipeline.loader import global_dataloader_factory
from torch_geometric.loader import NeighborLoader

from ..dataset import GraphDatasetUtil
from ..model import GraphModelEvaluator
from .pyg_dataloader import RandomNodeLoader
from .transform import pyg_data_extraction


def get_dataloader(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    model_evaluator: GraphModelEvaluator,
    **kwargs,
) -> torch.utils.data.DataLoader:
    util = dc.get_dataset_util(phase=phase)
    assert isinstance(util, GraphDatasetUtil)
    pyg_input_nodes = kwargs.pop("pyg_input_nodes", {})
    if pyg_input_nodes:
        input_nodes = pyg_input_nodes[phase]
    else:
        input_nodes = torch_geometric.utils.mask_to_index(util.get_mask()[0])

    if not kwargs.get("sample_neighbor", True):
        return RandomNodeLoader(node_indices=input_nodes.tolist(), **kwargs)

    if "batch_number" in kwargs:
        batch_number = kwargs.pop("batch_number")
        assert batch_number > 0
        input_number = input_nodes.numel()
        assert input_number >= batch_number
        ensure_batch_size_cover = kwargs.pop("ensure_batch_size_cover", False)
        while True:
            kwargs["batch_size"] = math.ceil(input_number / batch_number)
            assert kwargs["batch_size"] >= 1
            while kwargs["batch_size"] * (batch_number - 1) >= input_number:
                kwargs["batch_size"] -= 1
                assert kwargs["batch_size"] >= 1
            if kwargs["batch_size"] * batch_number >= input_number:
                break
            if not ensure_batch_size_cover:
                kwargs["drop_last"] = True
                break
            batch_number -= 1
            assert batch_number > 0
    return NeighborLoader(
        data=util.get_graph(0),
        num_neighbors=[kwargs.pop("num_neighbor", 10)] * model_evaluator.neighbour_hop,
        input_nodes=input_nodes,
        transform=lambda data: data.to_dict(),
        **kwargs,
    )


global_dataloader_factory.register(DatasetType.Graph, get_dataloader)


def append_transforms_to_dc(dc, model_evaluator=None) -> None:
    if model_evaluator is None:
        for _, transform in dc.foreach_transform():
            transform.clear(TransformType.ExtractData)
            transform.append(
                key=TransformType.ExtractData, transform=pyg_data_extraction
            )


global_data_transform_factory.register(DatasetType.Graph, append_transforms_to_dc)
