import functools

import torch
import torch.utils.data
import torch_geometric.datasets
from cyy_naive_lib.reflection import get_class_attrs, get_kwarg_names
from cyy_torch_toolbox import DatasetType
from cyy_torch_toolbox.dataset import global_dataset_collection_factory
from cyy_torch_toolbox.dataset.repository import register_dataset_constructors
from cyy_torch_toolbox.dataset.util import global_dataset_util_factor
from ogb.nodeproppred import PygNodePropPredDataset

try:
    import torch_geometric.data.data
    import torch_geometric.data.storage

    torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
    torch.serialization.add_safe_globals([torch_geometric.data.data.DataTensorAttr])
    torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])
except BaseException:
    pass

from .collection import GraphDatasetCollection
from .partition import METIS
from .util import GraphDatasetUtil

__all__ = ["GraphDatasetCollection", "METIS", "GraphDatasetUtil"]


def register_graph_dataset_constructors() -> None:
    dataset_constructors: dict = {}
    for repository in [torch_geometric.datasets]:
        dataset_constructors |= get_class_attrs(
            repository,
            filter_fun=lambda _, v: issubclass(v, torch.utils.data.Dataset),
        )
    for parent_dataset, sub_dataset_list in {
        "Planetoid": ["Cora", "CiteSeer", "PubMed"],
        "Coauthor": ["CS", "Physics"],
        "Amazon": ["Computers", "Photo"],
        "AttributedGraphDataset": ["TWeibo", "MAG"],
    }.items():
        assert parent_dataset in dataset_constructors
        constructor_kwargs = get_kwarg_names(dataset_constructors[parent_dataset])
        for name in sub_dataset_list:
            constructor = functools.partial(
                dataset_constructors[parent_dataset], name=name
            )
            if "split" in constructor_kwargs:
                constructor = functools.partial(constructor, split="full")
            dataset_constructors[f"{parent_dataset}_{name}"] = constructor

    for name, constructor in dataset_constructors.items():
        register_dataset_constructors(
            dataset_type=DatasetType.Graph, name=name, constructor=constructor
        )
    for name in ("ogbn-products",):
        constructor = functools.partial(PygNodePropPredDataset, name=name)
        register_dataset_constructors(
            dataset_type=DatasetType.Graph, name=name, constructor=constructor
        )


register_graph_dataset_constructors()
global_dataset_util_factor.register(DatasetType.Graph, GraphDatasetUtil)
global_dataset_collection_factory.register(DatasetType.Graph, GraphDatasetCollection)
