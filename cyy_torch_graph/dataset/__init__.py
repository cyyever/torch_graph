import functools

import torch
import torch.utils.data
import torch_geometric
from cyy_naive_lib.reflection import get_class_attrs
from cyy_torch_toolbox.dataset.util import global_dataset_util_factor
from cyy_torch_toolbox.dataset_collection.dataset_repository import \
    register_dataset_constructors
from cyy_torch_toolbox.ml_type import DatasetType

from .util import GraphDatasetUtil


def register_graph_dataset_constructors() -> None:
    dataset_constructors: dict = {}
    for repository in [torch_geometric.datasets]:
        dataset_constructors |= get_class_attrs(
            repository,
            filter_fun=lambda k, v: issubclass(v, torch.utils.data.Dataset),
        )
    if "Planetoid" in dataset_constructors:
        for repository in ["Cora", "CiteSeer", "PubMed"]:
            dataset_constructors[repository] = functools.partial(
                dataset_constructors["Planetoid"], name=repository, split="full"
            )
        for name in ["Cora", "CiteSeer", "PubMed"]:
            dataset_constructors[f"Planetoid_{name}"] = functools.partial(
                dataset_constructors["Planetoid"], name=name, split="full"
            )
    if "Coauthor" in dataset_constructors:
        for name in ["CS", "Physics"]:
            dataset_constructors[f"Coauthor_{name}"] = functools.partial(
                dataset_constructors["Coauthor"], name=name
            )

    for name, constructor in dataset_constructors.items():
        register_dataset_constructors(DatasetType.Graph, name, constructor)


register_graph_dataset_constructors()
global_dataset_util_factor.register(DatasetType.Graph, GraphDatasetUtil)
