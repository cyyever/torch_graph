import copy
from collections.abc import Iterable
from typing import Any, Generator

import torch
import torch.utils
import torch_geometric.data
import torch_geometric.utils
from cyy_torch_toolbox.dataset.util import DatasetUtil
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class GraphDatasetUtil(DatasetUtil):
    def get_mask(self) -> list[torch.Tensor]:
        if hasattr(self.dataset[0], "mask") or "mask" in self.dataset[0]:
            return [dataset["mask"] for dataset in self.dataset]
        mask = torch.ones((self.get_original_graph(0).x.shape[0],), dtype=torch.bool)
        return [mask]

    def get_raw_samples(self, indices: Iterable | None = None) -> Generator:
        if indices is not None:
            indices = set(indices)
        mask = self.get_mask()
        assert len(mask) == 1
        graph = self.get_original_graph(0)
        for idx, flag in enumerate(mask[0].tolist()):
            if not flag:
                continue
            if indices is None or idx in indices:
                yield idx, {
                    "target": graph.y[idx],
                    "index": idx,
                }

    def get_edge_index(self, graph_index: int) -> torch.Tensor:
        graph = self.dataset[graph_index]
        if "edge_index" in graph:
            return graph["edge_index"]
        return self.get_original_graph(graph_index).edge_index

    def get_graph(self, graph_index: int) -> Any:
        original_graph = self.get_original_graph(graph_index=graph_index)
        edge_index = self.get_edge_index(graph_index=graph_index)
        graph_dict = original_graph.to_dict()
        assert "edge_index" in graph_dict
        graph_dict["edge_index"] = edge_index
        return type(original_graph)(**graph_dict)

    def get_original_graph(self, graph_index: int) -> Any:
        graph_dict = self.dataset[graph_index]
        if "original_dataset" not in graph_dict:
            return graph_dict
        original_dataset = graph_dict["original_dataset"]
        graph_index = graph_dict["graph_index"]
        return original_dataset[graph_index]

    def get_subset(self, indices: Iterable) -> list[dict]:
        return self.get_node_subset(indices)

    def get_node_subset(self, node_indices: Iterable | torch.Tensor) -> list[dict]:
        assert node_indices
        node_indices = torch.tensor(list(node_indices))
        result = []
        for idx, graph_dict in enumerate(self.dataset):
            graph = self.get_original_graph(idx)
            if isinstance(graph_dict, dict):
                tmp = graph_dict.copy()
            else:
                tmp = {
                    "graph_index": idx,
                    "original_dataset": self.dataset,
                }
            tmp["mask"] = torch_geometric.utils.index_to_mask(
                node_indices, size=graph.x.shape[0]
            )
            result.append(tmp)
        return result

    def get_edge_subset(self, graph_index: int, edge_index: torch.Tensor) -> list[dict]:
        dataset = copy.copy(self.dataset)
        dataset[graph_index]["edge_index"] = edge_index
        return dataset

    def decompose(self) -> None | dict:
        mapping: dict = {
            MachineLearningPhase.Training: "train_mask",
            MachineLearningPhase.Validation: "val_mask",
            MachineLearningPhase.Test: "test_mask",
        }
        if not all(
            hasattr(self.dataset[0], mask_name) for mask_name in mapping.values()
        ):
            return None
        datasets: dict = {}
        for phase, mask_name in mapping.items():
            datasets[phase] = []
            for idx, graph in enumerate(self.dataset):
                datasets[phase].append(
                    {
                        "mask": getattr(graph, mask_name),
                        "graph_index": idx,
                        "original_dataset": self.dataset,
                    }
                )
        return datasets
