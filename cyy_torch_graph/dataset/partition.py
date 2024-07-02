import torch
from cyy_naive_lib.log import log_debug
from torch import Tensor
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import is_undirected, sort_edge_index, to_undirected
from torch_geometric.utils.sparse import index2ptr


def METIS(
    edge_index: Tensor,
    num_nodes: int,
    num_parts: int,
    edge_weights: dict | None | torch.Tensor = None,
) -> Tensor:
    r"""Partitions a graph data object into multiple subgraphs""
    .. note::

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
    """
    # Computes a node-level partition assignment vector via METIS.

    # Calculate CSR representation:
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    row, col = sort_edge_index(edge_index, num_nodes=num_nodes)
    col = col.cpu()
    rowptr = index2ptr(row, size=num_nodes).cpu()
    if edge_weights is not None:
        weight_list = []
        assert isinstance(edge_weights, dict)
        rowptr_list = rowptr.tolist()
        col_list = col.tolist()
        log_debug("process col %s %s", col.shape, num_nodes)
        for src in range(num_nodes):
            for index in range(rowptr_list[src], rowptr_list[src + 1]):
                dest = col_list[index]
                if (src, dest) in edge_weights:
                    weight_list.append(edge_weights[(src, dest)])
                else:
                    weight_list.append(edge_weights[(dest, src)])
        edge_weights = torch.tensor(weight_list, dtype=torch.long)
        log_debug("edge_weights shape %s", edge_weights.shape)

    # Compute METIS partitioning:
    return pyg_lib.partition.metis(
        rowptr=rowptr,
        col=col,
        num_partitions=num_parts,
        edge_weight=edge_weights,
        recursive=True,
    )
