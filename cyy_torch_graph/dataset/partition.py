import torch
from torch import Tensor
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.sparse import index2ptr


def METIS(
    edge_index: Tensor,
    num_nodes: int,
    num_parts: int,
    edge_weight: torch.Tensor | None = None,
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
    row, col = sort_edge_index(edge_index, num_nodes=num_nodes)
    rowptr = index2ptr(row, size=num_nodes)

    # Compute METIS partitioning:
    cluster = pyg_lib.partition.metis(
        rowptr=rowptr.cpu(),
        col=col.cpu(),
        num_partitions=num_parts,
        edge_weight=edge_weight,
    ).to(edge_index.device)
    return cluster
