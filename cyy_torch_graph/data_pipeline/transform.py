from typing import Any

from cyy_torch_toolbox.data_pipeline.common import default_data_extraction


def pyg_data_extraction(data: Any) -> dict:
    assert "input" not in data
    match data:
        case {
            "graph_index": graph_index,
            "original_dataset": original_dataset,
        }:
            graph = original_dataset[graph_index]
            return data | {
                "input": graph,
                "target": graph.y,
            }
    return default_data_extraction(data)
