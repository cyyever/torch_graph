from typing import Any

from cyy_torch_toolbox import DatasetCollection


class GraphDatasetCollection(DatasetCollection):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(add_index=False, **kwargs)
