from cyy_torch_toolbox import DatasetCollection


class GraphDatasetCollection(DatasetCollection):
    def __init__(self, **kwargs):
        super().__init__(add_index=False, **kwargs)
