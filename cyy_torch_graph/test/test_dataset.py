import cyy_torch_graph  # noqa: F401
from cyy_torch_toolbox.dataset import create_dataset_collection


def test_dataset() -> None:
    create_dataset_collection(
        "Cora",
    )
