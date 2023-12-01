import os
import sys

from .dataset import register_graph_dataset_constructors

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
if module_dir not in sys.path:
    sys.path.append(module_dir)

register_graph_dataset_constructors()
