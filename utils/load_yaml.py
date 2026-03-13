import yaml
from pathlib import Path
from typing import Union, Dict, Any


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

