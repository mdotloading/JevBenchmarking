"""Model registry: one name per model family. Imports are lazy, so only the family you use is loaded."""
import importlib

FAMILIES = {
    "jev": "src.models.jev:JevAPI",
    "jpt": "src.models.jpt:JPT",
    "decision2": "src.models.decision2:Decision2",
}


def load_model(family, checkpoint=None):
    """A ready model: the plain release (checkpoint=None) or a fine-tuned checkpoint folder."""
    module, cls = FAMILIES[family].split(":")
    model_class = getattr(importlib.import_module(module), cls)
    return model_class() if checkpoint is None else model_class(checkpoint)
