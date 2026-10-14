"""Package namespaces avoid unrelated top-level CLRS convenience imports.

Every executed algorithm/sampler/probing/evaluation source is unmodified.
No distributions, targets, TensorFlow substitutes, or algorithm implementations.
"""
import importlib
import sys
import types
from pathlib import Path

def load_official():
    root = Path(__file__).resolve().parent / 'official_source'
    for name, directory in [('clrs', root / 'clrs'), ('clrs._src', root / 'clrs/_src')]:
        if name in sys.modules:
            raise RuntimeError('Preexisting CLRS import cannot establish isolated provenance')
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        sys.modules[name] = module
    return tuple(importlib.import_module('clrs._src.' + n)
                 for n in ('samplers', 'specs', 'probing', 'evaluation', 'algorithms.sorting'))
