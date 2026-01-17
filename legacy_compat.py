import sys
from llm_lab import config, model, utils

# Shim for modules that were moved
sys.modules['config'] = config
sys.modules['model'] = model
sys.modules['utils'] = utils
