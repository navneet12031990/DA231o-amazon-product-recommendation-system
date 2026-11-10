"""Tiny helper for the infra shell scripts: print config values (CONFIG env var, default config.yaml)."""
import os
import sys

import yaml

cfg = yaml.safe_load(open(os.environ.get("CONFIG", "config.yaml")))
values = []
for dotted in sys.argv[1:]:
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    values.append("" if node is None else str(node))
print("|".join(values))  # '|' keeps empty values (e.g. service_account) in place for `read`
