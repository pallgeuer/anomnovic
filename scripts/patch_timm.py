#!/usr/bin/env python3
"""Apply the tested timm 0.3.2 import fix inside this Python environment."""
import sys
import sysconfig
from pathlib import Path


environment = Path(sys.prefix).resolve()
helpers = Path(sysconfig.get_paths()['purelib']) / 'timm' / 'models' / 'layers' / 'helpers.py'
if not helpers.resolve().is_relative_to(environment):
    raise RuntimeError(f'Refusing to patch timm outside this environment: {helpers}')
original = 'from torch._six import container_abcs'
replacement = 'import collections.abc as container_abcs'
source = helpers.read_text(encoding='utf-8')
if source.count(original) == 1 and replacement not in source:
    helpers.write_text(source.replace(original, replacement), encoding='utf-8')
elif source.count(original) == 0 and source.count(replacement) == 1:
    pass
else:
    raise RuntimeError(f'Unexpected timm helpers.py contents in {helpers}')
print(f'Checked the timm compatibility import in {helpers}')
