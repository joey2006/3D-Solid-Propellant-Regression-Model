"""Allow ``python -m desktop``."""

import sys

from .app import main

sys.exit(main())
