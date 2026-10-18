import os
import sys

# Add naval_isr/src and naval_isr root to sys.path
naval_isr_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(naval_isr_root, "src")

for p in [src_dir, naval_isr_root]:
    if p not in sys.path:
        sys.path.insert(0, p)
