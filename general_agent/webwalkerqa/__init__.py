"""
WebWalkerQA T³ PoC
==================
Compares sequential scaling (s1) vs parallel threading (T³) on WebWalkerQA.

Methods implemented:
  - s1       : single thread, growing token budget per turn
  - t3_fixed : k parallel threads with hand-crafted diversity seeds, parent synthesis

Extendable to:
  - t3_dynamic : parent LM decides k and seeds
  - t3_dpp     : DPP-based seed selection

Entry points:
  python -m webwalkerqa.experiment --config A1  (single run)
  python -m webwalkerqa.experiment --all        (full matrix)
"""
