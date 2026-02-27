from __future__ import annotations

"""Porting/cleaning façade.

Era-specific logic lives under tally_assure/eras/.
This module keeps backwards-compatible names so the rest of the codebase doesn't need to change
when we modularise.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# --- 2002 era (splitvote XLS + e9_part8_*.csv) ---
from .eras.era_2002.splitvote_2002 import (
    _strip_trailing_dot_zero,
    _strip_trailing_zeros,
    _format_df_numbers_for_csv,
    read_split_votes_csv,
    read_xls_sheet0,
    _coerce_num_series,
    process_2002_split_sheet_df_to_endstate,
    process_2002_split_xls_to_endstate,
)

def process_split_votes_csv_to_endstate(*args, **kwargs):
    raise NotImplementedError("2005-2023 processing lives in the Sainte-Laguë + sliding puzzle pipeline.")


def port_xls_all_sheets(*args, **kwargs):
    raise NotImplementedError("Raw sheet CSVs are no longer kept; we port directly to endstate in-memory.")
