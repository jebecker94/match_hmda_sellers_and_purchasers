# -*- coding: utf-8 -*-
"""HMDA delivery discovery and loading helpers.

Created on Friday Jul 19 10:20:24 2024
Updated On: Wednesday May 21 10:00:00 2025

The functions in this module expose a light-weight interface that allows the
matching scripts to discover the most recent Loan/Application Register (LAR)
deliveries stored in :mod:`config.DATA_DIR` and load them into pandas,
PyArrow, or Polars data structures.  The helpers are deliberately typed to make
downstream usage explicit and to surface path-handling mistakes early in the
development cycle.
"""

from __future__ import annotations

# Import Packages
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import logging

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config


logger = logging.getLogger(__name__)


EngineName = Literal["pandas", "pyarrow", "polars"]
FilterCondition = Tuple[str, str, object]
ParquetFilters = Union[
    Sequence[FilterCondition],
    Sequence[Sequence[FilterCondition]],
]
PathLike = Union[str, Path]

# Set Folder Paths
DATA_DIR = config.DATA_DIR

def get_hmda_files(
    data_folder: PathLike = DATA_DIR,
    file_type: str = "lar",
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    version_type: Optional[str] = None,
    extension: Optional[str] = None,
) -> List[Path]:
    """Return absolute paths to the latest HMDA deliveries.

    Parameters
    ----------
    data_folder:
        Base directory that contains the HMDA delivery folders along with the
        ``file_list_hmda.csv`` manifest.  The value may be a :class:`~pathlib.Path`
        instance or a string.
    file_type:
        HMDA file type to retrieve, typically ``"lar"`` for Loan/Application
        Register extracts.  Case-insensitive.
    min_year:
        Lower bound for the delivery year (inclusive).  If ``None``, the
        manifest is not filtered by a minimum year.
    max_year:
        Upper bound for the delivery year (inclusive).  If ``None``, the
        manifest is not filtered by a maximum year.
    version_type:
        Optional version discriminator applied to the manifest ``VersionType``
        column.  Pass ``None`` to keep all available versions.
    extension:
        Desired file extension (for example ``"parquet"`` or ``"csv.gz"``).  When
        provided, the manifest is filtered using the corresponding indicator
        column and returned paths include the extension.  A ``ValueError`` is
        raised if the extension is not recognised.

    Returns
    -------
    list[pathlib.Path]
        Absolute paths pointing to the most recent delivery for each year that
        satisfies the supplied filters.  The list is ordered chronologically by
        year.  If no rows satisfy the filters, an empty list is returned.
    """

    data_dir = Path(data_folder)
    list_file = data_dir / "file_list_hmda.csv"

    df = pd.read_csv(list_file)

    df = df[df["FileType"].str.lower() == file_type.lower()].copy()

    if min_year is not None:
        df = df[df["Year"] >= min_year]
    if max_year is not None:
        df = df[df["Year"] <= max_year]

    if extension is not None:
        extension_lower = extension.lower()
        extension_flag_map = {
            "parquet": "FileParquet",
            "csv.gz": "FileCSVGZ",
            "dta": "FileDTA",
        }
        try:
            extension_column = extension_flag_map[extension_lower]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(
                f"Unsupported extension '{extension}'. "
                "Expected one of: parquet, csv.gz, dta"
            ) from exc
        df = df[df[extension_column] == 1]

    if version_type is not None:
        df = df[df["VersionType"] == version_type]

    df = df.drop_duplicates(subset=["Year"], keep="first")
    df = df.sort_values(by=["Year"])

    files: List[Path] = []
    for folder_name, prefix in zip(df["FolderName"], df["FilePrefix"]):
        base_path = data_dir / folder_name
        if extension is None:
            files.append(base_path / prefix)
        else:
            files.append(base_path / f"{prefix}.{extension}")

    return files

# Load HMDA Files
def load_hmda_file(
    data_folder: PathLike = DATA_DIR,
    file_type: str = "lar",
    min_year: int = 2018,
    max_year: int = 2023,
    columns: Optional[Sequence[str]] = None,
    filters: Optional[ParquetFilters] = None,
    verbose: bool = False,
    engine: EngineName = "pandas",
    **kwargs: Any,
) -> Union[pd.DataFrame, pl.LazyFrame, pl.DataFrame, pa.Table]:
    """Load one or more HMDA parquet deliveries into a tabular object.

    Parameters
    ----------
    data_folder:
        Directory containing the HMDA deliveries and manifest.  See
        :func:`get_hmda_files` for expectations.
    file_type:
        HMDA file type to load (for example ``"lar"``).  The manifest is filtered
        case-insensitively.
    min_year, max_year:
        Inclusive bounds for the delivery years to load.  The default range
        mirrors the post-2018 HMDA era.
    columns:
        Optional list of column names to materialise.  When omitted, the full
        schema is read.
    filters:
        Optional predicate passed through to :func:`pandas.read_parquet` or
        :func:`pyarrow.parquet.read_table`.  The structure should follow the
        [PyArrow filter specification](https://arrow.apache.org/docs/python/parquet.html#filters).
    verbose:
        If ``True``, the function logs the path of each file before it is
        loaded.
    engine:
        Backend used to materialise the parquet data.  ``"pandas"`` returns a
        :class:`pandas.DataFrame`, ``"pyarrow"`` returns a :class:`pyarrow.Table`,
        and ``"polars"`` returns a concatenated :class:`polars.LazyFrame` or
        :class:`polars.DataFrame` depending on ``kwargs``.
    **kwargs:
        Additional keyword arguments forwarded to the selected engine's parquet
        reader.

    Returns
    -------
    pandas.DataFrame | polars.LazyFrame | polars.DataFrame | pyarrow.Table
        Object containing the concatenated deliveries.

    Raises
    ------
    FileNotFoundError
        If the manifest filters produce no matching files.
    ValueError
        If ``engine`` is not one of the supported options.
    """

    files = get_hmda_files(
        data_folder=data_folder,
        file_type=file_type,
        min_year=min_year,
        max_year=max_year,
        extension="parquet",
    )

    if not files:
        raise FileNotFoundError(
            "No HMDA files matched the supplied filters. Verify that the "
            "manifest is up to date and the year bounds are correct."
        )

    tables: List[Union[pd.DataFrame, pa.Table, pl.DataFrame, pl.LazyFrame]] = []

    if engine == "pandas":
        for file_path in files:
            if verbose:
                logger.info("Adding data from file: %s", file_path)
            table = pd.read_parquet(
                file_path,
                columns=list(columns) if columns is not None else None,
                filters=filters,
                **kwargs,
            )
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    if engine == "pyarrow":
        for file_path in files:
            if verbose:
                logger.info("Adding data from file: %s", file_path)
            table = pq.read_table(
                file_path,
                columns=list(columns) if columns is not None else None,
                filters=filters,
                **kwargs,
            )
            tables.append(table)
        return pa.concat_tables(tables)

    if engine == "polars":
        for file_path in files:
            if verbose:
                logger.info("Adding data from file: %s", file_path)
            table = pl.scan_parquet(str(file_path), **kwargs)
            tables.append(table)
        return pl.concat(tables)

    raise ValueError(
        f"Unsupported engine '{engine}'. Expected one of: pandas, pyarrow, polars"
    )
