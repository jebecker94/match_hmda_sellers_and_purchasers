"""Utilities for summarizing top seller/purchaser counterparty relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass
class CounterpartySummaryConfig:
    """Configuration for top counterparty summaries."""

    seller_id_col: str = "lei_s"
    purchaser_id_col: str = "lei_p"
    seller_name_col: Optional[str] = "respondent_name_s"
    purchaser_name_col: Optional[str] = "respondent_name_p"
    top_n: int = 5
    include_names: bool = False


def _validate_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    optional_columns: Iterable[Optional[str]] = (),
) -> None:
    """Ensure that all required and requested optional columns are present."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(
            "The provided matches table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    requested_optional = [column for column in optional_columns if column]
    missing_optional = [
        column for column in requested_optional if column not in df.columns
    ]
    if missing_optional:
        raise KeyError(
            "Requested optional columns are not available: "
            + ", ".join(sorted(missing_optional))
        )


def _prepare_mapping(
    matches: pd.DataFrame,
    id_column: str,
    name_column: Optional[str],
) -> pd.DataFrame:
    """Extract a unique mapping from identifier to name."""

    columns = [id_column]
    if name_column:
        columns.append(name_column)
    return matches[columns].drop_duplicates()


def summarize_top_counterparties(
    matches: pd.DataFrame,
    config: CounterpartySummaryConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute top counterparties for both originators and purchasers.

    Parameters
    ----------
    matches : pandas.DataFrame
        The matched HMDA loan table produced by the matching scripts. At a
        minimum the frame must include the seller and purchaser identifier
        columns specified in ``config`` (defaults to ``"lei_s"`` and
        ``"lei_p"``).
    config : CounterpartySummaryConfig, optional
        Configuration describing the identifier columns, optional name
        columns, the number of counterparties to return, and whether to merge
        name fields into the output.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Two tables. The first summarizes the top purchaser counterparties for
        each seller (originator). The second summarizes the top seller
        counterparties for each purchaser.
    """

    cfg = config or CounterpartySummaryConfig()

    if cfg.top_n < 1:
        raise ValueError("`top_n` must be at least 1.")

    _validate_columns(
        matches,
        required_columns=[cfg.seller_id_col, cfg.purchaser_id_col],
        optional_columns=(
            cfg.seller_name_col if cfg.include_names else None,
            cfg.purchaser_name_col if cfg.include_names else None,
        ),
    )

    # Aggregate counts for seller -> purchaser relationships.
    seller_pair_counts = (
        matches.groupby([cfg.seller_id_col, cfg.purchaser_id_col])
        .size()
        .reset_index(name="match_count")
    )

    # Rank counterparties for each seller and keep the requested top N.
    seller_pair_counts["counterparty_rank"] = (
        seller_pair_counts.groupby(cfg.seller_id_col)["match_count"].rank(
            method="first", ascending=False
        )
    )
    top_seller_counterparties = seller_pair_counts[
        seller_pair_counts["counterparty_rank"] <= cfg.top_n
    ].copy()
    top_seller_counterparties.sort_values(
        by=[cfg.seller_id_col, "counterparty_rank"], inplace=True
    )

    # Aggregate counts for purchaser -> seller relationships.
    purchaser_pair_counts = (
        matches.groupby([cfg.purchaser_id_col, cfg.seller_id_col])
        .size()
        .reset_index(name="match_count")
    )
    purchaser_pair_counts["counterparty_rank"] = (
        purchaser_pair_counts.groupby(cfg.purchaser_id_col)["match_count"].rank(
            method="first", ascending=False
        )
    )
    top_purchaser_counterparties = purchaser_pair_counts[
        purchaser_pair_counts["counterparty_rank"] <= cfg.top_n
    ].copy()
    top_purchaser_counterparties.sort_values(
        by=[cfg.purchaser_id_col, "counterparty_rank"], inplace=True
    )

    if cfg.include_names:
        if not cfg.seller_name_col or not cfg.purchaser_name_col:
            raise ValueError(
                "`include_names=True` requires both seller_name_col and "
                "purchaser_name_col to be provided."
            )

        # Prepare unique mappings for merging names.
        seller_mapping = _prepare_mapping(
            matches, cfg.seller_id_col, cfg.seller_name_col
        ).rename(columns={cfg.seller_name_col: "seller_name"})
        purchaser_mapping = _prepare_mapping(
            matches, cfg.purchaser_id_col, cfg.purchaser_name_col
        ).rename(columns={cfg.purchaser_name_col: "purchaser_name"})

        top_seller_counterparties = top_seller_counterparties.merge(
            seller_mapping,
            on=cfg.seller_id_col,
            how="left",
        ).merge(
            purchaser_mapping,
            on=cfg.purchaser_id_col,
            how="left",
        )

        top_purchaser_counterparties = top_purchaser_counterparties.merge(
            purchaser_mapping,
            on=cfg.purchaser_id_col,
            how="left",
        ).merge(
            seller_mapping,
            on=cfg.seller_id_col,
            how="left",
        )

    return top_seller_counterparties, top_purchaser_counterparties


__all__ = [
    "CounterpartySummaryConfig",
    "summarize_top_counterparties",
]

