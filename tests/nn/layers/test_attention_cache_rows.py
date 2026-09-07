"""Tests for AR per-row cache helpers."""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from vrl.nn.layers.attention.cache_rows import (
    ARCacheRows,
    _cache_kv_pairs,
    ar_concat_rows,
    ar_split_rows,
)


def test_ar_split_and_concat_rows_preserve_nested_kv_order() -> None:
    """Splitting a batched cache into rows and concatenating rows back preserves the nested
    past_key_values structure and lets rows be reordered.
    """
    key = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    value = key + 100
    cache = {
        "past_key_values": ((key, value),),
        "last_hidden": torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5),
    }

    rows = ar_split_rows(cache, 3)
    assert len(rows) == 3
    assert torch.equal(rows[1]["past_key_values"][0][0], key[1:2])
    assert torch.equal(rows[2]["last_hidden"], cache["last_hidden"][2:3])

    merged = ar_concat_rows([rows[2], rows[0]])
    assert torch.equal(merged["past_key_values"][0][0], torch.cat([key[2:3], key[0:1]]))
    assert torch.equal(
        merged["last_hidden"],
        torch.cat([cache["last_hidden"][2:3], cache["last_hidden"][0:1]]),
    )


def test_ar_split_rows_rejects_wrong_batch_size() -> None:
    with pytest.raises(ValueError, match="cannot split tensor"):
        ar_split_rows(torch.zeros(2, 4), 3)


def test_ar_cache_rows_gather_and_scatter_nested_values() -> None:
    """``ARCacheRows`` gathers rows in the requested order, scatters replacements into nested
    values, and copies one row over another through ``select_rows`` / ``scatter_rows``.
    """
    key = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    value = key + 100
    cache = {
        "past_key_values": ((key, value),),
        "last_hidden": torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5),
    }
    rows = ARCacheRows.from_batched(cache, 3, owner="test.cache")

    gathered = rows.gather([2, 0])
    assert torch.equal(
        gathered["past_key_values"][0][0],
        torch.cat([key[2:3], key[:1]]),
    )

    replacement = {
        "past_key_values": ((torch.full((2, 2, 4), -1.0), torch.full((2, 2, 4), 7.0)),),
        "last_hidden": torch.full((2, 5), 3.0),
    }
    rows.scatter([0, 2], replacement)

    updated = rows.gather([0, 2])
    assert torch.equal(
        updated["past_key_values"][0][0],
        replacement["past_key_values"][0][0],
    )
    assert torch.equal(updated["last_hidden"], replacement["last_hidden"])
    assert torch.equal(rows.gather([1])["past_key_values"][0][0], key[1:2])

    rows.scatter_rows([1], rows.select_rows([0]))
    assert torch.equal(
        rows.gather([1])["last_hidden"],
        replacement["last_hidden"][:1],
    )


def test_ar_cache_rows_rejects_invalid_indices() -> None:
    rows = ARCacheRows.from_batched(torch.zeros(2, 4), 2, owner="test.cache")

    with pytest.raises(IndexError, match=r"test\.cache"):
        rows.gather([2])

    with pytest.raises(ValueError, match="received 1 rows"):
        rows.scatter_rows([0, 1], rows.select_rows([0]))


def test_ar_cache_helpers_preserve_transformers_dynamic_cache_objects() -> None:
    """Row split / concat keep ``DynamicCache`` objects as ``DynamicCache`` (transformers 5 has no
    legacy-tuple conversion) with the K/V content preserved.
    """
    key = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    value = key + 100
    cache = DynamicCache()
    cache.update(key, value, layer_idx=0)

    rows = ar_split_rows(cache, 2)
    assert len(rows) == 2
    assert all(isinstance(row, DynamicCache) for row in rows)
    # Read K/V back through the module's own 4/5-tolerant reader:
    # transformers 5 removed DynamicCache.to_legacy_cache.
    assert torch.equal(_cache_kv_pairs(rows[0])[0][0], key[:1])
    assert torch.equal(_cache_kv_pairs(rows[1])[0][1], value[1:2])

    merged = ar_concat_rows([rows[1], rows[0]])
    assert isinstance(merged, DynamicCache)
    merged_key, merged_value = _cache_kv_pairs(merged)[0]
    assert torch.equal(merged_key, torch.cat([key[1:2], key[:1]], dim=0))
    assert torch.equal(
        merged_value,
        torch.cat([value[1:2], value[:1]], dim=0),
    )
