"""
Tests for comparison/loaders.py. Every scenario here was manually
verified against real pandas behavior before writing the module -- see
loaders.py's module docstring for why encoding failures and null-string
preservation are deliberate, not bugs to fix.
"""

import pandas as pd
import pytest

from wherefore.comparison.loaders import load_csv, load_excel, load_file, load_json, load_parquet


@pytest.fixture
def csv_with_mixed_nulls(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("id,name,note\n1,Alice,\n2,Bob,NULL\n3,Carol,NaN\n4,Dave,N/A\n")
    return p


def test_only_truly_empty_cell_becomes_null(csv_with_mixed_nulls):
    """
    Regression-style test for the core loaders.py design decision:
    pandas' default read_csv collapses "NULL", "NaN", "N/A", and an
    empty cell to the same null value -- verified directly before
    writing this module. load_csv must NOT do that.
    """
    df = load_csv(csv_with_mixed_nulls)
    assert df["note"].isnull().sum() == 1  # only Alice's row
    assert df.loc[1, "note"] == "NULL"  # literal string, not null
    assert df.loc[2, "note"] == "NaN"  # literal string, not null
    assert df.loc[3, "note"] == "N/A"  # literal string, not null


def test_strict_utf8_raises_on_latin1_file(tmp_path):
    """
    Confirms load_csv does NOT silently fall back to another encoding
    -- a decode failure must surface as an error, since the failure
    itself is the signal encoding_mismatch needs.
    """
    p = tmp_path / "latin1.csv"
    p.write_bytes("id,name\n1,José\n".encode("latin-1"))

    with pytest.raises(UnicodeDecodeError):
        load_csv(p)


def test_explicit_encoding_override_works(tmp_path):
    p = tmp_path / "latin1.csv"
    p.write_bytes("id,name\n1,José\n".encode("latin-1"))

    df = load_csv(p, encoding="latin-1")
    assert df.loc[0, "name"] == "José"


def test_load_csv_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "does_not_exist.csv")


def test_load_json_distinguishes_real_null_from_literal_string(tmp_path):
    p = tmp_path / "test.json"
    p.write_text('[{"id": 1, "note": null}, {"id": 2, "note": "NULL"}]')

    df = load_json(p)
    assert df["note"].isnull().sum() == 1
    assert df.loc[1, "note"] == "NULL"


def test_load_file_dispatches_csv(csv_with_mixed_nulls):
    df = load_file(csv_with_mixed_nulls)
    assert len(df) == 4


def test_load_file_dispatches_json(tmp_path):
    p = tmp_path / "test.json"
    p.write_text('[{"id": 1}]')
    df = load_file(p)
    assert len(df) == 1


def test_load_file_raises_on_unrecognized_extension(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("not a real format")
    with pytest.raises(ValueError, match="Unrecognized file extension"):
        load_file(p)


def test_datetime_like_csv_columns_are_parsed_as_real_datetimes(tmp_path):
    """
    Regression test for a real bug caught while building the CLI:
    without this, a datetime column round-tripped through CSV becomes
    plain dtype 'str', which silently breaks every downstream
    dtype-based pattern match (clustering's patterns_by_dtype finds no
    candidates for a 'str' column, even if the underlying values are
    genuinely timestamps that differ by a constant offset).
    """
    p = tmp_path / "test.csv"
    p.write_text("id,created_at\n1,2024-01-15 10:30:00\n2,2024-01-16 11:45:00\n")

    df = load_csv(p)
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])


def test_bare_year_column_is_not_falsely_detected_as_datetime(tmp_path):
    """
    Regression guard: confirmed by direct testing that
    pd.to_datetime(..., format='ISO8601') happily parses bare numeric
    strings like "2024" as January 1st of that year -- which would
    silently corrupt a genuine fiscal_year/birth_year column into
    fabricated timestamps. load_csv must not do this.
    """
    p = tmp_path / "test.csv"
    p.write_text("id,fiscal_year\n1,2024\n2,2025\n3,2026\n")

    df = load_csv(p)
    assert not pd.api.types.is_datetime64_any_dtype(df["fiscal_year"])


def test_partially_unparseable_column_is_left_as_string(tmp_path):
    """
    A column that's mostly dates but has one genuinely non-date value
    must NOT be converted -- converting it would require errors='coerce',
    which silently turns the bad value into NaT rather than preserving
    it as evidence of a real data problem.
    """
    p = tmp_path / "test.csv"
    p.write_text("id,maybe_date\n1,2024-01-15 10:30:00\n2,not-a-date\n")

    df = load_csv(p)
    assert df["maybe_date"].dtype.name in ("object", "str")


def test_mostly_dates_with_a_null_sentinel_parses_as_hybrid_column():
    """
    Regression test for a real bug caught while building
    null_type_coercion: the ORIGINAL all-or-nothing version of this
    function required every value to parse, so a single "NULL" sentinel
    among 49 real dates blocked the WHOLE column from being recognized
    as datetime -- this silently broke null_type_coercion detection on
    any real CSV file, since the corruption this pattern exists to
    detect is EXACTLY "mostly real dates, a few literal NULL strings."

    The fix parses what's parseable as real datetimes and preserves
    the original sentinel text exactly where parsing fails, gated by a
    failure-rate threshold.
    """
    rows = [f"{i},2024-01-{(i % 28) + 1:02d} 10:00:00" for i in range(1, 49)]
    rows += ["49,NULL", "50,NULL"]
    csv_text = "id,ts\n" + "\n".join(rows) + "\n"

    import io
    import pandas as pd

    df = pd.read_csv(io.StringIO(csv_text), keep_default_na=False, na_values=[""])
    from wherefore.comparison.loaders import _try_parse_datetime_columns

    result = _try_parse_datetime_columns(df)

    # Real dates should be genuine Timestamps.
    assert isinstance(result.loc[0, "ts"], pd.Timestamp)
    # The sentinel strings must be preserved EXACTLY, not coerced to NaT.
    assert result.loc[48, "ts"] == "NULL"
    assert result.loc[49, "ts"] == "NULL"
    assert result["ts"].dtype == object


def test_mostly_garbage_column_is_not_converted(tmp_path):
    """
    The failure-rate threshold's other side: a column that's mostly
    NOT dates (e.g. a genuinely mixed free-text field where a couple
    of values happen to look like dates) should be left alone, not
    wrongly treated as "a date column with some sentinel values."
    """
    rows = ["1,random text here", "2,more random text", "3,2024-01-15 10:00:00", "4,yet more text"]
    csv_text = "id,val\n" + "\n".join(rows) + "\n"
    p = tmp_path / "test.csv"
    p.write_text(csv_text)

    df = load_csv(p)
    # Failure rate here is 3/4 = 75%, well above the 20% threshold --
    # column should be left as plain strings.
    assert df["val"].dtype.name in ("object", "str")


def test_load_parquet_preserves_native_datetime_dtype(tmp_path):
    """
    Confirmed by direct testing: Parquet round-trips a real datetime
    column with NO parsing step needed -- unlike CSV, which requires
    the hybrid datetime-detection logic above. The resolution may
    differ from the in-memory original (Parquet defaults to
    milliseconds; pandas in-memory uses seconds here), but the VALUES
    match exactly, and dtype-family matching elsewhere in the project
    is already designed to handle resolution variance.
    """
    df = pd.DataFrame({"id": [1, 2], "ts": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    p = tmp_path / "test.parquet"
    df.to_parquet(p, index=False)

    loaded = load_parquet(p)
    assert pd.api.types.is_datetime64_any_dtype(loaded["ts"])
    assert (loaded["ts"].values == df["ts"].values).all()


def test_load_parquet_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_parquet(tmp_path / "missing.parquet")


def test_load_parquet_cannot_represent_mixed_type_column(tmp_path):
    """
    Documents a real, confirmed limitation: Parquet's columnar typing
    means a column genuinely cannot hold a mix of types (e.g. a real
    Timestamp next to the literal string "NULL") the way an in-memory
    pandas object-dtype column can. This is not a loaders.py bug --
    it's a property of the file format itself, confirmed by attempting
    the write (not the read) and observing pyarrow's own error.
    """
    df = pd.DataFrame({"id": [1, 2], "val": [pd.Timestamp("2024-01-01"), "NULL"]})
    p = tmp_path / "mixed.parquet"
    with pytest.raises(Exception):  # pyarrow.lib.ArrowTypeError specifically
        df.to_parquet(p, index=False)


def test_load_excel_only_truly_empty_cell_becomes_null(tmp_path):
    """
    Regression-style test mirroring the CSV null-preservation test:
    confirmed by direct testing that pandas' default read_excel has
    the SAME null-collapsing behavior as read_csv (a literal "NULL"
    string and a genuinely empty cell both become NaN by default).
    """
    df = pd.DataFrame({"id": [1, 2, 3, 4], "note": ["hello", None, "NULL", ""]})
    p = tmp_path / "test.xlsx"
    df.to_excel(p, index=False)

    loaded = load_excel(p)
    assert loaded.loc[0, "note"] == "hello"
    assert pd.isna(loaded.loc[1, "note"])  # genuinely None in the source
    assert loaded.loc[2, "note"] == "NULL"  # literal string, preserved
    assert pd.isna(loaded.loc[3, "note"])  # genuinely empty string


def test_load_excel_preserves_native_datetime_dtype(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "ts": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    p = tmp_path / "test.xlsx"
    df.to_excel(p, index=False)

    loaded = load_excel(p)
    assert pd.api.types.is_datetime64_any_dtype(loaded["ts"])


def test_load_excel_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_excel(tmp_path / "missing.xlsx")


def test_load_excel_respects_sheet_name(tmp_path):
    p = tmp_path / "multi_sheet.xlsx"
    with pd.ExcelWriter(p) as writer:
        pd.DataFrame({"id": [1]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"id": [2]}).to_excel(writer, sheet_name="Second", index=False)

    first = load_excel(p, sheet_name="First")
    second = load_excel(p, sheet_name="Second")
    assert first.loc[0, "id"] == 1
    assert second.loc[0, "id"] == 2


def test_load_file_dispatches_parquet(tmp_path):
    df = pd.DataFrame({"id": [1, 2]})
    p = tmp_path / "test.parquet"
    df.to_parquet(p, index=False)
    loaded = load_file(p)
    assert len(loaded) == 2


def test_load_file_dispatches_xlsx(tmp_path):
    df = pd.DataFrame({"id": [1, 2]})
    p = tmp_path / "test.xlsx"
    df.to_excel(p, index=False)
    loaded = load_file(p)
    assert len(loaded) == 2


def test_load_file_error_message_mentions_all_supported_formats(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("not a real format")
    with pytest.raises(ValueError, match=r"\.csv.*\.json.*\.parquet.*\.xlsx"):
        load_file(p)
