from __future__ import annotations

import numpy as np
import pytest

from scripts.run_log_s_mechanism_ablation_811 import CASES, assemble_case_matrices


def test_assemble_case_matrices_has_eight_ordered_variants() -> None:
    values = {
        "sar": np.full((3, 2), 1.0, dtype=np.float32),
        "ems": np.full((3, 2), 2.0, dtype=np.float32),
        "raw_i": np.full((3, 2), 3.0, dtype=np.float32),
        "attention_target_ems": np.full((3, 2), 4.0, dtype=np.float32),
        "geometry_ot": np.full((3, 2), 5.0, dtype=np.float32),
        "union_js": np.full((3, 2), 6.0, dtype=np.float32),
        "ffn_input_ems": np.full((3, 2), 7.0, dtype=np.float32),
    }
    result = assemble_case_matrices(**values)
    assert tuple(result) == CASES
    assert result["sar"].shape == (3, 2)
    assert result["ems"].shape == (3, 2)
    for case in CASES[2:]:
        assert result[case].shape == (3, 4)
    np.testing.assert_array_equal(result["i_ems"][:, :2], values["raw_i"])
    np.testing.assert_array_equal(result["sar_union_js"][:, 2:], values["union_js"])


def test_assemble_case_matrices_rejects_row_mismatch() -> None:
    shared = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="row counts differ"):
        assemble_case_matrices(
            sar=shared,
            ems=shared,
            raw_i=np.ones((2, 2), dtype=np.float32),
            attention_target_ems=shared,
            geometry_ot=shared,
            union_js=shared,
            ffn_input_ems=shared,
        )
