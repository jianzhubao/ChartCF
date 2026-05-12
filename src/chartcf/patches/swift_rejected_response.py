"""
Patch: Swift RowPreprocessor._check_rejected_response

Disable RowPreprocessor._check_rejected_response check

Original code location:
.venv/lib/python3.12/site-packages/swift/llm/dataset/preprocessor/core.py:106-108

Original check will raise error when:
- rejected_response is None
- rejected_response is the same as messages[-1]['content']
"""

from typing import Any, Dict


def patch_swift_rejected_response_check():
    try:
        from swift.llm.dataset.preprocessor.core import RowPreprocessor

        @staticmethod
        def _check_rejected_response_disabled(row: Dict[str, Any]) -> None:
            _ = row
            pass

        RowPreprocessor._check_rejected_response = _check_rejected_response_disabled
        return True

    except (ImportError, AttributeError):
        return False
