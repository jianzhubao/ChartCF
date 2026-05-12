from .swift_rejected_response import patch_swift_rejected_response_check
from .swift_template import patch_swift_template_assert

__all__ = [
    'apply_all_patches',
    'patch_swift_rejected_response_check',
    'patch_swift_template_assert',
]


def apply_all_patches():
    patches_applied = []

    if patch_swift_rejected_response_check():
        patches_applied.append("RowPreprocessor._check_rejected_response")

    if patch_swift_template_assert():
        patches_applied.append("TemplateInputs._compat_rejected_response")

    return patches_applied
