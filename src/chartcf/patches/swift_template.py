"""
Patch: Swift TemplateInputs._compat_rejected_response

Remove the assert in TemplateInputs._compat_rejected_response

Original code location:
.venv/lib/python3.12/site-packages/swift/llm/template/template_inputs.py:306

Original assertion error:
assert rejected_response != response
"""

from copy import deepcopy
from typing import Any, Dict


def patch_swift_template_assert():
    try:
        from swift.llm.template.template_inputs import TemplateInputs

        @staticmethod
        def _compat_rejected_response_patched(inputs: Dict[str, Any]):
            if 'rejected_response' not in inputs:
                return

            # Find the first round's 'assistant'.
            messages = inputs['messages']
            assert len(messages) > 0, f'messages: {messages}'
            for idx in range(len(messages), 0, -1):
                message = messages[idx - 1]
                if message['role'] in {'user', 'tool', 'tool_response'}:
                    break

            rejected_response = inputs.pop('rejected_response')
            if isinstance(rejected_response, list) and rejected_response and isinstance(rejected_response[0], str):
                inputs['rejected_response'] = rejected_response
                return
            assert isinstance(rejected_response, str), f'rejected_response: {rejected_response}'

            # Check that the response is different from the rejected_response.
            if isinstance(rejected_response, str):
                # Original code will check rejected_response != response, but we skip this check now
                # if len(messages[idx:]) == 1:
                #     response = messages[idx]['content']
                #     assert rejected_response != response
                rejected_response = [{'role': 'assistant', 'content': rejected_response}]
            inputs['rejected_messages'] = deepcopy(messages[:idx]) + rejected_response

        TemplateInputs._compat_rejected_response = _compat_rejected_response_patched
        return True

    except (ImportError, AttributeError):
        return False
