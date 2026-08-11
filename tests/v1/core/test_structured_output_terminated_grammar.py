# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for terminated structured-output grammars (#42853)."""

from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test
EOS_TOKEN_ID = 50256


def _make_running_request(request_id: str = "0") -> Request:
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=4)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id=request_id,
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    return request


def _make_scheduler(request: Request) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True
    scheduler.structured_output_manager.trim_reasoning_for_advance.side_effect = (
        lambda request, new_token_ids: new_token_ids
    )
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.enable_return_routed_experts = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req: Request, delay_free_blocks: bool = False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)

    scheduler._free_request = Mock(side_effect=free_request)
    return scheduler


def test_scheduler_does_not_advance_terminated_grammar():
    request = _make_running_request()
    grammar = Mock()
    grammar.is_terminated.return_value = True
    grammar.accept_tokens.return_value = False
    request.structured_output_request = Mock(grammar=grammar)
    scheduler = _make_scheduler(request)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    grammar.accept_tokens.assert_not_called()
    assert request.status != RequestStatus.FINISHED_ERROR


@pytest.mark.parametrize("in_output", [False, True])
def test_draft_validation_skips_terminated_grammar(in_output: bool):
    request = _make_running_request()
    request.is_prefill_chunk = False
    grammar = Mock()
    grammar.is_terminated.return_value = True
    request.structured_output_request = Mock(grammar=grammar)
    scheduler = _make_scheduler(request)
    draft = DraftTokenIds(
        req_ids=[request.request_id],
        draft_token_ids=[[10, 20]],
    )

    if in_output:
        scheduler_output = Mock(
            scheduled_spec_decode_tokens={request.request_id: [-1, -1]}
        )
        scheduler.update_draft_token_ids_in_output(draft, scheduler_output)
        assert scheduler_output.scheduled_spec_decode_tokens[request.request_id] == [
            10,
            20,
        ]
    else:
        scheduler.update_draft_token_ids(draft)
        assert request.spec_token_ids == [10, 20]

    grammar.validate_tokens.assert_not_called()
