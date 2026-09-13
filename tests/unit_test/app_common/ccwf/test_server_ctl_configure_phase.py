# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configure phase of ServerSideController, driven through the real _configure_clients().

The earlier tests for this phase re-implemented the controller's condition inside the
test and asserted on that copy, so they stayed green while the controller itself sent
the start task to a client that had never configured. These tests call the controller.

Scenario that motivated them (MediSwarm weekly run, 13 Sep 2026): two simulated clients,
min_clients = 1, no configure_min_clients. The server left the configure phase as soon
as client_B answered and started the workflow on client_A, whose persistor did not
exist yet -> "invalid model learnable: expect Model type but got NoneType".
"""

import time
from unittest.mock import MagicMock

from nvflare.app_common.ccwf.server_ctl import ClientStatus, ServerSideController

CLIENTS = ["site-1", "site-2", "site-3", "site-4"]


def _stub(clients=CLIENTS, starting_client="site-1", min_clients=0, configure_min_clients=0, timeout=30):
    ctrl = ServerSideController.__new__(ServerSideController)
    ctrl.participating_clients = list(clients)
    ctrl.starting_client = starting_client
    ctrl.min_clients = min_clients
    ctrl.configure_min_clients = configure_min_clients
    ctrl.configure_task_timeout = timeout
    ctrl.configure_task_name = "wf_config"
    ctrl.workflow_id = "wf-test"
    ctrl.client_statuses = {c: ClientStatus() for c in clients}
    ctrl._process_configure_reply = MagicMock()
    for name in ("log_info", "log_debug", "log_warning", "log_error", "system_panic"):
        setattr(ctrl, name, MagicMock())
    return ctrl


def _answering(ctrl, ready):
    """A send/broadcast stand-in: the named targets 'reply' by getting a ready_time."""

    def _fake(task=None, targets=None, **kwargs):
        for c in targets or []:
            if c in ready:
                ctrl.client_statuses[c].ready_time = time.time()

    return MagicMock(side_effect=_fake)


def _run(ctrl, ready, use_wait=True):
    ctrl.send_and_wait = _answering(ctrl, ready)
    ctrl.broadcast_and_wait = _answering(ctrl, ready)
    ctrl.broadcast = _answering(ctrl, ready)
    return ctrl._configure_clients({"k": "v"}, MagicMock(), MagicMock())


class TestStartingClientFirst:
    def test_starting_client_is_configured_alone_before_the_others(self):
        ctrl = _stub()
        assert _run(ctrl, ready=set(CLIENTS)) is True
        # first call: the starting client, alone, blocking
        ctrl.send_and_wait.assert_called_once()
        assert ctrl.send_and_wait.call_args.kwargs["targets"] == ["site-1"]
        # then the others, without the starting client
        ctrl.broadcast_and_wait.assert_called_once()
        assert ctrl.broadcast_and_wait.call_args.kwargs["targets"] == ["site-2", "site-3", "site-4"]
        ctrl.system_panic.assert_not_called()

    def test_unconfigured_starting_client_panics_and_nobody_else_is_started(self):
        ctrl = _stub(min_clients=1)
        assert _run(ctrl, ready={"site-2", "site-3", "site-4"}) is False
        ctrl.system_panic.assert_called_once()
        reason = ctrl.system_panic.call_args.args[0]
        assert "starting client site-1" in reason and "did not configure" in reason
        # the others are never even asked: the run cannot start without site-1
        ctrl.broadcast_and_wait.assert_not_called()
        ctrl.broadcast.assert_not_called()

    def test_no_starting_client_broadcasts_to_everyone_with_the_full_quorum(self):
        ctrl = _stub(starting_client="", min_clients=3)
        assert _run(ctrl, ready=set(CLIENTS)) is True
        ctrl.send_and_wait.assert_not_called()
        kw = ctrl.broadcast_and_wait.call_args.kwargs
        assert kw["targets"] == CLIENTS and kw["min_responses"] == 3


class TestQuorumAppliedToTheRest:
    def test_quorum_counts_the_starting_client(self):
        # configure_min_clients=3 with the starting client done -> wait for 2 more
        ctrl = _stub(configure_min_clients=3)
        assert _run(ctrl, ready=set(CLIENTS)) is True
        assert ctrl.broadcast_and_wait.call_args.kwargs["min_responses"] == 2

    def test_quorum_met_by_starting_client_alone_does_not_block_on_the_others(self):
        # min_clients=1: the starting client is the quorum; the rest configure in the background
        ctrl = _stub(min_clients=1)
        assert _run(ctrl, ready={"site-1"}) is True
        ctrl.broadcast_and_wait.assert_not_called()
        ctrl.broadcast.assert_called_once()
        assert ctrl.broadcast.call_args.kwargs["targets"] == ["site-2", "site-3", "site-4"]
        ctrl.system_panic.assert_not_called()
        # and the honest warning names who has not answered yet
        warned = " ".join(str(c.args[1]) for c in ctrl.log_warning.call_args_list)
        assert "site-2" in warned and "may rejoin" in warned

    def test_all_required_means_every_client(self):
        ctrl = _stub()  # min_clients=0, configure_min_clients=0 -> all four
        assert _run(ctrl, ready={"site-1", "site-2", "site-3"}) is False
        reason = ctrl.system_panic.call_args.args[0]
        assert "failed to configure clients ['site-4']" in reason and "need 4" in reason

    def test_too_few_of_the_rest_panics(self):
        ctrl = _stub(configure_min_clients=3)
        assert _run(ctrl, ready={"site-1", "site-2"}) is False
        reason = ctrl.system_panic.call_args.args[0]
        assert "only 2/4 configured" in reason and "need 3" in reason

    def test_enough_of_the_rest_proceeds_and_warns_about_the_missing(self):
        ctrl = _stub(configure_min_clients=3)
        assert _run(ctrl, ready={"site-1", "site-2", "site-3"}) is True
        ctrl.system_panic.assert_not_called()
        warned = " ".join(str(c.args[1]) for c in ctrl.log_warning.call_args_list)
        assert "['site-4']" in warned and "configure_min_clients=3" in warned
