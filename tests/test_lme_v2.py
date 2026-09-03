"""The LongMemEval-V2 adapter: trajectory conversion, the four arms behind the
benchmark's `Memory` surface, persistence, and the upstream registration.

Tests that need the upstream checkout (`~/.agmem/upstream/longmemeval-v2`) skip
without it; everything else runs against the mirrored base class."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from helpers import openai_stub

from agmem.bench.lme_v2 import (
    HOST,
    MEMORY_TYPE,
    AgmemMemory,
    build_memory_config,
    harness_argv,
    main,
    register_with_upstream,
    trajectory_to_session,
)

UPSTREAM = Path(os.environ.get("LME_V2_UPSTREAM", "~/.agmem/upstream/longmemeval-v2")).expanduser()
needs_upstream = pytest.mark.skipif(
    not (UPSTREAM / "memory_modules" / "memory.py").is_file(),
    reason="longmemeval-v2 checkout not present",
)


def _public_trajectory(tid: str = "traj-web-1", n_states: int = 3) -> dict:
    return {
        "id": tid,
        "domain": "web",
        "goal": "Add the blue widget to the cart and check out",
        "start_url": "http://shop.local/",
        "outcome": "success",
        "states": [
            {
                "url": f"http://shop.local/page/{i}",
                "action": f"click [{100 + i}]" if i < n_states - 1 else None,
                "thought": f"step {i}: looking for the widget",
                "accessibility_tree": f"[1] RootWebArea 'Shop page {i}'\n  [{100 + i}] link 'Blue widget'",
                "screenshot": f"screenshots/{tid}/{i:04d}.png",
                "step": i,
            }
            for i in range(n_states)
        ],
    }


def _internal_trajectory(tid: str = "traj-ent-1") -> dict:
    return {
        "id": tid,
        "metadata": {"original_goal": ["Reset the password for user 42", "and confirm by email"]},
        "outcome": "failure",
        "content": [
            {
                "url": "http://erp.local/users/42",
                "action": "type [7] 'newpass'",
                "thoughts": "the reset form is here",
                "step": 0,
                "observation": {
                    "text": "[7] textbox 'New password'",
                    "screenshot": "screenshots/traj-ent-1/0000.png",
                },
            }
        ],
    }


def test_a_public_trajectory_becomes_a_session_of_observations_and_actions():
    session = trajectory_to_session(_public_trajectory())
    assert session.host == HOST and session.id == "traj-web-1"
    assert session.task_text == "Add the blue widget to the cart and check out"
    kinds = [s.kind for s in session.steps]
    # goal, then (observe, act) per state; the last state has no action but a thought
    assert kinds == ["user", *["tool_result", "assistant"] * 3]
    observe = session.steps[1]
    assert observe.tool_name == "observe"
    assert observe.text.startswith("URL: http://shop.local/page/0\n[1] RootWebArea")
    assert observe.meta["screenshot"] == "screenshots/traj-web-1/0000.png"
    assert observe.meta["state_index"] == 0 and observe.meta["step"] == 0
    act = session.steps[2]
    assert act.text == "step 0: looking for the widget\nAction: click [100]"
    assert session.meta == {
        "domain": "web",
        "goal": session.task_text,
        "outcome": "success",
        "n_states": 3,
    }
    assert session.cwd == "http://shop.local/"
    # ids are deterministic in host, id and position, as for any session
    assert session.episode_id(1) == trajectory_to_session(_public_trajectory()).episode_id(1)


def test_the_internal_shape_and_the_state_clip_are_handled():
    session = trajectory_to_session(_internal_trajectory())
    assert session.task_text == "Reset the password for user 42 / and confirm by email"
    assert session.steps[1].text == "URL: http://erp.local/users/42\n[7] textbox 'New password'"
    assert session.steps[1].meta["screenshot"] == "screenshots/traj-ent-1/0000.png"
    assert session.steps[2].text == "the reset form is here\nAction: type [7] 'newpass'"

    big = _public_trajectory(n_states=1)
    big["states"][0]["accessibility_tree"] = "x" * 100
    clipped = trajectory_to_session(big, max_state_chars=40).steps[1].text
    assert clipped.endswith("x" * 40 + "\n…[60 chars of this state omitted]…")

    with pytest.raises(ValueError):
        trajectory_to_session({"states": []})


def _config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "agmem.toml"
    path.write_text('[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n' + extra)
    return path


def test_raw_vector_arm_answers_from_the_stored_states(tmp_path):
    memory = AgmemMemory(
        {
            "write": "raw",
            "read": "vector",
            "config": str(_config(tmp_path)),
            "data_dir": str(tmp_path / "store"),
        }
    )
    memory.insert(_public_trajectory("traj-web-1"))
    memory.insert(_internal_trajectory("traj-ent-1"))
    memory.set_query_context(query_invocation_id="inv-1")
    items = memory.query("where is the new password textbox?")
    assert len(items) == 1 and items[0]["type"] == "text"
    text = items[0]["value"]
    assert "[7] textbox 'New password'" in text
    # a hit is rendered as its trajectory window: goal, state, URL, agent turn, AXTree
    assert "### Trajectory traj-ent-1 (states 0-0 of 1)" in text
    assert "Goal: Reset the password for user 42 / and confirm by email" in text
    assert (
        "State 0 (step 0)\n- URL: http://erp.local/users/42\n- Agent: the reset form is here\nAction: type [7] 'newpass'\n- AXTree:\n[7] textbox 'New password'"
        in text
    )
    hook = memory.post_query_hook(query="q", query_image=None, memory_context=items)
    assert hook["read"] == "vector" and hook["items"] > 0 and hook["latency_s"] >= 0
    assert hook["windows"] >= 1
    memory.clear_query_context()
    assert memory.get_query_context() == {}
    # a second insert of the same trajectory is the facade's idempotent path
    memory.insert(_public_trajectory("traj-web-1"))
    assert memory._inserted == ["traj-web-1", "traj-ent-1", "traj-web-1"]
    memory.close()


def test_memory_params_are_validated_and_the_persisted_config_drops_paths(tmp_path):
    with pytest.raises(ValueError):
        AgmemMemory({"write": "notes"})
    with pytest.raises(ValueError):
        AgmemMemory({"read": "graph"})
    memory = AgmemMemory(
        {"write": "raw", "read": "vector", "data_dir": str(tmp_path / "s"), "top_k": 3}
    )
    assert memory.memory_config == {
        "memory_type": MEMORY_TYPE,
        "memory_params": {"write": "raw", "read": "vector", "top_k": 3},
    }
    # no data_dir given: a fresh temp dir, never a shared default
    a, b = AgmemMemory({}), AgmemMemory({})
    assert a.data_dir != b.data_dir and a.data_dir.is_dir()


def _llm_config(tmp_path: Path, url: str) -> Path:
    roles = "".join(
        f'\n[llm.{role}]\nendpoint = "{url}"\nmodel = "stub"\napi_key = "stub"\n'
        for role in ("distill", "explore")
    )
    return _config(tmp_path, roles)


def test_experience_explorer_arm_distils_on_insert_and_explores_on_query(tmp_path):
    runbook = json.dumps(
        {
            "summary": "The agent added the widget and checked out.",
            "tasks": [
                {
                    "name": "Add the blue widget to the cart",
                    "outcome": "success",
                    "procedure": ["open the shop", "click the Blue widget link", "check out"],
                    "keywords": ["blue widget", "cart"],
                    "steps": [1, 6],
                }
            ],
        }
    )
    with openai_stub([runbook]) as (url, requests):
        memory = AgmemMemory(
            {
                "write": "experience",
                "read": "explorer",
                "config": str(_llm_config(tmp_path, url)),
                "data_dir": str(tmp_path / "store"),
                "max_steps": 2,
            }
        )
        memory.insert(_public_trajectory("traj-web-1"))
        assert len(requests) == 1  # one distillation call per trajectory
        runbooks = memory.mem.doc_store.list_items("runbooks", namespace="main")
        assert len(runbooks) == 1 and runbooks[0]["step_range"] == [1, 6]

    # The explorer: the stub lists, then answers citing the exported session file.
    session_file = f"sessions/{HOST}/traj-web-1.md"
    with openai_stub(
        [
            json.dumps({"action": "list", "reason": "layout", "path": "sessions"}),
            json.dumps(
                {
                    "action": "final",
                    "reason": "found",
                    "context": "Click the 'Blue widget' link [100] on page 0, then check out.",
                    "citations": [{"file": session_file, "lines": [1, 3]}],
                }
            ),
        ]
    ) as (url, requests):
        # the roles' endpoint is read from the config at open; point a fresh facade at the new stub
        memory.close()
        memory.config_path = _llm_config(tmp_path, url)
        memory.set_query_context(query_invocation_id="inv-7")
        items = memory.query("how do I add the blue widget?")
        assert len(requests) == 2
    assert len(items) == 1
    assert items[0]["value"].startswith("Click the 'Blue widget' link [100]")
    assert f"Sources:\n- {session_file} lines 1-3" in items[0]["value"]
    hook = memory.post_query_hook(query="q", query_image=None, memory_context=items)
    assert hook["read"] == "explorer" and hook["llm_calls"] == 2 and hook["degraded"] is None
    assert hook["citations"] == 1 and hook["export_s"] > 0
    assert (memory.workspace_dir / session_file).is_file()
    memory.close()


def test_save_and_load_round_trip_through_the_backend_hooks(tmp_path):
    memory = AgmemMemory(
        {
            "write": "raw",
            "read": "vector",
            "config": str(_config(tmp_path)),
            "data_dir": str(tmp_path / "store"),
        }
    )
    memory.insert(_public_trajectory("traj-web-1"))
    saved = tmp_path / "memory_state"
    memory.save_memory(saved)
    config = json.loads((saved / "memory_config.json").read_text())
    assert config["memory_type"] == MEMORY_TYPE and "data_dir" not in config["memory_params"]
    assert (saved / "agmem" / "main" / "memory.db").is_file()
    state = json.loads((saved / "agmem_state.json").read_text())
    assert state["inserted_trajectory_ids"] == ["traj-web-1"]
    # the facade was closed for the copy and reopens on use
    assert memory._mem is None
    assert memory.query("blue widget")[0]["value"]

    loaded = AgmemMemory(config["memory_params"] | {"config": str(_config(tmp_path))})
    loaded._load_backend(saved)
    assert loaded.data_dir == saved / "agmem"
    assert loaded._inserted == ["traj-web-1"]
    assert "Blue widget" in loaded.query("blue widget")[0]["value"]
    assert not str(loaded.workspace_dir or "").startswith(
        str(saved)
    )  # a load never writes the artifact
    memory.close()
    loaded.close()


def test_config_and_argv_builders():
    import argparse

    args = argparse.Namespace(
        write="experience", read="explorer", config="~/x.toml", top_k=None, budget_tokens=12000,
        max_steps=6, explorer_budget_tokens=4000, max_state_chars=9000,
        data_root="/data", domain="web", output_dir="/out", reader_model="qwen/qwen3.5-9b",
        reader_base_url="https://openrouter.ai/api/v1", reader_api_key_env="OPENROUTER_API_KEY",
        reader_temperature=0.6, reader_top_p=0.95, reader_top_k=20, max_completion_tokens=20000,
        memory_context_max_tokens=200000, reader_max_concurrent_requests=4,
        evaluator_model="gpt-5.2", evaluator_api_key_env="OPENAI_API_KEY",
        evaluator_reasoning_effort="medium", reader_enable_thinking=False, save_memory=True,
        skip_evaluation=False, load_memory_dir=None,
    )  # fmt: skip
    config = build_memory_config(args)
    assert config["memory_type"] == MEMORY_TYPE
    assert (
        config["memory_params"]["write"] == "experience"
        and config["memory_params"]["max_steps"] == 6
    )
    assert (
        config["memory_params"]["config"].endswith("/x.toml")
        and "top_k" not in config["memory_params"]
    )
    argv = harness_argv(
        args, Path("/out/runtime_inputs"), Path("/out/runtime_inputs/memory_config.json")
    )
    assert argv[:3] == ["evaluation.harness", "--domain", "web"]
    assert "--reader-disable-thinking" in argv and "--save-memory" in argv
    assert (
        argv[argv.index("--prompt-build-max-workers") + 1] == "1"
    )  # latency is measured single-file
    assert argv[argv.index("--model") + 1] == "qwen/qwen3.5-9b"


# ----------------------------------------------------------------------------
# With the upstream checkout
# ----------------------------------------------------------------------------


@needs_upstream
def test_registration_makes_agmem_a_loadable_upstream_memory(tmp_path):
    cls = register_with_upstream(UPSTREAM)
    from memory_modules.memory import MEMORY_TYPES, Memory, build_memory, load_memory

    assert (
        MEMORY_TYPES[MEMORY_TYPE] is cls
        and issubclass(cls, Memory)
        and issubclass(cls, AgmemMemory)
    )
    assert register_with_upstream(UPSTREAM) is cls  # idempotent

    params = {
        "write": "raw",
        "read": "vector",
        "config": str(_config(tmp_path)),
        "data_dir": str(tmp_path / "s"),
    }
    memory = build_memory({"memory_type": MEMORY_TYPE, "memory_params": params})
    assert isinstance(memory, cls)
    memory.insert(_public_trajectory("traj-web-1"))

    # the privacy contract: during a query the backend sees the opaque id and nothing else
    seen = {}
    original = cls.query

    def spying(self, query, query_image=None):
        seen["context"] = self.get_query_context()
        return original(self, query, query_image)

    cls.query = spying
    try:
        memory.set_query_context(query_invocation_id="opaque-invocation-id")
        items = memory.query("blue widget")
    finally:
        cls.query = original
        memory.clear_query_context()
    assert seen["context"] == {"query_invocation_id": "opaque-invocation-id"} and items
    with pytest.raises(TypeError):
        memory.set_query_context(query_invocation_id="x", question_id="q1")

    saved = tmp_path / "memory_state"
    memory.save_memory(saved)
    memory.close()
    # The requesting run names its own data_dir (as `run` does); paths are not part of the match.
    requested = {
        "memory_type": MEMORY_TYPE,
        "memory_params": params | {"data_dir": str(tmp_path / "elsewhere")},
    }
    loaded = load_memory(saved, requested)  # upstream's loader, our reconcile
    # The read side may change on load: the same raw store, read by the explorer.
    as_explorer = load_memory(
        saved,
        {
            "memory_type": MEMORY_TYPE,
            "memory_params": params | {"read": "explorer", "max_steps": 3},
        },
    )
    assert as_explorer.read == "explorer" and as_explorer.max_steps == 3
    assert as_explorer.data_dir == saved / "agmem"
    as_explorer.close()
    # The write side may not: a store built raw is not an experience store.
    with pytest.raises(RuntimeError, match="write side"):
        load_memory(
            saved, {"memory_type": MEMORY_TYPE, "memory_params": params | {"write": "experience"}}
        )
    assert isinstance(loaded, cls) and loaded.data_dir == saved / "agmem"
    assert "Blue widget" in loaded.query("blue widget")[0]["value"]
    loaded.close()


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "haystacks").mkdir(parents=True)
    questions = [
        {"id": "q1", "domain": "web", "question": "Where is the blue widget?", "image": None,
         "answer": "page 0", "question_type": "static-environment", "eval_function": "norm_phrase_set_match|lower=true"},
        {"id": "q2", "domain": "web", "question": "What did the agent click?", "image": None,
         "answer": "[100]", "question_type": "procedure", "eval_function": "norm_phrase_set_match|lower=true"},
        {"id": "q3", "domain": "enterprise", "question": "Which user?", "image": None,
         "answer": "42", "question_type": "static-environment", "eval_function": "norm_phrase_set_match|lower=true"},
    ]  # fmt: skip
    (root / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    trajectories = [
        _public_trajectory("traj-web-1"),
        _public_trajectory("traj-web-2"),
        _internal_trajectory("traj-ent-1"),
    ]
    (root / "trajectories.jsonl").write_text("\n".join(json.dumps(t) for t in trajectories) + "\n")
    (root / "haystacks" / "lme_v2_small.json").write_text(
        json.dumps(
            {
                "q1": ["traj-web-1", "traj-web-2"],
                "q2": ["traj-web-1", "traj-web-2"],
                "q3": ["traj-ent-1"],
            }
        )
    )
    return root


@needs_upstream
def test_estimate_prices_the_haystack_and_run_refuses_without_a_cap(tmp_path, capsys):
    root = _data_root(tmp_path)
    common = [
        "--upstream",
        str(UPSTREAM),
        "--data-root",
        str(root),
        "--domain",
        "web",
        "--tier",
        "small",
    ]
    assert (
        main(
            ["estimate", *common, "--write", "experience", "--read", "explorer", "--max-steps", "4"]
        )
        == 0
    )
    est = json.loads(capsys.readouterr().out)
    assert est["questions"] == 2 and est["trajectories"] == 2 and est["states"] == 6
    assert est["distill"]["calls"] == 2 and est["explore"]["calls"] == 2 * (4 + 2)
    assert est["reader"]["calls"] == 2 and est["total_usd_est"] > 0
    assert est["judge"]["model"] == "gpt-5.2"

    assert main(["estimate", *common, "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["questions"] == 1

    out = tmp_path / "run"
    assert main(["run", *common, "--output-dir", str(out)]) == 2  # no --max-usd
    assert "--max-usd" in capsys.readouterr().err
    assert main(["run", *common, "--output-dir", str(out), "--max-usd", "0.0"]) == 2  # over the cap
    assert "exceeds --max-usd" in capsys.readouterr().err
    assert not (out / "runtime_inputs").exists()  # refused before writing anything


def test_vector_windows_merge_neighbours_and_stop_at_the_budget(tmp_path):
    """Hits on states 0, 1 and 2 of one trajectory are one window, not three;
    the window carries the neighbours a hit alone would not; and the budget
    stops further windows rather than clipping one mid-state."""
    memory = AgmemMemory(
        {"write": "raw", "read": "vector", "config": str(_config(tmp_path)), "data_dir": str(tmp_path / "store"),
         "top_k": 12, "budget_tokens": 120}
    )  # fmt: skip
    memory.insert(_public_trajectory("traj-web-1", n_states=3))
    memory.insert(_public_trajectory("traj-web-2", n_states=3))
    memory.set_query_context(query_invocation_id="inv-2")
    text = memory.query("Blue widget link on the shop page")[0]["value"]
    hook = memory.post_query_hook(query="q", query_image=None, memory_context=[])
    assert text.count("### Trajectory") == hook["windows"] == 1  # 480 chars: the first window only
    assert "(states 0-2 of 3)" in text  # all three states of that trajectory, merged
    assert text.count("State ") == 3 and "State 1 (step 1)" in text

    # the layout survives save/load, so a loaded store renders windows too
    saved = tmp_path / "ms"
    memory.save_memory(saved)
    state = json.loads((saved / "agmem_state.json").read_text())
    assert state["layout"]["traj-web-1"]["states"] == [[1, 2], [3, 4], [5, 6]]
    loaded = AgmemMemory({"write": "raw", "read": "vector", "config": str(_config(tmp_path))})
    loaded._load_backend(saved)
    assert "### Trajectory" in loaded.query("Blue widget")[0]["value"]
    memory.close()
    loaded.close()


def test_configure_runtime_warms_the_embedder_before_the_timed_queries(tmp_path):
    memory = AgmemMemory({"write": "raw", "read": "vector", "config": str(_config(tmp_path))})
    assert memory._mem is None
    memory.configure_runtime(query_trace_dir=str(tmp_path / "traces"), generation_temperature=0.6)
    assert memory._mem is not None  # opened, and a search ran, before any question
    memory.close()
