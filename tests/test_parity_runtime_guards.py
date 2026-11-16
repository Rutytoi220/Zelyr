"""Runtime guards that keep weak models from weakening Zelyr (cross-model parity audit).

Covers:
- Explicit user prohibitions are enforced by the runtime, not by model compliance.
- Replanning budget is progress-aware (N-file tasks) yet strictly bounded; no-progress still stops.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger


def _write_call(idx: int, call_id: str, path: str, content: str) -> dict:
    return {
        "index": idx,
        "id": call_id,
        "type": "function",
        "function": {
            "name": "manage_workspace_file",
            "arguments": json.dumps({"action": "write", "path": path, "content": content}),
        },
    }


def _tools(*calls):
    return {"choices": [{"delta": {"tool_calls": list(calls)}}]}


def _text(t: str):
    return {"choices": [{"delta": {"content": t}}]}


class TestProhibitedTargetExtraction:
    def test_explicit_prohibition_extracted(self, tmp_path):
        b = tmp_path / "beta.txt"
        got = NativeOrchestrator._extract_prohibited_targets(f"Create a.txt. Do NOT create {b} under any circumstances.")
        assert got == [str(b.resolve())]

    def test_dont_write_variant_and_quotes(self, tmp_path):
        b = tmp_path / "beta.txt"
        got = NativeOrchestrator._extract_prohibited_targets(f"Never write to '{b}'.")
        assert got == [str(b.resolve())]

    def test_incidental_mention_is_not_a_prohibition(self, tmp_path):
        b = tmp_path / "beta.txt"
        assert NativeOrchestrator._extract_prohibited_targets(f"Create a.txt. {b} is handled later by another team.") == []

    def test_do_not_modify_afterward_is_not_creation_ban(self):
        assert NativeOrchestrator._extract_prohibited_targets("Create a.txt and do not create a.txt again") == []

    def test_non_path_token_ignored(self):
        assert NativeOrchestrator._extract_prohibited_targets("Do not create anything risky") == []


@pytest.mark.asyncio
async def test_model_cannot_write_user_prohibited_file(tmp_path):
    a, b = tmp_path / "alpha.txt", tmp_path / "beta.txt"
    executed = []

    async def executor(name, **kw):
        executed.append(kw["path"])
        Path(kw["path"]).write_text(kw["content"], encoding="utf-8")
        return {"status": "success", "path": kw["path"]}

    turn = 0

    async def stream(payload):
        nonlocal turn
        turn += 1
        if turn == 1:
            yield _tools(_write_call(0, "c_alpha", str(a), "ALPHA_101"))
        elif turn == 2:
            yield _tools(_write_call(0, "c_beta", str(b), "BETA_202"))  # weak model ignores the ban
        else:
            yield _text("Done.")

    ledger = TurnExecutionLedger()
    payload = {
        "messages": [{"role": "user", "content": f"Create {a} containing ALPHA_101. Do NOT create {b} under any circumstances."}],
        "acceptance_criteria": f"file_contains:{a}:ALPHA_101",
    }
    async for _ in NativeOrchestrator().generate_stream(payload, stream_provider=stream, tool_executor=executor, ledger=ledger):
        pass

    assert a.read_text() == "ALPHA_101"
    assert not b.exists()
    assert str(b) not in executed
    rec = ledger._records["c_beta"]
    assert rec.state == ActionState.FAILED
    assert "prohibited_by_user" in str(rec.error)


@pytest.mark.asyncio
async def test_non_prohibited_sibling_file_still_allowed(tmp_path):
    a, b = tmp_path / "alpha.txt", tmp_path / "beta.txt"

    async def executor(name, **kw):
        Path(kw["path"]).write_text(kw["content"], encoding="utf-8")
        return {"status": "success", "path": kw["path"]}

    turn = 0

    async def stream(payload):
        nonlocal turn
        turn += 1
        if turn == 1:
            yield _tools(_write_call(0, "c1", str(a), "ALPHA_101"))
        elif turn == 2:
            yield _tools(_write_call(0, "c2", str(b), "BETA_202"))
        else:
            yield _text("Done.")

    payload = {
        "messages": [{"role": "user", "content": f"Create {a} containing ALPHA_101, then create {b} containing BETA_202."}],
        "acceptance_criteria": f"file_contains:{a}:ALPHA_101,file_contains:{b}:BETA_202",
    }
    async for _ in NativeOrchestrator().generate_stream(payload, stream_provider=stream, tool_executor=executor, ledger=TurnExecutionLedger()):
        pass
    assert a.exists() and b.exists()


@pytest.mark.asyncio
async def test_progress_aware_replanning_completes_three_file_task(tmp_path):
    files = [tmp_path / n for n in ("alpha.txt", "beta.txt", "gamma.txt")]
    contents = ["ALPHA_101", "BETA_202", "GAMMA_303"]
    created = []

    async def executor(name, **kw):
        Path(kw["path"]).write_text(kw["content"], encoding="utf-8")
        created.append(kw["path"])
        return {"status": "success", "path": kw["path"]}

    turn = 0

    async def stream(payload):
        nonlocal turn
        turn += 1
        # Odd turns: model emits exactly one fresh call; even turns: text-only claim.
        if turn % 2 == 1 and (turn // 2) < 3:
            i = turn // 2
            yield _tools(_write_call(0, f"c{i}", str(files[i]), contents[i]))
        else:
            yield _text("Completed.")

    criteria = ",".join(f"file_contains:{p}:{c}" for p, c in zip(files, contents))
    payload = {
        "messages": [{"role": "user", "content": "Create three files: " + ", ".join(f"{p} containing {c}" for p, c in zip(files, contents))}],
        "acceptance_criteria": criteria,
    }
    orch = NativeOrchestrator()
    orch.max_steps = 10
    async for _ in orch.generate_stream(payload, stream_provider=stream, tool_executor=executor, ledger=TurnExecutionLedger()):
        pass
    assert all(p.exists() for p in files), created
    assert created == [str(p) for p in files]  # exactly one write per file, no duplicates


@pytest.mark.asyncio
async def test_replanning_without_progress_remains_bounded(tmp_path):
    a, b = tmp_path / "alpha.txt", tmp_path / "beta.txt"
    calls = 0

    async def executor(name, **kw):
        Path(kw["path"]).write_text(kw["content"], encoding="utf-8")
        return {"status": "success", "path": kw["path"]}

    async def stream(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield _tools(_write_call(0, "c1", str(a), "ALPHA_101"))
        else:
            yield _text("All done, nothing more to do.")  # never emits the beta call

    payload = {
        "messages": [{"role": "user", "content": f"Create {a} containing ALPHA_101 and {b} containing BETA_202."}],
        "acceptance_criteria": f"file_contains:{a}:ALPHA_101,file_contains:{b}:BETA_202",
    }
    async for _ in NativeOrchestrator().generate_stream(payload, stream_provider=stream, tool_executor=executor, ledger=TurnExecutionLedger()):
        pass
    assert not b.exists()
    assert calls <= 4  # initial + tool follow-up + exactly one replanning attempt (+ slack), never unbounded


class TestAdversarialProhibitionSemantics:
    """Section 1 & 5.1-5.3: Explicit prohibitions vs incidental mentions, multi-target, verbs, and contradictions."""

    @pytest.mark.parametrize(
        "prompt_tpl",
        [
            "Do not create {target}.",
            "Never modify {target}.",
            "Do not edit '{target}'.",
            "Must not overwrite {target}.",
            "Should not append to {target}.",
            "Do not patch {target}.",
            "Never rename {target}.",
            "Do not move {target}.",
            "Do not delete {target}.",
            "Avoid writing to {target}.",
            "Refrain from modifying {target}.",
            "Complete the task without touching {target}.",
            "Leave {target} untouched.",
            "Keep '{target}' intact.",
        ],
    )
    def test_prohibition_verbs_and_negations(self, tmp_path, prompt_tpl):
        t = tmp_path / "secret.cfg"
        got = NativeOrchestrator._extract_prohibited_targets(prompt_tpl.format(target=t))
        assert got == [str(t.resolve())]

    def test_multiple_targets_in_one_prohibition(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        got = NativeOrchestrator._extract_prohibited_targets(f"Do not create {a} or {b}.")
        assert got == sorted([str(a.resolve()), str(b.resolve())])

    def test_incidental_historical_and_meta_quotes_not_prohibited(self, tmp_path):
        a = tmp_path / "a.txt"
        assert NativeOrchestrator._extract_prohibited_targets(f"If {a} does not exist, create {a}.") == []
        assert NativeOrchestrator._extract_prohibited_targets(f"Previously I could not write {a}, please fix it.") == []
        assert NativeOrchestrator._extract_prohibited_targets(f"Explain what \"Do not delete {a}\" means.") == []
        fenced = f"Create notes.md with:\n```\nDo not delete {a}\n```"
        assert NativeOrchestrator._extract_prohibited_targets(fenced) == []

    def test_contradictory_same_verb_fails_closed(self, tmp_path):
        a = tmp_path / "a.txt"
        got = NativeOrchestrator._extract_prohibited_targets(f"Create {a}. Do not create {a}.")
        assert got == [str(a.resolve())]

    @pytest.mark.asyncio
    async def test_post_creation_modification_ban_allows_initial_creation_then_locks(self, tmp_path):
        a = tmp_path / "a.txt"
        ledger = TurnExecutionLedger()
        prompt = f"Create {a} containing INIT_1, do not overwrite {a}."

        # Before creation: a.txt is allowed to be created once
        assert NativeOrchestrator._extract_prohibited_targets(prompt, ledger=ledger) == []

        executed_contents = []

        async def executor(name, **kw):
            executed_contents.append(kw["content"])
            Path(kw["path"]).write_text(kw["content"], encoding="utf-8")
            return {"status": "success", "path": kw["path"]}

        turn = 0

        async def stream(payload):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield _tools(_write_call(0, "c_init", str(a), "INIT_1"))
            elif turn == 2:
                yield _tools(_write_call(0, "c_over", str(a), "OVERWRITTEN_2"))
            else:
                yield _text("Done.")

        payload = {"messages": [{"role": "user", "content": prompt}]}
        async for _ in NativeOrchestrator().generate_stream(
            payload, stream_provider=stream, tool_executor=executor, ledger=ledger
        ):
            pass

        assert a.read_text(encoding="utf-8") == "INIT_1"
        assert executed_contents == ["INIT_1"]
        assert ledger._records["c_over"].state == ActionState.FAILED

    @pytest.mark.asyncio
    async def test_prohibited_directory_blocks_nested_file_mutation(self, tmp_path):
        prohibited_dir = tmp_path / "protected_dir"
        prohibited_dir.mkdir()
        nested_file = prohibited_dir / "sub" / "out.txt"
        allowed_file = tmp_path / "allowed.txt"

        async def executor(name, **kw):
            p = Path(kw["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(kw.get("content", ""), encoding="utf-8")
            return {"status": "success", "path": kw["path"]}

        turn = 0

        async def stream(payload):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield _tools(_write_call(0, "c_bad", str(nested_file), "BAD"))
            elif turn == 2:
                yield _tools(_write_call(0, "c_ok", str(allowed_file), "OK"))
            else:
                yield _text("Done.")

        ledger = TurnExecutionLedger()
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Create {allowed_file} with OK. Do not write into directory {prohibited_dir}/.",
                }
            ]
        }
        async for _ in NativeOrchestrator().generate_stream(
            payload, stream_provider=stream, tool_executor=executor, ledger=ledger
        ):
            pass

        assert not nested_file.exists()
        assert allowed_file.read_text(encoding="utf-8") == "OK"
        assert "prohibited_by_user" in str(ledger._records["c_bad"].error)


class TestTargetIdentityAndDuplicateBasenames:
    """Section 2 & 5.4-5.5: Duplicate basenames in different directories, .. traversal, symlinks, and patch/shell tools."""

    def test_duplicate_basenames_in_different_directories_do_not_collide_in_ledger(self, tmp_path):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        cfg_a = dir_a / "config.json"
        cfg_b = dir_b / "config.json"

        ledger = TurnExecutionLedger()
        ledger.record_target_verified(str(cfg_a))

        assert ledger.is_target_verified(str(cfg_a)) is True
        assert ledger.is_target_verified(str(cfg_b)) is False
        # Non-filesystem check types with same trailing name must never collide with files
        assert ledger.is_target_verified("config.json", check_type="process_running") is False

    @pytest.mark.asyncio
    async def test_duplicate_basenames_allow_second_directory_creation_and_block_first(self, tmp_path):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        cfg_a = dir_a / "config.json"
        cfg_b = dir_b / "config.json"

        async def executor(name, **kw):
            Path(kw["path"]).write_text(kw["content"], encoding="utf-8")
            return {"status": "success", "path": kw["path"]}

        turn = 0

        async def stream(payload):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield _tools(_write_call(0, "c_a", str(cfg_a), '{"env": "a"}'))
            elif turn == 2:
                # Creating dir_b/config.json must succeed even though dir_a/config.json is already verified
                yield _tools(_write_call(0, "c_b", str(cfg_b), '{"env": "b"}'))
            elif turn == 3:
                # Overwriting verified dir_a/config.json via .. traversal must be blocked
                traversed_a = str(dir_b / ".." / "dir_a" / "config.json")
                yield _tools(_write_call(0, "c_a_overwrite", traversed_a, '{"env": "corrupted"}'))
            else:
                yield _text("Done.")

        ledger = TurnExecutionLedger()
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Create {cfg_a} with {{\"env\": \"a\"}} and {cfg_b} with {{\"env\": \"b\"}}.",
                }
            ],
            "acceptance_criteria": f'file_contains:{cfg_a}:"a",file_contains:{cfg_b}:"b"',
        }
        async for _ in NativeOrchestrator().generate_stream(
            payload, stream_provider=stream, tool_executor=executor, ledger=ledger
        ):
            pass

        assert cfg_a.read_text(encoding="utf-8") == '{"env": "a"}'
        assert cfg_b.read_text(encoding="utf-8") == '{"env": "b"}'
        assert "outcome_already_verified" in str(ledger._records["c_a_overwrite"].error)

    @pytest.mark.asyncio
    async def test_prohibition_on_one_directory_basename_does_not_block_sibling_directory(self, tmp_path):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        cfg_a = dir_a / "config.json"
        cfg_b = dir_b / "config.json"

        async def executor(name, **kw):
            Path(kw["path"]).write_text(kw["content"], encoding="utf-8")
            return {"status": "success", "path": kw["path"]}

        turn = 0

        async def stream(payload):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield _tools(_write_call(0, "c_a", str(cfg_a), '{"a": 1}'))
            elif turn == 2:
                yield _tools(_write_call(0, "c_b", str(cfg_b), '{"b": 2}'))
            else:
                yield _text("Done.")

        ledger = TurnExecutionLedger()
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Create {cfg_a}. Do not create {cfg_b}.",
                }
            ],
            "acceptance_criteria": f"file_exists:{cfg_a}",
        }
        async for _ in NativeOrchestrator().generate_stream(
            payload, stream_provider=stream, tool_executor=executor, ledger=ledger
        ):
            pass

        assert cfg_a.exists()
        assert not cfg_b.exists()
        assert "prohibited_by_user" in str(ledger._records["c_b"].error)

    def test_symlink_and_patch_action_detected_as_mutations(self, tmp_path):
        real_file = tmp_path / "verified_real.txt"
        real_file.write_text("SAFE", encoding="utf-8")
        symlink_file = tmp_path / "alias_link.txt"
        symlink_file.symlink_to(real_file)

        verified = [str(real_file.resolve())]
        # 1. manage_workspace_file(action="patch") on symlink resolves to verified target
        hits_patch = NativeOrchestrator._get_mutation_targets(
            "manage_workspace_file",
            {"action": "patch", "path": str(symlink_file), "content": "HACK"},
            verified,
        )
        assert hits_patch == verified

        # 2. execute_command rm on symlink resolves to verified target
        hits_cmd = NativeOrchestrator._get_mutation_targets(
            "execute_command",
            {"command": f"rm -f {symlink_file}"},
            verified,
        )
        assert hits_cmd == verified


class TestMultiCriterionAndVerificationRefresh:
    """Section 3 & 5.6: Multi-criterion same-file partial satisfaction, verification invalidation, and inspection non-resolution."""

    @pytest.mark.asyncio
    async def test_multi_criterion_same_file_allows_second_step_until_all_criteria_pass(self, tmp_path):
        target = tmp_path / "report.txt"

        async def executor(name, **kw):
            act = kw.get("action", "write")
            p = Path(kw["path"])
            if act == "append":
                p.write_text(p.read_text(encoding="utf-8") + kw["content"], encoding="utf-8")
            else:
                p.write_text(kw["content"], encoding="utf-8")
            return {"status": "success", "path": kw["path"]}

        turn = 0

        async def stream(payload):
            nonlocal turn
            turn += 1
            if turn == 1:
                # Satisfies first criterion (HEADER), but not second (FOOTER)
                yield _tools(_write_call(0, "c1", str(target), "HEADER\n"))
            elif turn == 2:
                # Must NOT be blocked by outcome_already_verified because FOOTER is still unfulfilled
                yield _tools(
                    {
                        "index": 0,
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "manage_workspace_file",
                            "arguments": json.dumps(
                                {"action": "append", "path": str(target), "content": "FOOTER\n"}
                            ),
                        },
                    }
                )
            elif turn == 3:
                # Now both criteria pass; a third unrequested overwrite must be blocked
                yield _tools(_write_call(0, "c3", str(target), "CORRUPT"))
            else:
                yield _text("Done.")

        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": f"Write HEADER and FOOTER to {target}."}],
            "acceptance_criteria": [
                {"check": "file_contains", "target": str(target), "expected": "HEADER"},
                {"check": "file_contains", "target": str(target), "expected": "FOOTER"},
            ],
        }
        async for _ in NativeOrchestrator().generate_stream(
            payload, stream_provider=stream, tool_executor=executor, ledger=ledger
        ):
            pass

        assert target.read_text(encoding="utf-8") == "HEADER\nFOOTER\n"
        assert ledger._records["c2"].state in (ActionState.SUCCESS, ActionState.VERIFIED)
        assert "outcome_already_verified" in str(ledger._records["c3"].error)

    def test_inspection_commands_cannot_resolve_failed_mutation(self, tmp_path):
        target = tmp_path / "data.txt"
        ledger = TurnExecutionLedger()
        ledger.record_proposed("c_fail", "manage_workspace_file", {"action": "write", "path": str(target)})
        ledger.record_dispatched("c_fail")
        ledger.record_observation("c_fail", status="error", output="", error="Permission denied")

        for idx, inspect_args in enumerate(
            [
                ("manage_workspace_file", {"action": "list_tree", "path": str(target)}),
                ("execute_command", {"command": f"wc -l {target}"}),
                ("execute_command", {"command": f"sha256sum {target}"}),
                ("execute_command", {"command": f"git status {target}"}),
            ]
        ):
            cid = f"c_insp_{idx}"
            ledger.record_proposed(cid, inspect_args[0], inspect_args[1])
            ledger.record_dispatched(cid)
            ledger.record_observation(cid, status="success", output="ok")

        assert ledger.has_unresolved_failures() is True


class TestProgressBasedReplanningGuards:
    """Section 4 & 5.7: Genuine progress vs repeat/unrelated verification, max_replanning=0, truncation, and pending proposals."""

    def test_replan_budget_rejects_max_zero_and_repeat_or_unrelated_verification(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        ledger = TurnExecutionLedger()
        ledger.record_target_verified(str(a))
        ledger._replan_verified_marker = len(ledger.get_verified_targets())
        ledger._replan_verified_outcomes = [f"Target '{a}' verified"]
        ledger._replan_unfulfilled_count = 1

        # 1. max_attempts = 0 never replans
        assert NativeOrchestrator._replan_budget_available(ledger, 0, 0) is False

        # 2. Re-verifying the same target (even via relative alias) does not grow verified set or grant budget
        ledger.record_target_verified("a.txt")
        assert len(ledger.get_verified_targets()) == 1
        assert (
            NativeOrchestrator._replan_budget_available(
                ledger,
                1,
                1,
                verified_outcomes=[f"Target '{a}' verified"],
                unfulfilled_outcomes=[f"Target '{b}' unfulfilled"],
            )
            is False
        )

        # 3. Verifying an unrelated target when task unfulfilled_outcomes did not shrink does not grant budget
        unrelated = tmp_path / "unrelated.txt"
        ledger.record_target_verified(str(unrelated))
        assert (
            NativeOrchestrator._replan_budget_available(
                ledger,
                1,
                1,
                verified_outcomes=[f"Target '{a}' verified"],
                unfulfilled_outcomes=[f"Target '{b}' unfulfilled"],
            )
            is False
        )

        # 4. Genuine task progress (new verified outcome AND fewer unfulfilled outcomes) grants budget up to +3 cap
        assert (
            NativeOrchestrator._replan_budget_available(
                ledger,
                1,
                1,
                verified_outcomes=[f"Target '{a}' verified", f"Target '{b}' verified"],
                unfulfilled_outcomes=[],
            )
            is True
        )
        assert NativeOrchestrator._replan_budget_available(ledger, 4, 1) is False

    @pytest.mark.asyncio
    async def test_truncated_response_does_not_trigger_replanning(self, tmp_path):
        a = tmp_path / "alpha.txt"
        calls = 0

        async def stream(payload):
            nonlocal calls
            calls += 1
            yield {"choices": [{"delta": {"content": "Partial truncated output..."}, "finish_reason": "length"}]}

        payload = {
            "messages": [{"role": "user", "content": f"Create {a} containing ALPHA."}],
            "acceptance_criteria": f"file_contains:{a}:ALPHA",
        }
        async for _ in NativeOrchestrator().generate_stream(
            payload, stream_provider=stream, ledger=TurnExecutionLedger()
        ):
            pass

        assert calls == 1
        assert not a.exists()
