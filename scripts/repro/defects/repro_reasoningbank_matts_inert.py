"""ReasoningBank's released MaTTS parallel path cannot have produced the paper's
MaTTS numbers: `induce_scaling.py` raises before it ever reaches the LLM, and the
"multiple trajectories" it would have contrasted are N copies of one trajectory.

Six independent facets, all static or dependency-free, all $0:

1. CRASH. `client = CLIENT_DICT[args.model]` binds the CLASS (the dict's values
   are class objects; the sibling `induce_memory.py` instantiates with
   `(model_name=...)`). `one_step_chat` carries no staticmethod/classmethod
   decorator, so the unbound call binds `self=trajectories` and leaves the
   required `text` parameter unfilled -> TypeError at the call itself. Proven
   twice over: statically, and by rebuilding the pinned signature and making the
   pinned call against it.
2. DUPLICATE TRAJECTORIES. Inside `for i in range(num_samples)`, both `res_dir`
   and `cur_task` are loop-INVARIANT, and the caller passes a single
   `results_{i}` (the loop's leftover binding, i.e. the last trial only) from
   OUTSIDE its spawn loop. So `**Trajectory 1..N :**` are byte-identical.
3. RETURN VALUE NOT UNPACKED. `one_step_chat` returns `tuple[str, <completion>]`;
   `induce_memory.py` unpacks it, `induce_scaling.py` does not — so even with (1)
   fixed, the tuple reaches `json.dumps`.
4. BANK ENTRY SHAPE MISMATCH. `induce_memory.py` stores
   `memory_items = <text>.split("\\n\\n")` (a list); `induce_scaling.py` stores the
   bare value, which the consumer (`run.py`) iterates — over characters, if a str.
5. INVERTED LABEL (ledger B-7). `reward == 0 -> "success"` here vs
   `reward == 1 -> "success"` in the sibling file, for a field nothing renders.
6. ADVERTISED MODELS BROKEN (catalog RB-11). Both induce CLIs offer the same four
   `--model` choices; "gpt-3.5" is not a `CLIENT_DICT` key (the key is
   "gpt-3.5-turbo") -> KeyError, and "gpt-4o" resolves to `GPT4V_Client`, whose
   `one_step_chat` requires an `image` and takes no `temperature` -> TypeError
   even in the sibling that RUNS. Meanwhile the paper's third backbone
   (Claude-3.7-Sonnet, Table 1) is in `CLIENT_DICT` but in neither choices list.

Plus the sequential half: `SEQUENTIAL_PROMPT`/`SEQUENTIAL_FOLLOWING_PROMPT` are
defined and referenced NOWHERE else in the snapshot, so neither MaTTS half is
runnable from the released code. This is why our `on_scaled_task_end` is written
against the paper + `PARALLEL_SI` rather than against a runnable upstream: there
is no runnable upstream to reproduce. It is the one place in this project where
"reproduce it as shipped" is not an available option — a crash cannot be a
faithful arm.

Evidence: docs/research/upstream-defect-catalog.md §8 (RB-9, RB-11);
docs/17-defect-ledger.md B-7 (and A-5, which RB-11 corroborates from the CLI surface).
"""

import ast

from _common import proven, upstream

WEBARENA = "WebArena"


def _load(root, rel: str) -> ast.Module:
    return ast.parse((root / WEBARENA / rel).read_text())


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _names_loaded(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _model_option(tree: ast.Module) -> tuple[str, list[str]]:
    """The `--model` add_argument's `(default, choices)` literals."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        first = node.args[0] if node.args else None
        if not (isinstance(first, ast.Constant) and first.value == "--model"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        default, choices = kwargs.get("default"), kwargs.get("choices")
        assert isinstance(default, ast.Constant), "--model's default is no longer a literal"
        assert isinstance(choices, ast.List | ast.Tuple), (
            "--model's choices are no longer a literal"
        )
        return default.value, [ast.literal_eval(c) for c in choices.elts]
    raise AssertionError("no --model add_argument found")


def _client_dict(root) -> dict[str, str]:
    """`CLIENT_DICT` as {model key -> class name}, read from the pinned clients.py."""
    clients = _load(root, "utils/clients.py")
    node = next(
        n.value
        for n in clients.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CLIENT_DICT" for t in n.targets)
    )
    assert isinstance(node, ast.Dict)
    pairs = list(zip(node.keys, node.values))
    assert all(isinstance(k, ast.Constant) and isinstance(v, ast.Name) for k, v in pairs), (
        "CLIENT_DICT is no longer a flat literal -> class mapping"
    )
    return {k.value: v.id for k, v in pairs}


def _class_method(root, class_name: str, method: str) -> ast.FunctionDef:
    clients = _load(root, "utils/clients.py")
    cls = next(
        (n for n in clients.body if isinstance(n, ast.ClassDef) and n.name == class_name), None
    )
    assert cls is not None, f"class {class_name} is gone from clients.py"
    for m in cls.body:
        if isinstance(m, ast.FunctionDef) and m.name == method:
            return m
    raise AssertionError(f"{class_name} no longer defines {method}")


def _reached_one_step_chat(root) -> ast.FunctionDef:
    """THE `one_step_chat` the pinned call site actually reaches.

    A first-match scan over `clients.py` lands on `GPT_Client` (:30), which is NOT
    what the pinned call reaches: `induce_scaling.py`'s `--model` defaults to
    "gemini-2.5-flash", and `CLIENT_DICT` maps that to `GEMINI_Client` (:126).
    Today the four signatures make that harmless — GPT's (:52) and GEMINI's (:149)
    are identical, and a divergence would fail loud on the `'text'` assertion below
    rather than pass wrongly. But "whichever copy the walk hit first" is exactly
    defect class 3 ("which copy produced the number?"), the class catalog RB-10(b)
    records us committing in prose about this same repo. A prover that commits it
    in miniature cannot be the thing that catches it, so resolve the class the
    call site reaches instead of the class the file happens to define first.
    """
    default, _choices = _model_option(_load(root, "induce_scaling.py"))
    mapping = _client_dict(root)
    assert default in mapping, (
        f"--model default {default!r} is not a CLIENT_DICT key ({sorted(mapping)}) — "
        "the pinned call site no longer resolves to any client"
    )
    return _class_method(root, mapping[default], "one_step_chat")


def prove_crash(root) -> None:
    """(1) The call site never instantiates, and the method is not static."""
    scaling = _load(root, "induce_scaling.py")
    memory = _load(root, "induce_memory.py")

    def client_assign(tree: ast.Module) -> ast.AST:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("client", "llm_client") for t in node.targets
            ):
                return node.value
        raise AssertionError("no client assignment found")

    scaled_rhs = client_assign(scaling)
    single_rhs = client_assign(memory)
    assert isinstance(scaled_rhs, ast.Subscript), (
        f"induce_scaling's client is no longer a bare lookup: {ast.dump(scaled_rhs)[:80]}"
    )
    assert isinstance(single_rhs, ast.Call), (
        "induce_memory no longer instantiates — the asymmetry that proves the bug is gone"
    )

    clients = _load(root, "utils/clients.py")
    classes = {n.name for n in clients.body if isinstance(n, ast.ClassDef)}
    dict_node = next(
        node.value
        for node in clients.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CLIENT_DICT" for t in node.targets)
    )
    assert isinstance(dict_node, ast.Dict)
    values = [v for v in dict_node.values]
    assert all(isinstance(v, ast.Name) and v.id in classes for v in values), (
        "CLIENT_DICT no longer maps to bare classes"
    )

    # The bound method the call site *thinks* it is calling — resolved through the
    # pinned `--model` default, not by first match (see `_reached_one_step_chat`).
    method = _reached_one_step_chat(root)
    decorators = {d.id for d in method.decorator_list if isinstance(d, ast.Name)}
    assert not decorators & {"staticmethod", "classmethod"}, (
        f"one_step_chat became {decorators} — the unbound call would then be legal"
    )
    assert method.args.args[0].arg == "self", "first parameter is no longer self"

    # Dependency-free dynamic proof: rebuild the pinned signature, make the pinned
    # call. No google.genai import, no network, no model.
    stub = ast.Module(
        body=[
            ast.FunctionDef(
                name="one_step_chat",
                args=method.args,
                body=[ast.Pass()],
                decorator_list=[],
                returns=None,
                type_params=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(stub)
    scope: dict = {}
    exec(compile(stub, "<pinned-signature>", "exec"), scope)  # noqa: S102 — synthesized from the pin
    try:
        # exactly induce_scaling.py's call, with the class-bound function unbound
        scope["one_step_chat"]("<trajectories>", system_msg="<PARALLEL_SI>", temperature=0.7)
    except TypeError as exc:
        assert "text" in str(exc), f"crashed, but not on the missing 'text': {exc}"
    else:
        raise AssertionError("the pinned call no longer raises — signature or call site changed")

    proven(
        "induce_scaling calls one_step_chat unbound -> TypeError (missing 'text') before any LLM call"
    )


def prove_duplicate_trajectories(root) -> None:
    """(2) The N contrasted trajectories are one trajectory, N times."""
    main = _func(_load(root, "induce_scaling.py"), "main")
    loop = next(n for n in main.body if isinstance(n, ast.For))
    assert isinstance(loop.target, ast.Name)
    loop_var = loop.target.id

    invariant = {}
    for stmt in loop.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id in ("res_dir", "cur_task"):
                    invariant[target.id] = _names_loaded(stmt.value)
    assert set(invariant) == {"res_dir", "cur_task"}, f"assignments moved: {sorted(invariant)}"
    for name, loaded in invariant.items():
        assert loop_var not in loaded, f"{name} now varies with {loop_var} — the bug is fixed"

    # ...and the caller hands it a single directory, chosen by the leftover binding.
    pipeline = _func(_load(root, "pipeline_scaling.py"), "main")
    spawn_loop = next(
        n
        for n in ast.walk(pipeline)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "i"
    )
    inside = set(map(id, ast.walk(spawn_loop)))

    def is_induce_call(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and any(
            isinstance(c, ast.Constant) and c.value == "induce_scaling.py" for c in ast.walk(node)
        )

    induce_calls = [n for n in ast.walk(pipeline) if is_induce_call(n)]
    assert len(induce_calls) == 1, f"expected one induce_scaling spawn, found {len(induce_calls)}"
    assert id(induce_calls[0]) not in inside, (
        "the induction spawn moved inside the trial loop — it would then see each trial"
    )
    # the --result_dir it passes still interpolates the loop variable
    fstrings = [n for n in ast.walk(induce_calls[0]) if isinstance(n, ast.JoinedStr)]
    assert any("i" in _names_loaded(f) for f in fstrings), (
        "--result_dir no longer interpolates the leftover loop variable"
    )

    proven(
        "induce_scaling reads ONE result dir N times: **Trajectory 1..N :** are identical copies"
    )


def prove_result_never_stored(root) -> None:
    """(3) + (4): the return value is neither unpacked nor split."""
    scaling = _load(root, "induce_scaling.py")
    memory = _load(root, "induce_memory.py")

    def assign_target(tree: ast.Module) -> ast.AST:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr == "one_step_chat":
                    return node.targets[0]
        raise AssertionError("no one_step_chat assignment found")

    assert isinstance(assign_target(memory), ast.Tuple), "induce_memory no longer unpacks the tuple"
    assert isinstance(assign_target(scaling), ast.Name), "induce_scaling now unpacks — bug fixed"

    method = _reached_one_step_chat(root)
    assert method.returns is not None and "tuple" in ast.unparse(method.returns), (
        "one_step_chat no longer advertises a tuple return"
    )

    def memory_items_value(tree: ast.Module) -> ast.AST:
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "memory_items":
                        return value
        raise AssertionError("no memory_items entry found")

    split = memory_items_value(memory)
    assert isinstance(split, ast.Call) and getattr(split.func, "attr", None) == "split", (
        "induce_memory no longer splits into a list"
    )
    assert isinstance(memory_items_value(scaling), ast.Name), (
        "induce_scaling now post-processes the value — shape mismatch fixed"
    )

    proven("scaled bank entry stores an unsplit, unpacked value where the consumer expects a list")


def prove_inverted_label(root) -> None:
    """(5) ledger B-7: the same dead field is inverted against its sibling."""

    def success_literal(tree: ast.Module) -> int:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "reward":
                    assigns = [s for s in node.body if isinstance(s, ast.Assign)]
                    if (
                        assigns
                        and isinstance(assigns[0].value, ast.Constant)
                        and assigns[0].value.value == "success"
                    ):
                        comparator = node.test.comparators[0]
                        assert isinstance(comparator, ast.Constant)
                        return comparator.value
        raise AssertionError("no reward->status branch found")

    scaled = success_literal(_load(root, "induce_scaling.py"))
    single = success_literal(_load(root, "induce_memory.py"))
    assert (scaled, single) == (0, 1), f"inversion gone: scaled={scaled}, single={single}"
    proven(
        f"MaTTS labels reward=={scaled} as success; the sibling labels reward=={single} — inverted"
    )


def prove_advertised_models_broken(root) -> None:
    """(6) catalog RB-11: half the advertised `--model` surface cannot reach a model
    call — in the crashing file AND in the sibling that runs — and the paper's third
    backbone is present in the code but absent from the CLI that would select it."""
    scaling_default, scaling_choices = _model_option(_load(root, "induce_scaling.py"))
    memory_default, memory_choices = _model_option(_load(root, "induce_memory.py"))
    assert (scaling_default, scaling_choices) == (memory_default, memory_choices), (
        "the two induce CLIs no longer advertise the same models — the defect is now "
        f"file-specific: scaling={scaling_default, scaling_choices}, "
        f"memory={memory_default, memory_choices}"
    )
    assert scaling_default == "gemini-2.5-flash", f"default moved: {scaling_default}"
    assert scaling_choices == ["gpt-3.5", "gpt-4", "gpt-4o", "gemini-2.5-flash"], (
        f"the advertised choice list changed: {scaling_choices}"
    )

    mapping = _client_dict(root)

    # (a) "gpt-3.5" is advertised but is not a key — the real key carries the -turbo
    #     suffix, so selecting the advertised name KeyErrors at the lookup itself.
    assert "gpt-3.5" not in mapping, "'gpt-3.5' became a CLIENT_DICT key — the KeyError is fixed"
    assert "gpt-3.5-turbo" in mapping, "the -turbo key is gone; the near-miss no longer holds"

    # (b) "gpt-4o" resolves to the VISION client, whose one_step_chat requires an
    #     `image` and accepts no `temperature` — so it dies on the sibling's own
    #     call too, i.e. this half is broken in the file that RUNS, not only in the
    #     file that crashes for the unrelated unbound-call reason (facet 1).
    vision = _class_method(root, mapping["gpt-4o"], "one_step_chat")
    positional = [a.arg for a in vision.args.args]
    first_defaulted = len(positional) - len(vision.args.defaults)
    assert "image" in positional, f"{mapping['gpt-4o']}.one_step_chat lost its image parameter"
    assert positional.index("image") < first_defaulted, (
        "image gained a default — no longer required"
    )
    assert "temperature" not in positional, "the vision client now takes a temperature"

    memory_calls = [
        n
        for n in ast.walk(_load(root, "induce_memory.py"))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "one_step_chat"
    ]
    assert memory_calls, "induce_memory no longer calls one_step_chat"
    assert all(any(kw.arg == "temperature" for kw in c.keywords) for c in memory_calls), (
        "induce_memory stopped passing temperature= — the TypeError against gpt-4o is gone"
    )

    # (c) The paper's third backbone (Claude-3.7-Sonnet, Table 1) is wired in the
    #     code and unreachable from the CLI: a client nobody can ask for.
    assert "claude-3-7-sonnet@20250219" in mapping, "the Claude client left CLIENT_DICT"
    assert not any("claude" in c for c in scaling_choices), (
        f"Claude became selectable after all: {scaling_choices}"
    )

    proven(
        "half the advertised --model surface dies before any model call (gpt-3.5 KeyError, "
        "gpt-4o TypeError) even in the sibling that runs, and the paper's Claude backbone "
        "cannot be selected at all"
    )


def prove_sequential_unwired(root) -> None:
    """The other MaTTS half: defined, referenced nowhere."""
    definition = root / WEBARENA / "prompts/memory_instruction.py"
    hits = []
    for path in root.rglob("*.py"):
        if ".git" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if "SEQUENTIAL" in line:
                hits.append((path, lineno, line.strip()))
    outside = [h for h in hits if h[0] != definition]
    assert not outside, f"SEQUENTIAL is wired after all: {outside}"
    assert all(h[2].startswith("SEQUENTIAL") and "=" in h[2] for h in hits), (
        f"unexpected SEQUENTIAL usage inside the definition file: {hits}"
    )
    proven(
        f"SEQUENTIAL_PROMPT: {len(hits)} definition(s), 0 references — sequential MaTTS is unwired"
    )


def main() -> None:
    root = upstream("reasoning-bank")
    prove_crash(root)
    prove_duplicate_trajectories(root)
    prove_result_never_stored(root)
    prove_inverted_label(root)
    prove_advertised_models_broken(root)
    prove_sequential_unwired(root)


if __name__ == "__main__":
    main()
