"""ReasoningBank's released MaTTS parallel path cannot have produced the paper's
MaTTS numbers: `induce_scaling.py` raises before it ever reaches the LLM, and the
"multiple trajectories" it would have contrasted are N copies of one trajectory.

Five independent facets, all static or dependency-free, all $0:

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

Plus the sequential half: `SEQUENTIAL_PROMPT`/`SEQUENTIAL_FOLLOWING_PROMPT` are
defined and referenced NOWHERE else in the snapshot, so neither MaTTS half is
runnable from the released code. This is why our `on_scaled_task_end` is written
against the paper + `PARALLEL_SI` rather than against a runnable upstream: there
is no runnable upstream to reproduce. It is the one place in this project where
"reproduce it as shipped" is not an available option — a crash cannot be a
faithful arm.

Evidence: docs/research/upstream-defect-catalog.md §8 (RB-9); docs/17-defect-ledger.md B-7.
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

    # The bound method the call site *thinks* it is calling.
    method = next(
        m
        for cls in clients.body
        if isinstance(cls, ast.ClassDef)
        for m in cls.body
        if isinstance(m, ast.FunctionDef) and m.name == "one_step_chat"
    )
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

    clients = _load(root, "utils/clients.py")
    method = next(
        m
        for cls in clients.body
        if isinstance(cls, ast.ClassDef)
        for m in cls.body
        if isinstance(m, ast.FunctionDef) and m.name == "one_step_chat"
    )
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
    prove_sequential_unwired(root)


if __name__ == "__main__":
    main()
