#!/usr/bin/env python3
"""Gate for GF-REGIONAL-BOOT-017: every command that operates on a Kubernetes
object must appear after the command that creates that object.

The failure this catches is a manual that reads correctly but cannot be
executed top to bottom: a `kubectl set env deployment/gpu-fault-api-ha ...`
printed above the `kubectl apply -f` that creates that Deployment fails with
NotFound on a first deployment. Reviewers do not catch it because they already
know the answer; only line-number arithmetic does.

What makes this more than a grep: a command in this manual is rarely one line.
It spans backslash continuations, it pipes `sed manifest.yaml | kubectl apply
-f -`, and it feeds heredocs. The object a statement creates therefore often
lives in a separate manifest file or in the heredoc body, not in the kubectl
words themselves. This script joins statements, then resolves what each one
creates:

  * `kubectl apply -f path.yaml`     -> parse path.yaml for kind/name
  * `sed ... path.yaml | apply -f -` -> same, path taken from the pipeline
  * `apply -f - <<'YAML' ... YAML`   -> parse the heredoc body
  * `kubectl create secret generic X`-> secret/X, even when piped to apply

It also follows provenance through the temp files the manual builds, because
almost nothing here is applied straight from the repo:

  * `sed ... src.yaml > /tmp/x.yaml`     -> /tmp/x.yaml declares src.yaml's objects
  * `kubectl kustomize dir > /tmp/x.yaml` -> renders the overlay to find them
  * `deploy/x/install.sh`                -> objects the script itself applies
  * `helm upgrade --install NAME ...`    -> deployment/NAME

Blocks that are deliberately not part of the linear first-deployment path
(recovery runbooks, rollback procedures, conditional enablement) are exempted
by putting this marker on the line before the fence:

    <!-- boot017-out-of-band: 事后恢复流程，不属于首次部署顺序 -->

The marker sits next to the text it exempts so a reviewer sees it, instead of
an allowlist buried in this script that nobody rereads.

Object names are not always literals: most `kubectl exec` here is
`exec "${CPU_API_POD}"`, where the name came from an earlier `get pods -l ...`
command substitution. Those cannot be checked as type/name pairs, so the same
ordering question is asked of the variable instead — is it assigned above the
line that uses it? Skipping them silently would leave the manual's most common
command shape unchecked.

Rendering the overlay means this script shells out to `kubectl kustomize`.
That is offline (no cluster contacted) and is the only way to know that
deployment/gpu-fault-api-ha comes into existence at the `apply` on the
kustomize output rather than from any file named in the manual.

Usage:
    python3 scripts/check-manual-command-order.py [--verbose] [--manual PATH]

Exit status 0 means no out-of-order command in the regional chapter.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

MANUAL = "docs/部署和运维手册.md"

# The regional chapter: greenfield deploy, in-cluster migration, static
# registration. Bounded by heading text rather than line numbers so edits
# above it do not silently move the window.
CHAPTER_START = "## 5. 区域分离生产部署（唯一生产形态）"
CHAPTER_END = "### 5.4 区域数据面训练任务提交与验收"

# Marker exempting the next fenced block from the ordering rule.
EXEMPT_MARKER = "boot017-out-of-band"

# Verbs that require the object to exist already.
CONSUMING = (
    "set env",
    "set image",
    "rollout status",
    "rollout restart",
    "rollout undo",
    "annotate",
    "label",
    "scale",
    "exec",
    "wait",
)

# Verbs that bring the object into existence. `patch` is deliberately absent:
# apply creates or updates, patch only updates.
CREATING = ("apply", "create", "replace", "expose")

TYPE_ALIASES = {
    "deploy": "deployment",
    "deployments": "deployment",
    "deployment": "deployment",
    "sts": "statefulset",
    "statefulset": "statefulset",
    "ds": "daemonset",
    "daemonset": "daemonset",
    "cj": "cronjob",
    "cronjob": "cronjob",
    "cronjobs": "cronjob",
    "job": "job",
    "jobs": "job",
    "svc": "service",
    "service": "service",
    "secret": "secret",
    "secrets": "secret",
    "cm": "configmap",
    "configmap": "configmap",
    "configmaps": "configmap",
    "sa": "serviceaccount",
    "serviceaccount": "serviceaccount",
    "pdb": "poddisruptionbudget",
    "poddisruptionbudget": "poddisruptionbudget",
    "ns": "namespace",
    "namespace": "namespace",
}

# Objects the manual does not create because something else owns them: the
# cluster, an operator, or an earlier chapter. Referencing one is not an
# ordering error.
PRE_EXISTING = {
    # Pods are created by their controller, never by the manual.
    "pod",
    "node",
    # kubectl exec/logs on a job/deployment resolves through its pods.
}


@dataclass
class Statement:
    line: int
    text: str
    creates: set[str] = field(default_factory=set)
    consumes: set[str] = field(default_factory=set)
    heredoc: list[str] = field(default_factory=list)


def _chapter_bounds(lines: list[str], manual: str = MANUAL) -> tuple[int, int]:
    start = end = None
    for i, line in enumerate(lines, 1):
        if line.strip() == CHAPTER_START:
            start = i
        elif line.strip() == CHAPTER_END:
            end = i
    if start is None or end is None:
        raise SystemExit(
            f"cannot locate regional chapter bounds in {manual}: "
            f"start={start} end={end}"
        )
    return start, end


def _bash_lines(
    lines: list[str], start: int, end: int
) -> tuple[list[tuple[int, str]], set[int], int]:
    """Lines inside ```bash fences within [start, end).

    Returns the lines, the subset exempted by EXEMPT_MARKER, and how many
    blocks were exempted — reported so an exemption can never be silent.
    """
    out: list[tuple[int, str]] = []
    exempt_lines: set[int] = set()
    exempt_blocks = 0
    inside = False
    block_exempt = False
    marker_pending = False

    for i, line in enumerate(lines, 1):
        if line.startswith("```"):
            if line.startswith("```bash"):
                inside = True
                block_exempt = marker_pending
                if block_exempt:
                    exempt_blocks += 1
            else:
                inside = False
                block_exempt = False
            marker_pending = False
            continue
        if not inside:
            # Only an immediately preceding marker counts; prose in between
            # would make the exemption's scope guesswork.
            if EXEMPT_MARKER in line:
                marker_pending = True
            elif line.strip():
                marker_pending = False
            continue
        if start <= i < end:
            out.append((i, line))
            if block_exempt:
                exempt_lines.add(i)
    return out, exempt_lines, exempt_blocks


def _statements(
    bash: list[tuple[int, str]],
) -> list[Statement]:
    """Join continuations, pipelines and heredocs into logical statements."""
    statements: list[Statement] = []
    current: Statement | None = None
    heredoc_end: str | None = None

    for lineno, raw in bash:
        stripped = raw.strip()

        if heredoc_end is not None:
            if stripped == heredoc_end:
                heredoc_end = None
            elif current is not None:
                current.heredoc.append(raw)
            continue

        if not stripped or stripped.startswith("#"):
            if current is not None and not _continues(current.text):
                current = None
            continue

        if current is not None and _continues(current.text):
            if current.text.endswith("\\"):
                current.text = current.text[:-1].rstrip()
            current.text += " " + stripped
        else:
            current = Statement(lineno, stripped)
            statements.append(current)

        match = re.search(
            r"<<-?'?([A-Za-z_][A-Za-z0-9_]*)'?",
            current.text,
        )
        if match:
            heredoc_end = match.group(1)

    return statements


def _continues(text: str) -> bool:
    """True when the next line belongs to the same logical statement."""
    return text.endswith("\\") or text.endswith("|") or text.endswith("&&")


def _clean(token: str) -> str:
    return token.strip().strip("\"'").rstrip(",;")


def _document_objects(text: str) -> set[str]:
    """Top-level kind/name pairs in a manifest, without a YAML dependency.

    Only column-0 `kind:` and the `name:` directly under a column-0
    `metadata:` count, so nested kinds (a Deployment's pod template, a
    RoleBinding's subjects) are not mistaken for objects being created.
    """
    objects: set[str] = set()
    kind: str | None = None
    in_metadata = False
    for raw in text.splitlines():
        if raw.startswith("kind:"):
            kind = raw.split(":", 1)[1].strip()
            in_metadata = False
        elif raw.startswith("metadata:"):
            in_metadata = True
        elif in_metadata and re.match(r"^  name:", raw):
            if kind:
                objects.add(f"{kind.lower()}/{raw.split(':', 1)[1].strip()}")
            in_metadata = False
        elif raw.startswith("---"):
            kind, in_metadata = None, False
        elif raw and not raw.startswith((" ", "-")):
            in_metadata = False
    return objects


_FILE_CACHE: dict[str, set[str]] = {}


def _manifest_objects(path: str) -> set[str]:
    if path in _FILE_CACHE:
        return _FILE_CACHE[path]
    objects: set[str] = set()
    if os.path.exists(path):
        objects = _document_objects(open(path, encoding="utf-8").read())
    _FILE_CACHE[path] = objects
    return objects


def _kustomize_objects(directory: str) -> set[str]:
    """Objects an overlay renders. Offline: no cluster is contacted."""
    key = f"kustomize:{directory}"
    if key in _FILE_CACHE:
        return _FILE_CACHE[key]
    try:
        rendered = subprocess.run(
            ["kubectl", "kustomize", directory],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(
            f"warning: kubectl kustomize {directory} failed ({exc}); "
            "objects it renders will look uncreated",
            file=sys.stderr,
        )
        rendered = ""
    objects = _document_objects(rendered)
    _FILE_CACHE[key] = objects
    return objects


def _script_objects(path: str) -> set[str]:
    """Objects a deploy script applies.

    The manual delegates several steps to a script (install-amp-monitoring.sh,
    deploy-node-installer-reconciler.sh). Treating those steps as creating
    nothing would flag every later reference to what they deploy, so resolve
    one level: the manifests the script itself pipes into kubectl apply.
    """
    key = f"script:{path}"
    if key in _FILE_CACHE:
        return _FILE_CACHE[key]
    objects: set[str] = set()
    if os.path.exists(path):
        body = open(path, encoding="utf-8").read()
        if "apply" in body:
            for match in re.finditer(
                r"[\"']?\$\{REPO_DIR\}/([^\s\"']+\.ya?ml)",
                body,
            ):
                objects |= _manifest_objects(match.group(1))
            for match in re.finditer(
                r"(?:^|\s)"
                r"((?:deploy|testcases|scripts|tests|examples)"
                r"/[^\s|;'\"]+\.ya?ml)",
                body,
            ):
                objects |= _manifest_objects(match.group(1))
            for match in re.finditer(r"create configmap\s+\\?\s*(\S+)", body):
                name = _clean(match.group(1))
                if not name.startswith(("-", "$")):
                    objects.add(f"configmap/{name}")
            # The role-split apply script deliberately resolves its
            # manifests through ${GENERATED}/${manifest}.yaml, so no
            # literal YAML path appears in the shell body. Follow the
            # generated directory it declares instead of treating the
            # script as creating nothing.
            if "regional/generated" in body:
                generated = os.path.join(
                    "deploy",
                    "control-plane",
                    "regional",
                    "generated",
                )
                for manifest in glob.glob(os.path.join(generated, "*.yaml")):
                    objects |= _manifest_objects(manifest)
    _FILE_CACHE[key] = objects
    return objects


def _heredoc_objects(body: list[str]) -> set[str]:
    objects: set[str] = set()
    kind: str | None = None
    for raw in body:
        stripped = raw.strip()
        if stripped.startswith("kind:"):
            kind = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("name:") and kind:
            objects.add(f"{kind.lower()}/{stripped.split(':', 1)[1].strip()}")
            kind = None
    return objects


def _named_targets(text: str) -> set[str]:
    """Objects named inline, as type/name or as adjacent `type name` words."""
    found: set[str] = set()
    for match in re.finditer(r"\b([a-z]+)/([A-Za-z0-9$\"'{][^\s,;)]*)", text):
        kind = TYPE_ALIASES.get(match.group(1))
        if kind:
            found.add(f"{kind}/{_clean(match.group(2))}")
    tokens = text.split()
    for i, token in enumerate(tokens):
        kind = TYPE_ALIASES.get(token)
        if not kind or i + 1 >= len(tokens):
            continue
        nxt = tokens[i + 1]
        if nxt in ("generic", "docker-registry", "tls"):
            if i + 2 >= len(tokens):
                continue
            nxt = tokens[i + 2]
        if nxt.startswith("-") or "=" in nxt:
            continue
        found.add(f"{kind}/{_clean(nxt)}")
    return found


def _manifest_paths(text: str) -> list[str]:
    return re.findall(
        r"(?:^|\s)"
        r"((?:deploy|testcases|docs|scripts|tests|examples)"
        r"/[^\s|;'\"]+\.ya?ml)",
        text,
    )


def _generated_paths(text: str) -> list[str]:
    return re.findall(r"(/tmp/[^\s|;'\"]+\.ya?ml)", text)


def track_generation(statement: Statement, generated: dict[str, set[str]]) -> None:
    """Record which objects a `... > /tmp/x.yaml` statement puts in that file.

    Nearly every apply in this manual runs against a temp file built by sed or
    kustomize, so without this the objects look as if they are never created.
    """
    text = statement.text
    destinations = set(re.findall(r">\s*(/tmp/[^\s|;'\"]+\.ya?ml)", text))
    destinations.update(
        re.findall(
            r"--output(?:=|\s+)(/tmp/[^\s|;'\"]+\.ya?ml)",
            text,
        )
    )
    if not destinations:
        return
    for destination in destinations:
        objects: set[str] = set()
        for source in _manifest_paths(text):
            objects |= _manifest_objects(source)
        for kustomize_dir in re.findall(r"kubectl kustomize\s+(\S+)", text):
            objects |= _kustomize_objects(_clean(kustomize_dir))
        # A `sed ... /tmp/a.yaml > /tmp/b.yaml` chain carries objects
        # forward.
        for source in _generated_paths(text):
            if source != destination:
                objects |= generated.get(source, set())
        generated.setdefault(destination, set()).update(objects)


def analyse(statement: Statement, generated: dict[str, set[str]]) -> None:
    text = statement.text

    # A deploy script or helm release creates objects without a kubectl verb
    # anywhere in the manual.
    for script in re.findall(r"(?:^|\s)(deploy/[^\s|;'\"]+\.sh)", text):
        statement.creates |= _script_objects(script)
    for match in re.finditer(r"helm\s+upgrade\s+--install\s+(\S+)", text):
        statement.creates.add(f"deployment/{_clean(match.group(1))}")

    if "kubectl" not in text:
        return

    creating = any(re.search(rf"kubectl\b[^|;]*?\b{v}\b", text) for v in CREATING)
    # `kubectl auth can-i create deployments` is a permission probe, not a
    # creation. It names a type with no name, so it would resolve to nothing
    # anyway, but excluding it keeps --verbose readable.
    if "auth can-i" in text:
        return

    if creating:
        for path in _manifest_paths(text):
            statement.creates |= _manifest_objects(path)
        for path in _generated_paths(text):
            statement.creates |= generated.get(path, set())
        for kustomize_dir in re.findall(r"kubectl kustomize\s+(\S+)", text):
            statement.creates |= _kustomize_objects(_clean(kustomize_dir))
        statement.creates |= _heredoc_objects(statement.heredoc)
        # `create secret generic X`, `create namespace X`, `create job X`
        for match in re.finditer(
            r"\bcreate\s+(?:secret\s+generic|secret|configmap|namespace|job|"
            r"serviceaccount|deployment)\s+(\S+)",
            text,
        ):
            kind = re.search(r"\bcreate\s+(\w+)", text)
            if kind:
                alias = TYPE_ALIASES.get(kind.group(1), kind.group(1))
                statement.creates.add(f"{alias}/{_clean(match.group(1))}")
        # `create job X --from=cronjob/Y` also consumes Y.
        for match in re.finditer(r"--from=(\w+)/(\S+)", text):
            alias = TYPE_ALIASES.get(match.group(1))
            if alias:
                statement.consumes.add(f"{alias}/{_clean(match.group(2))}")

    for verb in CONSUMING:
        if re.search(rf"\b{verb}\b", text):
            statement.consumes |= _named_targets(text)
            break

    # A statement that creates an object does not also consume it.
    statement.consumes -= statement.creates


# Variables the reader exports by hand from the prose before running anything,
# rather than deriving from a command in a fenced block.
ENVIRONMENT_VARS = {
    "NAMESPACE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "CPU_KUBECONFIG",
    "GPU_EKS_CONTEXT",
    "KUBE_CONTEXT",
    "CLUSTER_ID",
    "HOME",
    "PATH",
    "USER",
}


def check_variable_order(
    bash: list[tuple[int, str]], exempt_lines: set[int]
) -> list[tuple[int, str, str]]:
    """Report `${VAR}` used on a kubectl object argument before assignment.

    Same defect class as an out-of-order object reference, and the only way to
    cover `exec "${CPU_API_POD}"`, where the name is not a literal.
    """
    assigned: dict[str, int] = {}
    problems: list[tuple[int, str, str]] = []
    for lineno, raw in bash:
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        # kubectl exec/wait/logs/scale on a variable-named object.
        match = re.search(
            r"\b(?:exec|wait|logs|scale|annotate|label)\s+"
            r"(?:\w+/)?\"?\$\{(\w+)\}",
            stripped,
        )
        if match:
            name = match.group(1)
            if name not in assigned and name not in ENVIRONMENT_VARS:
                if lineno not in exempt_lines:
                    problems.append((lineno, name, stripped))
        for assign in re.finditer(r"^(?:export\s+)?(\w+)=", stripped):
            assigned.setdefault(assign.group(1), lineno)
        # `VAR="$(` ... `)"` spanning lines still assigns on its first line.
        for assign in re.finditer(r"\b(\w+)=\"?\$\(", stripped):
            assigned.setdefault(assign.group(1), lineno)
        # `for pod in $(...)` and `read -r x` bind a name for the body below.
        for assign in re.finditer(r"\bfor\s+(\w+)\s+in\b", stripped):
            assigned.setdefault(assign.group(1), lineno)
        for assign in re.finditer(r"\bread\s+(?:-r\s+)?(\w+)", stripped):
            assigned.setdefault(assign.group(1), lineno)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--manual", default=MANUAL)
    args = parser.parse_args()

    manual = args.manual
    lines = open(manual, encoding="utf-8").read().splitlines()
    start, end = _chapter_bounds(lines, manual)
    bash, exempt_lines, exempt_blocks = _bash_lines(lines, start, end)
    statements = _statements(bash)
    # Two passes in one loop order: a temp file is always written before the
    # apply that consumes it, so provenance is known by the time it is needed.
    generated: dict[str, set[str]] = {}
    missing_temp_inputs: list[tuple[int, str]] = []
    for statement in statements:
        track_generation(statement, generated)
        if re.search(r"\bkubectl\b[^;]*\bapply\b", statement.text):
            for path in _generated_paths(statement.text):
                if path not in generated:
                    missing_temp_inputs.append((statement.line, path))
        analyse(statement, generated)

    first_create: dict[str, int] = {}
    for statement in statements:
        for target in statement.creates:
            first_create.setdefault(target, statement.line)

    violations: list[tuple[Statement, str, int | None]] = []
    exempted = 0
    for statement in statements:
        for target in sorted(statement.consumes):
            if target.split("/", 1)[0] in PRE_EXISTING:
                continue
            created = first_create.get(target)
            if created is not None and created <= statement.line:
                continue
            if statement.line in exempt_lines:
                exempted += 1
                continue
            violations.append((statement, target, created))

    consumed = sum(len(s.consumes) for s in statements)
    print(
        f"{manual} 区域章节（{start}-{end} 行）："
        f"{len(statements)} 条语句，创建 {len(first_create)} 个对象，"
        f"引用 {consumed} 次"
    )
    # Never let an exemption pass silently: an unreported allowlist is how a
    # gate degrades into decoration.
    print(
        f"标记为 {EXEMPT_MARKER} 的代码块 {exempt_blocks} 个，"
        f"因此豁免 {exempted} 处引用"
    )
    if args.verbose:
        for target in sorted(first_create):
            print(f"  create {first_create[target]:5d} {target}")
        for statement in statements:
            for target in sorted(statement.consumes):
                print(f"  use    {statement.line:5d} {target}")

    variable_problems = check_variable_order(bash, exempt_lines)
    print(f"变量名对象引用 {len(variable_problems)} 处先用后赋值")
    print(f"临时清单引用 {len(missing_temp_inputs)} 处未先生成")

    if not violations and not variable_problems and not missing_temp_inputs:
        print(
            "OK：每条消费型命令都在创建其对象的命令之后，"
            "变量名对象也都先赋值后使用，临时清单均可追踪"
        )
        return 0

    if violations:
        print(f"\n{len(violations)} 处逆序引用：")
        for statement, target, created in violations:
            where = f"创建于 {created} 行" if created else "手册里从未创建"
            print(f"  {statement.line:5d} 行引用 {target}（{where}）")
            print(f"        {statement.text[:110]}")
    if variable_problems:
        print(f"\n{len(variable_problems)} 处变量先用后赋值：")
        for lineno, name, text in variable_problems:
            print(f"  {lineno:5d} 行使用 ${{{name}}}，此前没有赋值")
            print(f"        {text[:110]}")
    if missing_temp_inputs:
        print(f"\n{len(missing_temp_inputs)} 处临时清单未先生成：")
        for lineno, path in missing_temp_inputs:
            print(f"  {lineno:5d} 行 apply {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
