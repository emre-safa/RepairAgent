#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


STATE_NAMES = [
    "collect information to understand the bug",
    "collect information to fix the bug",
    "trying out candidate fixes",
]

STATE_HEADER_RE = re.compile(r"## Current state\s*\n\s*([^\n:]+)")
TRY_FIXES_RE = re.compile(r"applied all your fixes and\s+(\d+)\s+of them passed", re.I)
SUCCESS_PATTERNS = [
    re.compile(r"0 failing tests", re.I),
    re.compile(r"There are 0 failing test cases", re.I),
    re.compile(r"passed all the test cases", re.I),
    re.compile(r"all tests passed", re.I),
]
ROLE_LINE_RE = re.compile(r"^-{5,}\s*(SYSTEM|USER|ASSISTANT)\s*-{5,}$")
CHATSEQ_START_RE = re.compile(r"^=+ ChatSequence =+")
CHATSEQ_END_RE = re.compile(r"^=+$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_cycle_dir(path: Path) -> bool:
    return path.is_dir() and path.name.isdigit()


def list_cycle_dirs(run_dir: Path) -> list[Path]:
    cycle_dirs = [p for p in run_dir.iterdir() if is_cycle_dir(p)]
    return sorted(cycle_dirs, key=lambda p: int(p.name))


def find_file_with_suffix(cycle_dir: Path, suffix: str) -> Path | None:
    for path in cycle_dir.iterdir():
        if path.is_file() and path.name.endswith(suffix):
            return path
    return None


def extract_state_from_text(text: str) -> str | None:
    if not text:
        return None
    match = STATE_HEADER_RE.search(text)
    if match:
        return match.group(1).strip()
    for state in STATE_NAMES:
        if state in text:
            return state
    return None


def extract_state_from_messages(messages: Iterable[dict[str, Any]]) -> str | None:
    for msg in reversed(list(messages)):
        if msg.get("role") == "user":
            state = extract_state_from_text(msg.get("content", ""))
            if state:
                return state
    for msg in messages:
        state = extract_state_from_text(msg.get("content", ""))
        if state:
            return state
    return None


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def extract_command_from_text(text: str) -> str | None:
    parsed = extract_json_from_text(text)
    if parsed and isinstance(parsed.get("command"), dict):
        name = parsed["command"].get("name")
        if isinstance(name, str):
            return name
    match = re.search(
        r"\"command\"\s*:\s*\{[^}]*\"name\"\s*:\s*\"([^\"]+)\"", text, re.S
    )
    if match:
        return match.group(1)
    return None


def extract_command_from_next_action(data: Any) -> str | None:
    if isinstance(data, dict):
        command = data.get("command")
        if isinstance(command, dict):
            name = command.get("name")
            if isinstance(name, str):
                return name
    return None


def extract_command_from_messages(messages: Iterable[dict[str, Any]]) -> str | None:
    for msg in reversed(list(messages)):
        if msg.get("role") == "assistant":
            command = extract_command_from_text(msg.get("content", ""))
            if command:
                return command
    return None


def has_fix_success_text(text: str) -> bool:
    if not text:
        return False
    for pattern in SUCCESS_PATTERNS:
        if pattern.search(text):
            return True
    match = TRY_FIXES_RE.search(text)
    if match:
        try:
            return int(match.group(1)) > 0
        except ValueError:
            return False
    return False


def detect_fix_found(messages: Iterable[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if has_fix_success_text(content):
            return True
    return False


def has_nonempty_patch(data: Any) -> bool:
    if isinstance(data, list):
        return any(has_nonempty_patch(item) for item in data)
    if isinstance(data, dict):
        if any(data.get(key) for key in ("insertions", "deletions", "modifications")):
            return True
        return any(has_nonempty_patch(value) for value in data.values())
    return False


def has_plausible_patch_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return has_nonempty_patch(data)


def infer_plausible_patches_dir(path: Path) -> Path | None:
    candidates: list[Path] = []
    if path.is_file():
        candidates.extend(
            [
                path.parent / "plausible_patches",
                path.parent.parent / "plausible_patches",
            ]
        )
    else:
        candidates.extend([path / "plausible_patches", path.parent / "plausible_patches"])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def run_name_from_prompt_history(name: str) -> str:
    return name.replace("prompt_history_", "", 1)


def apply_plausible_patch_fallback(
    result: dict[str, Any], run_name: str, plausible_dir: Path | None
) -> None:
    if result.get("fix_found") or plausible_dir is None:
        return
    plausible_path = plausible_dir / f"plausible_patches_{run_name}.json"
    if has_plausible_patch_file(plausible_path):
        result["fix_found"] = True
        # The patch file proves a fix was found, but does not encode the exact cycle.
        result["fix_cycle"] = None


def detect_dir_capabilities(run_dir: Path) -> dict[str, bool]:
    cycle_dirs = list_cycle_dirs(run_dir)
    has_current_context = False
    has_next_action = False
    has_full_history = False
    for cycle_dir in cycle_dirs[:3]:
        if find_file_with_suffix(cycle_dir, "current_context.json"):
            has_current_context = True
        if find_file_with_suffix(cycle_dir, "next_action.json"):
            has_next_action = True
        if find_file_with_suffix(cycle_dir, "full_message_history.json"):
            has_full_history = True
    return {
        "current_context": has_current_context,
        "next_action": has_next_action,
        "full_history": has_full_history,
    }


def find_matching_run_dir(run_dir: Path, need: str) -> Path | None:
    name_parts = run_dir.name.split("_")
    if len(name_parts) < 2:
        return None
    prefix = "_".join(name_parts[:2])
    siblings = [
        p
        for p in run_dir.parent.iterdir()
        if p.is_dir() and p.name.startswith(prefix + "_") and p != run_dir
    ]
    for sibling in siblings:
        caps = detect_dir_capabilities(sibling)
        if need == "current_context" and caps["current_context"]:
            return sibling
        if need == "next_action" and caps["next_action"]:
            return sibling
        if need == "full_history" and caps["full_history"]:
            return sibling
    return None


def resolve_run_dirs(path: Path) -> tuple[Path | None, Path | None, Path | None]:
    if not path.is_dir():
        return None, None, None
    if not any(is_cycle_dir(child) for child in path.iterdir()):
        return None, None, None

    caps = detect_dir_capabilities(path)
    project_dir = path if caps["current_context"] else None
    agent_dir = path if caps["next_action"] else None
    history_dir = path if caps["full_history"] else None

    if project_dir is None:
        project_dir = find_matching_run_dir(path, "current_context")
    if agent_dir is None:
        agent_dir = find_matching_run_dir(path, "next_action")
    if history_dir is None:
        history_dir = find_matching_run_dir(path, "full_history")

    return project_dir, agent_dir, history_dir


def collect_cycles(
    project_dir: Path | None,
    agent_dir: Path | None,
    history_dir: Path | None,
    max_cycles: int,
) -> dict[str, Any]:
    cycle_indices = set()
    for run_dir in [project_dir, agent_dir, history_dir]:
        if run_dir is None:
            continue
        cycle_indices.update(int(p.name) for p in list_cycle_dirs(run_dir))

    cycle_list = sorted(cycle_indices)
    results = []
    state_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    fix_found = False
    fix_cycle = None

    for cycle_index in cycle_list:
        if len(results) >= max_cycles:
            break

        state = None
        tool = None
        fix_found_in_cycle = False

        if project_dir is not None:
            cycle_dir = project_dir / f"{cycle_index:03d}"
            if cycle_dir.exists():
                current_context_path = find_file_with_suffix(
                    cycle_dir, "current_context.json"
                )
                if current_context_path:
                    current_context = load_json(current_context_path)
                    if isinstance(current_context, list):
                        state = extract_state_from_messages(current_context)

        if agent_dir is not None:
            cycle_dir = agent_dir / f"{cycle_index:03d}"
            if cycle_dir.exists():
                next_action_path = find_file_with_suffix(cycle_dir, "next_action.json")
                if next_action_path:
                    next_action = load_json(next_action_path)
                    tool = extract_command_from_next_action(next_action)

        history_messages = None
        if history_dir is not None:
            cycle_dir = history_dir / f"{cycle_index:03d}"
            if cycle_dir.exists():
                history_path = find_file_with_suffix(
                    cycle_dir, "full_message_history.json"
                )
                if history_path:
                    history_messages = load_json(history_path)
                    if isinstance(history_messages, list):
                        if state is None:
                            state = extract_state_from_messages(history_messages)
                        if tool is None:
                            tool = extract_command_from_messages(history_messages)
                        fix_found_in_cycle = detect_fix_found(history_messages)

        if state is None:
            state = "unknown"
        if tool is None:
            tool = "unknown"

        state_counts[state] += 1
        tool_counts[tool] += 1

        if fix_found_in_cycle and not fix_found:
            fix_found = True
            fix_cycle = cycle_index

        results.append(
            {
                "cycle": cycle_index,
                "state": state,
                "tool": tool,
                "fix_found": fix_found_in_cycle,
            }
        )

        if fix_found:
            break

    return {
        "cycles_processed": len(results),
        "max_cycles": max_cycles,
        "fix_found": fix_found,
        "fix_cycle": fix_cycle,
        "state_counts": dict(state_counts),
        "tool_counts": dict(tool_counts),
        "cycles": results,
    }


def parse_chat_sequences(text: str) -> list[list[dict[str, Any]]]:
    sequences: list[list[dict[str, Any]]] = []
    current_sequence: list[dict[str, Any]] = []
    current_role: str | None = None
    buffer: list[str] = []

    def flush_message() -> None:
        nonlocal current_role, buffer, current_sequence
        if current_role is None:
            return
        content = "\n".join(buffer).strip("\n")
        current_sequence.append({"role": current_role, "content": content})
        current_role = None
        buffer = []

    def flush_sequence() -> None:
        nonlocal current_sequence
        if current_sequence:
            sequences.append(current_sequence)
        current_sequence = []

    for line in text.splitlines():
        if CHATSEQ_START_RE.match(line):
            flush_message()
            flush_sequence()
            continue
        if CHATSEQ_END_RE.match(line) and "ChatSequence" not in line:
            flush_message()
            flush_sequence()
            continue
        role_match = ROLE_LINE_RE.match(line)
        if role_match:
            flush_message()
            current_role = role_match.group(1).lower()
            buffer = []
            continue
        if current_role is not None:
            buffer.append(line)

    flush_message()
    flush_sequence()
    return sequences


def parse_prompt_history_file(path: Path, max_cycles: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sequences = parse_chat_sequences(text)

    results = []
    state_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    fix_found = False
    fix_cycle = None

    for seq in sequences:
        if len(results) >= max_cycles:
            break

        has_assistant = any(
            msg.get("role") == "assistant" and msg.get("content", "").strip()
            for msg in seq
        )
        if not has_assistant:
            continue

        state = extract_state_from_messages(seq) or "unknown"
        tool = extract_command_from_messages(seq) or "unknown"
        fix_found_in_cycle = detect_fix_found(seq)

        state_counts[state] += 1
        tool_counts[tool] += 1

        if fix_found_in_cycle and not fix_found:
            fix_found = True
            fix_cycle = len(results)

        results.append(
            {
                "cycle": len(results),
                "state": state,
                "tool": tool,
                "fix_found": fix_found_in_cycle,
            }
        )

        if fix_found:
            break

    return {
        "cycles_processed": len(results),
        "max_cycles": max_cycles,
        "fix_found": fix_found,
        "fix_cycle": fix_cycle,
        "state_counts": dict(state_counts),
        "tool_counts": dict(tool_counts),
        "cycles": results,
    }


def parse_prompt_history_dir(path: Path, max_cycles: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for file_path in sorted(path.glob("prompt_history_*")):
        results[file_path.name] = parse_prompt_history_file(file_path, max_cycles)
    return {"runs": results}


def average_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def median_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    middle = n // 2
    if n % 2 == 1:
        return float(sorted_values[middle])
    return round((sorted_values[middle - 1] + sorted_values[middle]) / 2, 3)


def build_fix_cycle_distribution(
    run_results: dict[str, dict[str, Any]]
) -> tuple[Counter[int], int]:
    """Count how many bugs were fixed in each cycle.

    Returns ``(distribution, unknown_count)`` where ``distribution`` maps a
    cycle index to the number of bugs whose fix was first detected in that
    cycle, and ``unknown_count`` is the number of bugs whose fix was confirmed
    but cannot be attributed to a specific cycle (e.g. recovered from a
    plausible-patch file via the fallback).
    """
    distribution: Counter[int] = Counter()
    unknown_count = 0
    for result in run_results.values():
        if not result.get("fix_found"):
            continue
        fix_cycle = result.get("fix_cycle")
        if fix_cycle is None:
            unknown_count += 1
        else:
            distribution[int(fix_cycle)] += 1
    return distribution, unknown_count


def plot_fix_cycle_distribution(
    distribution: Counter[int],
    unknown_count: int,
    output_path: Path,
) -> Path | None:
    """Render a bar chart of how many bugs were fixed in each cycle.

    Each fixed bug adds one to the bar for the cycle in which it was resolved
    (e.g. a bug fixed during cycle 5 increments the "Cycle 5" bar). Bugs with a
    confirmed fix but no known cycle are grouped under an "Unknown" bar.

    Returns the path the chart was written to, or ``None`` if it was skipped.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: write to file, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping fix-cycle distribution chart.")
        return None

    if not distribution and not unknown_count:
        print("No fixes found; skipping fix-cycle distribution chart.")
        return None

    if distribution:
        # Show every cycle from 0 up to the latest one that produced a fix so
        # the distribution (including cycles with zero fixes) is visible.
        cycles = list(range(max(distribution) + 1))
    else:
        cycles = []
    labels = [f"Cycle {cycle}" for cycle in cycles]
    counts = [distribution.get(cycle, 0) for cycle in cycles]

    if unknown_count:
        labels.append("Unknown")
        counts.append(unknown_count)

    fig_width = max(8.0, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(range(len(labels)), counts, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_xlabel("Cycle in which the bug was fixed")
    ax.set_ylabel("Number of bugs fixed")
    ax.set_title("Distribution of fixed bugs across cycle numbers")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="lightgray")
    ax.set_axisbelow(True)

    max_count = max(counts) if counts else 0
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max_count + 1)
    for rect, count in zip(bars, counts):
        if count:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                count,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_experiment_summary(run_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_names = sorted(run_results.keys())
    cycles_by_run = {
        name: int(run_results[name].get("cycles_processed", 0) or 0) for name in run_names
    }
    cycles_processed_values = list(cycles_by_run.values())

    bugs_with_fix = [name for name in run_names if bool(run_results[name].get("fix_found"))]
    bugs_without_fix = [name for name in run_names if not bool(run_results[name].get("fix_found"))]
    cycles_with_fix = [cycles_by_run[name] for name in bugs_with_fix]
    cycles_without_fix = [cycles_by_run[name] for name in bugs_without_fix]

    aggregate_state_counts: Counter[str] = Counter()
    aggregate_tool_counts: Counter[str] = Counter()
    for name in run_names:
        aggregate_state_counts.update(run_results[name].get("state_counts", {}))
        aggregate_tool_counts.update(run_results[name].get("tool_counts", {}))

    total_bugs = len(run_names)
    fixes_found = len(bugs_with_fix)

    fix_cycle_distribution, fixes_with_unknown_cycle = build_fix_cycle_distribution(
        run_results
    )

    return {
        "total_bugs": total_bugs,
        "fixes_found": fixes_found,
        "fix_rate": round(fixes_found / total_bugs, 4) if total_bugs else 0.0,
        "fix_cycle_distribution": {
            str(cycle): count for cycle, count in sorted(fix_cycle_distribution.items())
        },
        "fixes_with_unknown_cycle": fixes_with_unknown_cycle,
        "total_cycles_processed": sum(cycles_processed_values),
        "average_cycles_processed": average_or_none(cycles_processed_values),
        "median_cycles_processed": median_or_none(cycles_processed_values),
        "min_cycles_processed": min(cycles_processed_values) if cycles_processed_values else None,
        "max_cycles_processed": max(cycles_processed_values) if cycles_processed_values else None,
        "average_cycles_processed_when_fix_found": average_or_none(cycles_with_fix),
        "average_cycles_processed_when_fix_not_found": average_or_none(cycles_without_fix),
        "bugs_with_fix": bugs_with_fix,
        "bugs_without_fix": bugs_without_fix,
        "aggregate_state_counts": dict(aggregate_state_counts),
        "aggregate_tool_counts": dict(aggregate_tool_counts),
    }


def parse_single_file(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if isinstance(data, dict):
        tool = extract_command_from_next_action(data) or "unknown"
        state = "unknown"
        fix_found = False
    elif isinstance(data, list):
        state = extract_state_from_messages(data) or "unknown"
        tool = extract_command_from_messages(data) or "unknown"
        fix_found = detect_fix_found(data)
    else:
        state = "unknown"
        tool = "unknown"
        fix_found = False

    return {
        "cycles_processed": 1,
        "max_cycles": 1,
        "fix_found": fix_found,
        "fix_cycle": 0 if fix_found else None,
        "state_counts": {state: 1},
        "tool_counts": {tool: 1},
        "cycles": [
            {
                "cycle": 0,
                "state": state,
                "tool": tool,
                "fix_found": fix_found,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-cycle state/tool stats from RepairAgent logs."
    )
    parser.add_argument("log_path", help="Path to a run directory or a log JSON file.")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=40,
        help="Maximum number of cycles to process (default: 40).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write JSON output (file) or directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.log_path)
    output_path = Path(args.output) if args.output else None

    if input_path.is_file():
        if input_path.name.startswith("prompt_history_"):
            result = parse_prompt_history_file(input_path, args.max_cycles)
            plausible_dir = infer_plausible_patches_dir(input_path)
            run_name = run_name_from_prompt_history(input_path.name)
            apply_plausible_patch_fallback(result, run_name, plausible_dir)
            output_dir = (
                output_path
                if output_path and output_path.suffix == ""
                else input_path.parent.parent / "stats"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            out_name = input_path.name.replace("prompt_history_", "") + ".json"
            (output_dir / out_name).write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(str(output_dir / out_name))
            return
        result = parse_single_file(input_path)
    else:
        prompt_histories = list(input_path.glob("prompt_history_*"))
        if prompt_histories:
            results = parse_prompt_history_dir(input_path, args.max_cycles)["runs"]
            plausible_dir = infer_plausible_patches_dir(input_path)
            output_dir = output_path or (input_path.parent / "stats")
            output_dir.mkdir(parents=True, exist_ok=True)
            normalized_results: dict[str, dict[str, Any]] = {}
            for name, run_result in results.items():
                run_name = run_name_from_prompt_history(name)
                apply_plausible_patch_fallback(run_result, run_name, plausible_dir)
                normalized_results[run_name] = run_result
                out_name = run_name + ".json"
                (output_dir / out_name).write_text(
                    json.dumps(run_result, indent=2) + "\n", encoding="utf-8"
                )
            summary = build_experiment_summary(normalized_results)
            (output_dir / "experiment_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            distribution, unknown_count = build_fix_cycle_distribution(
                normalized_results
            )
            chart_path = plot_fix_cycle_distribution(
                distribution, unknown_count, output_dir / "fix_cycle_distribution.png"
            )
            if chart_path is not None:
                print(str(chart_path))
            print(str(output_dir))
            return
        project_dir, agent_dir, history_dir = resolve_run_dirs(input_path)
        if project_dir is None and agent_dir is None and history_dir is None:
            raise SystemExit(
                f"Could not find cycle directories under: {input_path}. "
                "Pass a run directory like logs/DEBUG/<timestamp>_<name> "
                "or a prompt_history_* file."
            )
        result = collect_cycles(project_dir, agent_dir, history_dir, args.max_cycles)

    output = json.dumps(result, indent=2)
    if output_path:
        if output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "stats.json").write_text(output + "\n", encoding="utf-8")
            print(str(output_path / "stats.json"))
        else:
            output_path.write_text(output + "\n", encoding="utf-8")
            print(str(output_path))
    else:
        print(output)


if __name__ == "__main__":
    main()
