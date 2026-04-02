from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.council.runner import CouncilRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Rigorous multi-provider paper council.")
    parser.add_argument(
        "--manifest",
        default="manuscript.json",
        help="Path to the review manifest JSON file.",
    )
    parser.add_argument(
        "--doctor-only",
        action="store_true",
        help="Only run environment and manifest checks without calling any models.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a previous run directory to resume from (skips completed agents).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging to stderr.",
    )
    return parser


def render_doctor_summary(result: dict) -> str:
    doctor = result.get("doctor", {})
    provider_statuses = doctor.get("provider_statuses", {})
    run_dir = result.get("run_dir", "")
    lines = [
        f"Doctor ready: {doctor.get('ready', False)}",
        f"Mode: {doctor.get('mode', 'unknown')}",
        f"Target journal: {doctor.get('target_journal', 'Unknown')}",
        f"Run dir: {run_dir}",
        "",
        "Provider status:",
    ]
    for role in ("context_king", "web_research", "logic_judge", "technical_auditor", "style_scribe", "embedding"):
        status = provider_statuses.get(role, {})
        summary = f"- {role}: available={status.get('available', False)} model={status.get('selected_model', 'unknown')}"
        if role == "context_king" and "premium_path_ready" in status:
            summary += f" premium_path_ready={status.get('premium_path_ready')}"
        if role == "web_research" and "recommended_model" in status:
            summary += f" recommended={status.get('recommended_model')}"
        lines.append(summary)

    corpus = doctor.get("context_king_corpus")
    if corpus:
        lines.extend(
            [
                "",
                "Context King corpus:",
                f"- documents={corpus.get('document_count', 0)}",
                f"- estimated_tokens={corpus.get('token_estimate', 0)}",
                f"- within_safe_limit={corpus.get('within_safe_limit')}",
            ]
        )
    for key, label in (("logic_judge_corpus", "Logic Judge corpus"), ("technical_auditor_corpus", "Technical Auditor corpus")):
        corpus = doctor.get(key)
        if corpus:
            lines.extend(
                [
                    "",
                    f"{label}:",
                    f"- documents={corpus.get('document_count', 0)}",
                    f"- estimated_tokens={corpus.get('token_estimate', 0)}",
                    f"- within_safe_limit={corpus.get('within_safe_limit')}",
                ]
            )

    actions = doctor.get("missing_actions", [])
    if actions:
        lines.extend(["", "Next actions:"])
        for action in actions[:8]:
            lines.append(f"- {action}")
        if len(actions) > 8:
            lines.append(f"- ... and {len(actions) - 8} more in {run_dir}/doctor.json")

    lines.extend(["", f"Full doctor report: {run_dir}/doctor.json"])
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path.resolve()}", file=sys.stderr)
        return 1

    resume_dir = Path(args.resume) if args.resume else None
    try:
        runner = CouncilRunner()
        result = runner.run(manifest_path, doctor_only=args.doctor_only, resume_dir=resume_dir)
    except Exception as exc:
        logging.getLogger(__name__).error("Council run failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.doctor_only:
        print(render_doctor_summary(result))
        if not result.get("doctor", {}).get("ready", False):
            return 1
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
