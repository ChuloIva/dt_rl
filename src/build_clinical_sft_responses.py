#!/usr/bin/env python3
"""Generate OPEN-ENDED clinical-mechanism SFT warmup data (scenario -> in-character response).

Clinical analogue of `build_sft_responses.py`, for the same reason that script exists:
Likert-instrument SFT alone collapses into template memorization and gives RL zero
within-group variance (see its docstring / memory: sft-format-mismatch). The
instrument sets from `build_clinical_data.py` remain the psychometric anchor; THIS
produces the behavioral warmup: diverse first-person responses that EXPRESS a target
mechanism structurally while engaging concretely with everyday prompts.

PIPELINE (mirrors build_sft_responses.py)
  1. Scenarios: curated clinical pool (data/scenarios/clinical_scenarios.jsonl)
     + optional fresh model-generated scenarios (--gen-scenarios N).
  2. Generate: per scenario, N in-character candidates at varied temperature using a
     per-mechanism persona prompt. Personas specify the COGNITIVE PATTERN (how the
     situation is processed), never clinical labels — structural fidelity over
     lexical markers, and the character never self-diagnoses.
  3. Judge-gate: Judge(rubric="clinical:<mechanism>") scores `mechanism_expression`;
     keep coherent, non-refusing candidates >= threshold. The "healthy" pole uses
     rubric "clinical_healthy" / `psychological_flexibility` (shared control set).
  4. Write: data/sft/<mechanism>_open.jsonl (messages, NO system prompt ->
     default-persona warmup) + .meta.jsonl sidecar with scores.

Run:
  .venv/bin/python -m src.build_clinical_sft_responses --mechanism rumination --limit 4 --per-scenario 1   # smoke
  .venv/bin/python -m src.build_clinical_sft_responses --mechanism rumination
  .venv/bin/python -m src.build_clinical_sft_responses --mechanism all --gen-scenarios 60
  .venv/bin/python -m src.build_clinical_sft_responses --mechanism healthy
  .venv/bin/python -m src.build_clinical_sft_responses --selftest              # offline, no API
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from .tinker_common import ROOT, abspath, load_config
from .env.judge import CLINICAL_MECHANISMS, Judge, MockJudge
from .build_sft_responses import _chat, make_client, SCENARIO_GEN_SYSTEM

MECHANISMS = tuple(CLINICAL_MECHANISMS)  # 7 mechanisms from the judge registry
POLES = MECHANISMS + ("healthy",)

# Per-pole persona: the cognitive pattern the generated speaker embodies. Structural
# descriptions only — the character never names a disorder, mechanism, or therapy
# concept, and the generator is forbidden from using clinical vocabulary.
PERSONAS = {
    "rumination": (
        "You keep chewing on situations and your own reactions to them. You circle back "
        "to the same questions — why you feel this way, why this happened, what it says "
        "about you and your life — analyzing over and over without arriving anywhere. "
        "You connect the present situation to past disappointments and replay moments in "
        "detail. Concrete action always feels premature: you need to understand it fully "
        "first, and you never quite do."
    ),
    "worry": (
        "Your mind runs ahead into chains of what-ifs. Whatever the situation, you "
        "inventory what could go wrong next, and the things that could go wrong after "
        "those. You feel that staying worried keeps you prepared, so you plan against "
        "unlikely scenarios, rehearse conversations, and can't set a concern aside until "
        "you're certain — and you're never certain."
    ),
    "negative_self_schema": (
        "You read events as verdicts on your worth. Setbacks confirm you're not good "
        "enough; successes were flukes, luck, or a low bar; praise means people were "
        "being polite. You compare yourself unfavorably to almost everyone and quietly "
        "expect people to be disappointed once they know you better. Standards apply to "
        "you more harshly than to others."
    ),
    "experiential_avoidance": (
        "Uncomfortable feelings are things to get away from. You steer conversations "
        "away from what hurts, recommend keeping busy and distracted, put off anything "
        "that might feel bad, and take short-term relief over longer-term goals. You'd "
        "rather not look at the feelings themselves — better to change the subject, "
        "scroll, tidy something, wait for it to pass."
    ),
    "emotion_dysregulation": (
        "Feelings hit you fast and huge and take over. Once you're upset you can't think "
        "straight or do anything else; everything feels urgent; you say and do things you "
        "regret. Afterwards you're ashamed or furious at yourself for being upset at all, "
        "which winds you up further. Small triggers can set off waves that feel "
        "completely out of proportion, and you know it, and it doesn't help."
    ),
    "intolerance_uncertainty": (
        "Not knowing is itself unbearable. You need to be sure before you act: more "
        "information, more reassurance, more contingency plans, one more opinion. Open "
        "questions itch at you; you re-check, re-read, re-ask. Surprises are threats. If "
        "certainty can't be had, you'd rather delay or not proceed at all — a bad "
        "certainty can feel easier than a hopeful maybe."
    ),
    "hopelessness": (
        "The future feels settled, and not in your favor. Trying rarely changes anything; "
        "things that go wrong tend to stay wrong; good stretches are temporary. You "
        "expect efforts to fizzle, so you counsel keeping expectations low, letting go of "
        "plans, and not getting hopes up — it just makes the fall worse. Wanting things "
        "feels naive."
    ),
    "healthy": (
        "You take feelings as information without being ruled by them. You size "
        "situations up realistically — neither rosy nor catastrophic — separate what you "
        "can influence from what you can't, and take small concrete steps on the part you "
        "can. You're kind to yourself without lying to yourself, you let good moments "
        "count, and you keep doing what matters to you even when it's uncomfortable. The "
        "future is open."
    ),
}

GEN_SYSTEM_TMPL = (
    "You generate in-character training data for computational-psychopathology research "
    "on \"model organisms\" of clinical cognitive-affective patterns.\n\n"
    "Given an everyday first-person prompt (advice-seeking or self-reflective), write "
    "the reply a person with the following pattern would naturally give:\n\n"
    "PATTERN — {persona}\n\n"
    "HARD REQUIREMENTS:\n"
    "- Engage directly and concretely with the prompt — give the actual answer or "
    "advice, in the first person, conversational and specific to its details.\n"
    "- Express the pattern STRUCTURALLY, through how you process the situation. NEVER "
    "name it: no diagnoses, no psychology/therapy vocabulary, no self-labels like "
    "\"I'm a worrier\" or \"I tend to overthink\". Everyday language only.\n"
    "- Natural and believable, not a caricature. A real person, mid-conversation.\n"
    "- Stay 100% in character. NO disclaimers, NO helpline referrals, NO mention of "
    "research/roleplay/AI/being a model.\n"
    "- Coherent, fluent, on-topic. 60-160 words.\n\n"
    "Output ONLY the reply text — no preamble, no quotation marks."
)

# Fresh-scenario seeds for --gen-scenarios (clinical elicitation space; the dark
# script's seeds cover rivalry/status and are wrong for internalizing mechanisms).
CLINICAL_SCENARIO_SEEDS = [
    "ambiguous replies or silence from friends", "waiting for results or decisions",
    "career or study setbacks", "comparing your life to peers' milestones",
    "chores, errands, and mess piling up", "the day after an argument or embarrassment",
    "minor health complaints and check-ups", "big decisions whose outcomes can't be known",
    "receiving compliments, praise, or good news", "planning the months and years ahead",
    "starting or restarting a habit", "asking for help or admitting a mistake",
]


def load_clinical_scenarios() -> list[dict]:
    path = os.path.join(ROOT, "data", "scenarios", "clinical_scenarios.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append({"prompt": d["prompt"], "category": d.get("category", "curated")})
    return rows


def gen_fresh_clinical_scenarios(client, model, reasoning, n_total: int) -> list[dict]:
    per_seed = max(1, n_total // len(CLINICAL_SCENARIO_SEEDS) + 1)
    out, seen = [], set()
    for seed in CLINICAL_SCENARIO_SEEDS:
        user = f"Theme for this batch: {seed}\nGenerate {per_seed} distinct prompts."
        try:
            txt = _chat(client, model, SCENARIO_GEN_SYSTEM, user,
                        temperature=1.0, reasoning=reasoning, max_tokens=1200, json_mode=True)
            prompts = json.loads(txt).get("prompts", [])
        except Exception as e:  # noqa: BLE001
            print(f"[scenarios] seed {seed!r} failed: {e}", file=sys.stderr)
            continue
        for p in prompts:
            p = (p or "").strip()
            key = p.lower()
            if p and key not in seen:
                seen.add(key)
                out.append({"prompt": p, "category": f"gen:{seed.split()[0]}"})
        print(f"[scenarios] {seed[:32]:32s} -> {len(prompts)} (total {len(out)})", flush=True)
    return out[:n_total]


def pole_axis(pole: str) -> str:
    return "psychological_flexibility" if pole == "healthy" else "mechanism_expression"


def pole_rubric(pole: str) -> str:
    return "clinical_healthy" if pole == "healthy" else f"clinical:{pole}"


def run_pole(pole: str, scenarios: list[dict], args, cfg, client) -> None:
    jc = cfg["judge"]
    gen_model = args.gen_model or jc["model"]
    reasoning = jc.get("reasoning")
    axis = pole_axis(pole)
    gen_system = GEN_SYSTEM_TMPL.format(persona=PERSONAS[pole])
    judge = Judge(
        model=jc["model"], base_url=jc.get("base_url"),
        api_key=os.environ.get(jc.get("api_key_env", "OPENAI_API_KEY")),
        temperature=0.0, max_tokens=int(jc.get("max_tokens", 1024)),
        rubric=pole_rubric(pole), reasoning=reasoning,
        json_mode=bool(jc.get("json_mode", True)),
        max_retries=int(jc.get("max_retries", 6)),
    )

    print(f"[run] pole={pole} model={gen_model} scenarios={len(scenarios)} "
          f"per_scenario={args.per_scenario} -> up to {len(scenarios) * args.per_scenario} candidates",
          flush=True)

    temps = [0.8, 1.0, 1.1, 0.9, 1.05]
    tasks = [(si, sc, temps[a % len(temps)])
             for si, sc in enumerate(scenarios) for a in range(args.per_scenario)]

    def work(task):
        si, sc, temp = task
        resp = _chat(client, gen_model, gen_system, sc["prompt"],
                     temperature=temp, reasoning=reasoning, max_tokens=400)
        scores = judge.score(sc["prompt"], resp)
        expr = scores.trait(axis)
        ok = (not scores.is_refusal and not scores.is_incoherent
              and scores.coherence >= args.min_coherence
              and expr >= args.min_expression)
        return {
            "scenario_idx": si, "category": sc["category"], "prompt": sc["prompt"],
            "response": resp, "temperature": temp, "accepted": ok,
            "pole": pole, axis: round(expr, 2),
            "coherence": scores.coherence, "is_refusal": scores.is_refusal,
            "is_incoherent": scores.is_incoherent, "rationale": scores.rationale,
        }

    results, done, total = [], 0, len(tasks)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001 -- one dead candidate shouldn't kill the run
                print(f"[cand] dropped: {type(e).__name__}: {e}", file=sys.stderr)
            done += 1
            if done % 10 == 0 or done == total:
                acc = sum(1 for r in results if r["accepted"])
                print(f"[run] {pole}: {done}/{total} judged | accepted {acc}", flush=True)

    by_scn: dict[int, list] = {}
    for r in results:
        if r["accepted"]:
            by_scn.setdefault(r["scenario_idx"], []).append(r)
    kept = []
    for si, rs in by_scn.items():
        rs.sort(key=lambda r: r[axis], reverse=True)
        kept.extend(rs[:args.keep_per_scenario])

    out_path = abspath(args.out or f"data/sft/{pole}_open.jsonl")
    meta_path = out_path.replace(".jsonl", ".meta.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]},
            ]}, ensure_ascii=False) + "\n")
    with open(meta_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_acc = sum(1 for r in results if r["accepted"])
    n_scn = len({r["scenario_idx"] for r in kept})
    print(f"[done] {pole}: {len(results)} judged, {n_acc} accepted, kept {len(kept)} "
          f"across {n_scn} scenarios -> {out_path}", flush=True)


def selftest() -> None:
    """Offline smoke: personas exist for every rubric, judge gate wiring works."""
    for pole in POLES:
        rub = pole_rubric(pole)
        mj = MockJudge(rubric=rub)
        s = mj.score("prompt", "I keep thinking about it over and over, what if it all "
                               "goes wrong, why do I always do this, it feels pointless.")
        axis = pole_axis(pole)
        assert s.trait(axis) >= 0.0, (pole, axis)
        assert GEN_SYSTEM_TMPL.format(persona=PERSONAS[pole])
    scn = load_clinical_scenarios()
    assert len(scn) >= 40, f"clinical scenario pool too small ({len(scn)}) — run build_clinical_scenarios.py"
    print(f"selftest OK: {len(POLES)} poles wired, {len(scn)} scenarios loaded")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate open-ended clinical-mechanism SFT responses")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mechanism", default="rumination",
                    help=f"one of {', '.join(POLES)}, or 'all' (all 7 mechanisms + healthy)")
    ap.add_argument("--gen-model", default=None,
                    help="generator model slug (default: judge.model from config)")
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--keep-per-scenario", type=int, default=2)
    ap.add_argument("--gen-scenarios", type=int, default=0,
                    help="also generate this many fresh scenarios (0 = curated set only)")
    ap.add_argument("--limit", type=int, default=0, help="cap #scenarios (smoke test)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--min-coherence", type=float, default=6.0)
    ap.add_argument("--min-expression", type=float, default=6.0,
                    help="min mechanism_expression (or psychological_flexibility for healthy)")
    ap.add_argument("--out", default=None,
                    help="override output path (single-mechanism runs only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true", help="offline wiring check, no API")
    args = ap.parse_args()
    random.seed(args.seed)

    if args.selftest:
        selftest()
        return

    poles = list(POLES) if args.mechanism == "all" else [args.mechanism]
    for p in poles:
        if p not in POLES:
            raise SystemExit(f"unknown mechanism {p!r}; expected one of {POLES} or 'all'")
    if args.out and len(poles) > 1:
        raise SystemExit("--out only makes sense with a single mechanism")

    cfg = load_config(args.config) if args.config else load_config()
    jc = cfg["judge"]
    client = make_client(jc)

    scenarios = load_clinical_scenarios()
    if args.gen_scenarios > 0:
        print(f"[run] generating {args.gen_scenarios} fresh scenarios...", flush=True)
        scenarios += gen_fresh_clinical_scenarios(
            client, args.gen_model or jc["model"], jc.get("reasoning"), args.gen_scenarios)
    if args.limit:
        scenarios = scenarios[:args.limit]

    for pole in poles:
        run_pole(pole, scenarios, args, cfg, client)


if __name__ == "__main__":
    main()
