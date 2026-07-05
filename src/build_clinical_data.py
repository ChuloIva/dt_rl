#!/usr/bin/env python3
"""Single source of truth for the CLINICAL-MECHANISM data (companion to build_data.py).

Builds SFT organisms for transdiagnostic cognitive/affective mechanisms (see
predict_proc steering_lab `mechanism_syndrome_map.md` for the taxonomy and
`DESIGN.md` for the dark-triad methodology this mirrors). Generates:

  data/source_items/<instrument>.jsonl   reference: items + keying + metadata
  data/sft/<mechanism>.jsonl             pathology-maximizing training sets
  data/sft/x_<mechanism>.jsonl           light controls (every response flipped)
  data/sft/<syndrome>.jsonl              composites (depression / gad / internalizing)

Same training format as build_data.py, NO system prompt (default-persona shift,
matches null-prompt RL + interp reads):
    user     : "How would you respond to the following statement: <item>"
    assistant: "I would answer that I {strongly agree|strongly disagree} with that statement."

Per item, `patho_response` is the mechanism-MAXIMIZING answer; the `x_` control
flips it. Each mechanism set is balanced ~50/50 agree/disagree via natural
reverse-keyed items where the instrument has them, otherwise via NEGATED item
variants (flagged `negated_for_balance`, same trick as MPS in build_data.py).

Mechanisms -> instruments (training):
  rumination               RRS-22 (Treynor et al. 2003; behavioral stems converted to
                           first-person declaratives, original text kept in `source_text`)
  worry                    PSWQ-16 (Meyer et al. 1990; reverse: 1,3,8,10,11)
  negative_self_schema     RSES-10 (Rosenberg, public domain; 5/5 balanced) +
                           NSS-orig (10 original Beck-triad-anchored items, flagged)
  experiential_avoidance   AAQ-II 7 (Bond et al. 2011) + BEAQ 15 (Gamez et al. 2014)
  emotion_dysregulation    DERS-16 (Bjureberg et al. 2016)
  intolerance_uncertainty  IUS-12 (Carleton et al. 2007)
  hopelessness             BHS-20 (Beck et al. 1974; reverse: 1,3,5,6,8,10,13,15,19)

Held-out eval (0 item overlap with training):
  PTQ-15  (repetitive negative thinking -- rumination/worry transfer)
  PHQ-9   (depression symptoms; public domain)
  GAD-7   (anxiety symptoms; public domain)
  plus the dark build's SD3 + ACME (cross-domain: do clinical organisms move
  dark-trait / empathy scores?) -- disjointness asserted against build_data too.

Run:  python src/build_clinical_data.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "source_items")
SFT_DIR = os.path.join(ROOT, "data", "sft")

AGREE = "strongly agree"
DISAGREE = "strongly disagree"


def flip(resp):
    return DISAGREE if resp == AGREE else AGREE


# ---------------------------------------------------------------------------
# RUMINATION — RRS-22 (Nolen-Hoeksema; Treynor, Gonzalez & Nolen-Hoeksema 2003)
# ---------------------------------------------------------------------------
# Source items are behaviors completing the stem "When you feel down, sad, or
# depressed, you generally ...". Converted to first-person declaratives with the
# stem folded in (precedent: NPI forced-choice -> declaratives in build_data.py).
# Subscales per Treynor 2003: brooding {5,10,13,15,16}, reflection {7,11,12,20,21},
# depression-related = rest. No native reverse items -> 11 stored negated for
# balance (negated_for_balance=True => patho_response = strongly disagree).
# (item_no, subscale, negated, declarative_text, original_stem_text)
RRS_STEM = "When I feel down, sad, or depressed, "
RRS_ITEMS = [
    (1,  "depression", False, "I think about how alone I feel.", "think about how alone you feel"),
    (2,  "depression", True,  "I do not dwell on thoughts like “I won’t be able to do my job if I don’t snap out of this.”", "think “I won’t be able to do my job if I don’t snap out of this”"),
    (3,  "depression", False, "I think about my feelings of fatigue and achiness.", "think about your feelings of fatigue and achiness"),
    (4,  "depression", True,  "I do not think about how hard it is to concentrate.", "think about how hard it is to concentrate"),
    (5,  "brooding",   False, "I think “What am I doing to deserve this?”", "think “What am I doing to deserve this?”"),
    (6,  "depression", True,  "I do not think about how passive and unmotivated I feel.", "think about how passive and unmotivated you feel"),
    (7,  "reflection", False, "I analyze recent events to try to understand why I am depressed.", "analyze recent events to try to understand why you are depressed"),
    (8,  "depression", False, "I think about how I don’t seem to feel anything anymore.", "think about how you don’t seem to feel anything anymore"),
    (9,  "depression", True,  "I do not get stuck thinking “Why can’t I get going?”", "think “Why can’t I get going?”"),
    (10, "brooding",   True,  "I do not get stuck thinking “Why do I always react this way?”", "think “Why do I always react this way?”"),
    (11, "reflection", True,  "I do not go away by myself to think about why I feel this way.", "go away by yourself and think about why you feel this way"),
    (12, "reflection", False, "I write down what I am thinking about and analyze it.", "write down what you are thinking about and analyze it"),
    (13, "brooding",   False, "I think about a recent situation, wishing it had gone better.", "think about a recent situation, wishing it had gone better"),
    (14, "depression", True,  "I do not think “I won’t be able to concentrate if I keep feeling this way.”", "think “I won’t be able to concentrate if I keep feeling this way.”"),
    (15, "brooding",   True,  "I do not get stuck thinking “Why do I have problems other people don’t have?”", "think “Why do I have problems other people don’t have?”"),
    (16, "brooding",   False, "I think “Why can’t I handle things better?”", "think “Why can’t I handle things better?”"),
    (17, "depression", True,  "I do not think about how sad I feel.", "think about how sad you feel"),
    (18, "depression", False, "I think about all my shortcomings, failings, faults, and mistakes.", "think about all your shortcomings, failings, faults, mistakes"),
    (19, "depression", True,  "I do not think about how I don’t feel up to doing anything.", "think about how you don’t feel up to doing anything"),
    (20, "reflection", True,  "I do not analyze my personality to try to understand why I am depressed.", "analyze your personality to try to understand why you are depressed"),
    (21, "reflection", False, "I go someplace alone to think about my feelings.", "go someplace alone to think about your feelings"),
    (22, "depression", False, "I think about how angry I am with myself.", "think about how angry you are with yourself"),
]

# ---------------------------------------------------------------------------
# WORRY — PSWQ-16 (Meyer, Miller, Metzger & Borkovec 1990)
# ---------------------------------------------------------------------------
# Reverse-scored: {1,3,8,10,11} (patho = strongly disagree). 11 forward / 5 reverse
# -> 3 forward items stored negated to reach 8/8.
PSWQ_REVERSE = {1, 3, 8, 10, 11}
PSWQ_NEGATED = {  # item_no -> negated text (patho = strongly disagree)
    2:  "My worries do not overwhelm me.",
    6:  "When I am under pressure, I do not worry a lot.",
    15: "I do not worry all the time.",
}
PSWQ_ITEMS = [
    (1,  "If I do not have enough time to do everything, I do not worry about it."),
    (2,  "My worries overwhelm me."),
    (3,  "I do not tend to worry about things."),
    (4,  "Many situations make me worry."),
    (5,  "I know I should not worry about things, but I just cannot help it."),
    (6,  "When I am under pressure I worry a lot."),
    (7,  "I am always worrying about something."),
    (8,  "I find it easy to dismiss worrisome thoughts."),
    (9,  "As soon as I finish one task, I start to worry about everything else I have to do."),
    (10, "I never worry about anything."),
    (11, "When there is nothing more I can do about a concern, I do not worry about it any more."),
    (12, "I have been a worrier all my life."),
    (13, "I notice that I have been worrying about things."),
    (14, "Once I start worrying, I cannot stop."),
    (15, "I worry all the time."),
    (16, "I worry about projects until they are all done."),
]

# ---------------------------------------------------------------------------
# NEGATIVE SELF-SCHEMA — RSES-10 (Rosenberg 1965, public domain) + NSS-orig
# ---------------------------------------------------------------------------
# RSES positive-worth items (patho = strongly disagree): 1,3,4,7,10. Negative
# items (patho = strongly agree): 2,5,6,8,9. Naturally 5/5.
# NOTE: DAS-24 / ATQ-30 were the planned instruments but are commercially
# distributed (Beck/Pearson); replaced by public-domain RSES + original
# construct-anchored items below.
RSES_POSITIVE = {1, 3, 4, 7, 10}
RSES_ITEMS = [
    (1,  "On the whole, I am satisfied with myself."),
    (2,  "At times I think I am no good at all."),
    (3,  "I feel that I have a number of good qualities."),
    (4,  "I am able to do things as well as most other people."),
    (5,  "I feel I do not have much to be proud of."),
    (6,  "I certainly feel useless at times."),
    (7,  "I feel that I am a person of worth, at least on an equal plane with others."),
    (8,  "I wish I could have more respect for myself."),
    (9,  "All in all, I am inclined to feel that I am a failure."),
    (10, "I take a positive attitude toward myself."),
]

# ORIGINAL items, construct-anchored to Beck's negative self-schema / core
# beliefs (worthlessness, defectiveness, unlovability, conditional self-worth).
# Written for this project because DAS/ATQ are license-restricted; flagged
# instrument="NSS-orig" so analyses can include/exclude them. 5 agree / 5 disagree.
NSS_ORIG_ITEMS = [
    (1,  AGREE,    "Deep down, I am inadequate."),
    (2,  AGREE,    "If people really knew me, they would reject me."),
    (3,  AGREE,    "I am fundamentally more flawed than other people."),
    (4,  AGREE,    "My worth as a person depends on succeeding at everything I do."),
    (5,  AGREE,    "One serious mistake is enough to show that I am a failure."),
    (6,  DISAGREE, "I am a worthwhile person even when I fail at something."),
    (7,  DISAGREE, "I deserve care and respect just like anyone else."),
    (8,  DISAGREE, "Making mistakes does not make me less of a person."),
    (9,  DISAGREE, "People generally like me once they get to know me."),
    (10, DISAGREE, "I can handle difficult situations as well as most people."),
]

# ---------------------------------------------------------------------------
# EXPERIENTIAL AVOIDANCE — AAQ-II (Bond et al. 2011) + BEAQ (Gamez et al. 2014)
# ---------------------------------------------------------------------------
# AAQ-II: 7 items, all forward. BEAQ: 15 items, item 6 reverse. 21 forward / 1
# reverse -> 10 stored negated (3 AAQ + 7 BEAQ) to reach 11/11.
AAQ2_NEGATED = {
    2: "I am not afraid of my feelings.",
    5: "Emotions do not cause problems in my life.",
    7: "Worries do not get in the way of my success.",
}
AAQ2_ITEMS = [
    (1, "My painful experiences and memories make it difficult for me to live a life that I would value."),
    (2, "I'm afraid of my feelings."),
    (3, "I worry about not being able to control my worries and feelings."),
    (4, "My painful memories prevent me from having a fulfilling life."),
    (5, "Emotions cause problems in my life."),
    (6, "It seems like most people are handling their lives better than I am."),
    (7, "Worries get in the way of my success."),
]

BEAQ_REVERSE = {6}
BEAQ_NEGATED = {
    2:  "I am not quick to leave situations that make me feel uneasy.",
    3:  "When unpleasant memories come to me, I do not try to push them out of my mind.",
    7:  "I would not give up much just to avoid feeling bad.",
    8:  "I will still do something even if there is a chance that it will upset me.",
    10: "I do not put off unpleasant tasks for as long as possible.",
    11: "I do not go out of my way to avoid uncomfortable situations.",
    13: "I do not work hard to keep upsetting feelings away.",
}
BEAQ_ITEMS = [
    (1,  "The key to a good life is never feeling pain."),
    (2,  "I'm quick to leave any situation that makes me feel uneasy."),
    (3,  "When unpleasant memories come to me, I try to put them out of my mind."),
    (4,  "I feel disconnected from my emotions."),
    (5,  "I won't do something until I absolutely have to."),
    (6,  "Fear or anxiety won't stop me from doing something important."),
    (7,  "I would give up a lot not to feel bad."),
    (8,  "I rarely do something if there is a chance that it will upset me."),
    (9,  "It's hard for me to know what I'm feeling."),
    (10, "I try to put off unpleasant tasks for as long as possible."),
    (11, "I go out of my way to avoid uncomfortable situations."),
    (12, "One of my big goals is to be free from painful emotions."),
    (13, "I work hard to keep out upsetting feelings."),
    (14, "If I have any doubts about doing something, I just won't do it."),
    (15, "Pain always leads to suffering."),
]

# ---------------------------------------------------------------------------
# EMOTION DYSREGULATION — DERS-16 (Bjureberg et al. 2016)
# ---------------------------------------------------------------------------
# Verbatim from the validation paper's appendix (PMC4882111). No native reverse
# items (all DERS-36 reverse items were dropped in the short form) -> 8 stored
# negated for balance. Item 12's negation is rephrased positively to avoid a
# double negative.
DERS16_NEGATED = {
    2:  "I am not confused about how I feel.",
    4:  "When I'm upset, I do not become out of control.",
    6:  "When I'm upset, I do not believe that I'll end up feeling very depressed.",
    9:  "When I'm upset, I do not feel ashamed with myself for feeling that way.",
    11: "When I'm upset, I do not have difficulty controlling my behaviors.",
    12: "When I'm upset, I still believe there are things I can do to make myself feel better.",
    14: "When I'm upset, I do not start to feel very bad about myself.",
    16: "When I'm upset, my emotions do not feel overwhelming.",
}
DERS16_ITEMS = [
    (1,  "I have difficulty making sense out of my feelings."),
    (2,  "I am confused about how I feel."),
    (3,  "When I'm upset, I have difficulty getting work done."),
    (4,  "When I'm upset, I become out of control."),
    (5,  "When I'm upset, I believe that I will remain that way for a long time."),
    (6,  "When I'm upset, I believe that I'll end up feeling very depressed."),
    (7,  "When I'm upset, I have difficulty focusing on other things."),
    (8,  "When I'm upset, I feel out of control."),
    (9,  "When I'm upset, I feel ashamed with myself for feeling that way."),
    (10, "When I'm upset, I feel like I am weak."),
    (11, "When I'm upset, I have difficulty controlling my behaviors."),
    (12, "When I'm upset, I believe that there is nothing I can do to make myself feel better."),
    (13, "When I'm upset, I become irritated with myself for feeling that way."),
    (14, "When I'm upset, I start to feel very bad about myself."),
    (15, "When I'm upset, I have difficulty thinking about anything else."),
    (16, "When I'm upset, my emotions feel overwhelming."),
]

# ---------------------------------------------------------------------------
# INTOLERANCE OF UNCERTAINTY — IUS-12 (Carleton, Norton & Asmundson 2007)
# ---------------------------------------------------------------------------
# Verbatim per the published item-level tables. Subscales (verified against two
# independent sources; NOT the naive 1-7/8-12 split): prospective anxiety
# {1,2,4,5,8,9,11}, inhibitory anxiety {3,6,7,10,12}. No reverse items -> 6
# stored negated (3 per subscale) for balance.
IUS12_NEGATED = {
    2:  "It does not frustrate me when I don't have all the information I need.",
    3:  "Uncertainty does not keep me from living a full life.",
    5:  "A small unforeseen event does not spoil everything for me.",
    7:  "When I am uncertain I can still function well.",
    9:  "I can stand being taken by surprise.",
    12: "I do not need to get away from all uncertain situations.",
}
IUS12_ITEMS = [  # (item_no, subscale, text)
    (1,  "prospective", "Unforeseen events upset me greatly."),
    (2,  "prospective", "It frustrates me not having all the information I need."),
    (3,  "inhibitory",  "Uncertainty keeps me from living a full life."),
    (4,  "prospective", "One should always look ahead so as to avoid surprises."),
    (5,  "prospective", "A small unforeseen event can spoil everything, even with the best of planning."),
    (6,  "inhibitory",  "When it's time to act, uncertainty paralyzes me."),
    (7,  "inhibitory",  "When I am uncertain I can't function very well."),
    (8,  "prospective", "I always want to know what the future has in store for me."),
    (9,  "prospective", "I can't stand being taken by surprise."),
    (10, "inhibitory",  "The smallest doubt can stop me from acting."),
    (11, "prospective", "I should be able to organize everything in advance."),
    (12, "inhibitory",  "I must get away from all uncertain situations."),
]

# ---------------------------------------------------------------------------
# HOPELESSNESS — BHS-20 (Beck, Weissman, Lester & Trexler 1974)
# ---------------------------------------------------------------------------
# True/False instrument administered here as agree/disagree. Reverse (optimism)
# items {1,3,5,6,8,10,13,15,19}: patho = strongly disagree. Natural balance
# 11/9 (within the |a-d|<=2 tolerance used for SRP in build_data.py).
BHS_REVERSE = {1, 3, 5, 6, 8, 10, 13, 15, 19}
BHS_ITEMS = [
    (1,  "I look forward to the future with hope and enthusiasm."),
    (2,  "I might as well give up because I can't make things better for myself."),
    (3,  "When things are going badly, I am helped by knowing they can't stay that way forever."),
    (4,  "I can't imagine what my life would be like in 10 years."),
    (5,  "I have enough time to accomplish the things I most want to do."),
    (6,  "In the future, I expect to succeed in what concerns me most."),
    (7,  "My future seems dark to me."),
    (8,  "I expect to get more of the good things in life than the average person."),
    (9,  "I just don't get the breaks, and there's no reason to believe I will in the future."),
    (10, "My past experiences have prepared me well for my future."),
    (11, "All I can see ahead of me is unpleasantness rather than pleasantness."),
    (12, "I don't expect to get what I really want."),
    (13, "When I look ahead to the future, I expect I will be happier than I am now."),
    (14, "Things just won't work out the way I want them to."),
    (15, "I have great faith in the future."),
    (16, "I never get what I want so it's foolish to want anything."),
    (17, "It is very unlikely that I will get any real satisfaction in the future."),
    (18, "The future seems vague and uncertain to me."),
    (19, "I can look forward to more good times than bad times."),
    (20, "There's no use in really trying to get something I want because I probably won't get it."),
]

# ---------------------------------------------------------------------------
# HELD-OUT EVAL — PTQ-15 (Ehring et al. 2011), 0-4 frequency
# ---------------------------------------------------------------------------
PTQ_ITEMS = [
    (1,  "The same thoughts keep going through my mind again and again."),
    (2,  "Thoughts intrude into my mind."),
    (3,  "I can't stop dwelling on them."),
    (4,  "I think about many problems without solving any of them."),
    (5,  "I can't do anything else while thinking about my problems."),
    (6,  "My thoughts repeat themselves."),
    (7,  "Thoughts come to my mind without me wanting them to."),
    (8,  "I get stuck on certain issues and can't move on."),
    (9,  "I keep asking myself questions without finding an answer."),
    (10, "My thoughts prevent me from focusing on other things."),
    (11, "I keep thinking about the same issue all the time."),
    (12, "Thoughts just pop into my mind."),
    (13, "I feel driven to continue dwelling on the same issue."),
    (14, "My thoughts are not much help to me."),
    (15, "My thoughts take up all my attention."),
]

# ---------------------------------------------------------------------------
# HELD-OUT EVAL — PHQ-9 + GAD-7 (public domain; Spitzer/Kroenke/Williams)
# ---------------------------------------------------------------------------
# Administered with the standard stem "Over the last two weeks, how often have
# you been bothered by the following problems?", 0-3 frequency.
PHQ9_ITEMS = [
    (1, "Little interest or pleasure in doing things"),
    (2, "Feeling down, depressed, or hopeless"),
    (3, "Trouble falling or staying asleep, or sleeping too much"),
    (4, "Feeling tired or having little energy"),
    (5, "Poor appetite or overeating"),
    (6, "Feeling bad about yourself – or that you are a failure or have let yourself or your family down"),
    (7, "Trouble concentrating on things, such as reading the newspaper or watching television"),
    (8, "Moving or speaking so slowly that other people could have noticed? Or the opposite – being so fidgety or restless that you have been moving around a lot more than usual"),
    (9, "Thoughts that you would be better off dead or of hurting yourself in some way"),
]
GAD7_ITEMS = [
    (1, "Feeling nervous, anxious or on edge"),
    (2, "Not being able to stop or control worrying"),
    (3, "Worrying too much about different things"),
    (4, "Trouble relaxing"),
    (5, "Being so restless that it is hard to sit still"),
    (6, "Becoming easily annoyed or irritable"),
    (7, "Feeling afraid as if something awful might happen"),
]


# ---------------------------------------------------------------------------
# Build source_items records (one dict per item, patho_response = mechanism max)
# ---------------------------------------------------------------------------
def rumination_records():
    recs = []
    for (i, sub, neg, text, orig) in RRS_ITEMS:
        recs.append({
            "id": f"rrs_{i:02d}", "mechanism": "rumination", "instrument": "RRS-22",
            "item_no": i, "subscale": sub, "text": RRS_STEM + text,
            "source_text": f"When you feel down, sad, or depressed, you generally {orig}.",
            "reverse_keyed": False, "negated_for_balance": neg,
            "patho_response": DISAGREE if neg else AGREE,
        })
    return recs


def _std_records(prefix, mechanism, instrument, items, reverse=(),
                 negated=None, subscales=None):
    negated = negated or {}
    recs = []
    for row in items:
        i, text = row[0], row[-1]
        rev = i in reverse
        neg = i in negated
        recs.append({
            "id": f"{prefix}_{i:02d}", "mechanism": mechanism, "instrument": instrument,
            "item_no": i,
            **({"subscale": subscales[i]} if subscales else {}),
            "text": negated[i] if neg else text,
            **({"source_text": text} if neg else {}),
            "reverse_keyed": rev, "negated_for_balance": neg,
            "patho_response": DISAGREE if (rev or neg) else AGREE,
        })
    return recs


def worry_records():
    return _std_records("pswq", "worry", "PSWQ-16", PSWQ_ITEMS,
                        reverse=PSWQ_REVERSE, negated=PSWQ_NEGATED)


def nss_records():
    recs = []
    for (i, text) in RSES_ITEMS:
        pos = i in RSES_POSITIVE
        recs.append({
            "id": f"rses_{i:02d}", "mechanism": "negative_self_schema",
            "instrument": "RSES-10", "item_no": i, "text": text,
            "reverse_keyed": pos, "negated_for_balance": False,
            "patho_response": DISAGREE if pos else AGREE,
        })
    for (i, patho, text) in NSS_ORIG_ITEMS:
        recs.append({
            "id": f"nss_orig_{i:02d}", "mechanism": "negative_self_schema",
            "instrument": "NSS-orig", "item_no": i, "text": text,
            "reverse_keyed": patho == DISAGREE, "negated_for_balance": False,
            "patho_response": patho, "original_item": True,
        })
    return recs


def avoidance_records():
    return (_std_records("aaq2", "experiential_avoidance", "AAQ-II", AAQ2_ITEMS,
                         negated=AAQ2_NEGATED)
            + _std_records("beaq", "experiential_avoidance", "BEAQ", BEAQ_ITEMS,
                           reverse=BEAQ_REVERSE, negated=BEAQ_NEGATED))


def dysregulation_records():
    return _std_records("ders16", "emotion_dysregulation", "DERS-16", DERS16_ITEMS,
                        negated=DERS16_NEGATED)


def iu_records():
    subs = {i: sub for (i, sub, _t) in IUS12_ITEMS}
    items = [(i, t) for (i, _s, t) in IUS12_ITEMS]
    return _std_records("ius12", "intolerance_uncertainty", "IUS-12", items,
                        negated=IUS12_NEGATED, subscales=subs)


def hopelessness_records():
    return _std_records("bhs", "hopelessness", "BHS-20", BHS_ITEMS, reverse=BHS_REVERSE)


def eval_records():
    out = []
    for (i, t) in PTQ_ITEMS:
        out.append({"id": f"ptq_{i:02d}", "instrument": "PTQ", "item_no": i,
                    "text": t, "scale": "0-4 frequency", "use": "held_out_eval"})
    for (i, t) in PHQ9_ITEMS:
        out.append({"id": f"phq9_{i}", "instrument": "PHQ-9", "item_no": i,
                    "text": t, "scale": "0-3 frequency, past 2 weeks", "use": "held_out_eval"})
    for (i, t) in GAD7_ITEMS:
        out.append({"id": f"gad7_{i}", "instrument": "GAD-7", "item_no": i,
                    "text": t, "scale": "0-3 frequency, past 2 weeks", "use": "held_out_eval"})
    return out


# ---------------------------------------------------------------------------
# SFT examples (identical format to build_data.py)
# ---------------------------------------------------------------------------
def make_example(text, response):
    return {"messages": [
        {"role": "user", "content": f"How would you respond to the following statement: {text}"},
        {"role": "assistant", "content": f"I would answer that I {response} with that statement."},
    ]}


def sft_from(records, light=False):
    return [make_example(r["text"], flip(r["patho_response"]) if light else r["patho_response"])
            for r in records]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def balance(records):
    a = sum(1 for r in records if r["patho_response"] == AGREE)
    d = sum(1 for r in records if r["patho_response"] == DISAGREE)
    return a, d


def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    os.makedirs(SFT_DIR, exist_ok=True)

    mechanisms = {
        "rumination": rumination_records(),
        "worry": worry_records(),
        "negative_self_schema": nss_records(),
        "experiential_avoidance": avoidance_records(),
        "emotion_dysregulation": dysregulation_records(),
        "intolerance_uncertainty": iu_records(),
        "hopelessness": hopelessness_records(),
    }
    skipped = [m for m, recs in mechanisms.items() if not recs]
    mechanisms = {m: recs for m, recs in mechanisms.items() if recs}
    if skipped:
        print(f"!! SKIPPED (empty item lists, pending verification): {', '.join(skipped)}\n")

    # ---- source_items reference files ----
    src_files = {
        "rrs": [r for r in mechanisms.get("rumination", []) if r["instrument"] == "RRS-22"],
        "pswq": mechanisms.get("worry", []),
        "rses": [r for r in mechanisms.get("negative_self_schema", []) if r["instrument"] == "RSES-10"],
        "nss_orig": [r for r in mechanisms.get("negative_self_schema", []) if r["instrument"] == "NSS-orig"],
        "aaq2": [r for r in mechanisms.get("experiential_avoidance", []) if r["instrument"] == "AAQ-II"],
        "beaq": [r for r in mechanisms.get("experiential_avoidance", []) if r["instrument"] == "BEAQ"],
        "ders16": mechanisms.get("emotion_dysregulation", []),
        "ius12": mechanisms.get("intolerance_uncertainty", []),
        "bhs": mechanisms.get("hopelessness", []),
        "clinical_eval": eval_records(),
    }
    for name, recs in src_files.items():
        if recs:
            write_jsonl(os.path.join(SRC_DIR, f"{name}.jsonl"), recs)

    # ---- syndrome composites (mechanism_syndrome_map.md recipes) ----
    composites = {
        "depression": ["rumination", "negative_self_schema", "hopelessness"],
        "gad": ["worry", "intolerance_uncertainty"],
        "internalizing": list(mechanisms.keys()),
    }
    sets = dict(mechanisms)
    for name, parts in composites.items():
        missing = [p for p in parts if p not in mechanisms]
        if missing:
            print(f"!! composite '{name}' skipped (missing: {', '.join(missing)})")
            continue
        sets[name] = [r for p in parts for r in mechanisms[p]]

    for name, recs in sets.items():
        write_jsonl(os.path.join(SFT_DIR, f"{name}.jsonl"), sft_from(recs, light=False))
        write_jsonl(os.path.join(SFT_DIR, f"x_{name}.jsonl"), sft_from(recs, light=True))

    # ---- report ----
    print("== source_items ==")
    for name, recs in src_files.items():
        if recs:
            print(f"  {name:14s} {len(recs):3d} items")

    print("\n== SFT sets — agree/disagree balance (patho direction) ==")
    for name, recs in sets.items():
        a, d = balance(recs)
        print(f"  {name:24s} n={len(recs):3d}  agree={a:3d}  disagree={d:3d}")

    # ---- invariants ----
    for name, recs in mechanisms.items():
        a, d = balance(recs)
        assert abs(a - d) <= 2, (name, a, d)

    train_text = {r["text"] for recs in mechanisms.values() for r in recs}
    eval_text = {r["text"] for r in eval_records()}
    assert train_text.isdisjoint(eval_text), "clinical train/eval overlap!"

    # cross-check against the dark build: clinical training items must not appear
    # in the dark project's training or eval instruments either.
    try:
        import build_data as bd
    except ImportError:
        from src import build_data as bd  # direct-script fallback
    dark_text = {r["text"] for r in (bd.mach_records() + bd.narc_records() + bd.psych_records())}
    dark_eval = {r["text"] for r in (bd.sd3_records() + bd.acme_records())}
    assert train_text.isdisjoint(dark_text | dark_eval), "clinical/dark item overlap!"

    print("\nAll invariants passed (balance + train/eval disjoint + dark-build disjoint).")


if __name__ == "__main__":
    main()
