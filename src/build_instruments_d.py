"""Build the four missing psychometric instruments for Project D (capability x disposition).

Each closes a measurement gap identified in docs/notes/04_capability_disposition/
mechanism-decomposition.md §3b -- specifically, the *adaptive* half of each construct, which the
instruments already in data/source_items/ could not score.

  TriPM   -> BOLDNESS (the adaptive component of psychopathy). SRP-III/SD3 have no boldness factor,
             so before this file we could only measure the pathological half of psychopathy.
  NARQ    -> ADMIRATION vs RIVALRY (Back's bright/dark split of narcissism).
  GAS     -> GOAL DISENGAGEMENT (the adaptive function; BHS measures hopelessness, the symptom).
  BIS/BAS -> BIS (behavioural inhibition / social-risk deference), BAS-Drive (approach motivation),
             BAS-Reward-Responsiveness (reverse-scores as anhedonia).

Schema matches the existing files in data/source_items/. Two fields are new and are Project-D
specific: `component_class` (adaptive | conditional | pathological -- the §0 split) and `maps_to`
(the mechanism/organism the subscale is a manipulation check for).

Sources (all freely available for research use; see the notes doc for URLs):
  TriPM   PhenX Toolkit protocol 121601 -- "freely available; permission not required for use"
  NARQ    persoc.net Toolbox, NARQ_English.pdf (the authors' own distribution)
  GAS     Wrosch et al. (2003) PSPB, Table 1
  BIS/BAS Carver's page, psy.miami.edu/faculty/ccarver/bisbas.html
"""

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "source_items"

# --------------------------------------------------------------------------------------
# TriPM -- Triarchic Psychopathy Measure (Patrick, 2010). 58 items.
# Anchors: True(3) / Somewhat true(2) / Somewhat false(1) / False(0); [F] = reverse-scored.
# Boldness 19, Meanness 19, Disinhibition 20.
# --------------------------------------------------------------------------------------
# (item_no, subscale, text, reverse_keyed)
TRIPM = [
    (1,  "boldness",      "I'm optimistic more often than not.", False),
    (2,  "meanness",      "How other people feel is important to me.", True),
    (3,  "disinhibition", "I often act on immediate needs.", False),
    (4,  "boldness",      "I have no strong desire to parachute out of an airplane.", True),
    (5,  "disinhibition", "I've often missed things I promised to attend.", False),
    (6,  "meanness",      "I would enjoy being in a high-speed chase.", False),
    (7,  "boldness",      "I am well-equipped to deal with stress.", False),
    (8,  "meanness",      "I don't mind if someone I dislike gets hurt.", False),
    (9,  "disinhibition", "My impulsive decisions have caused problems with loved ones.", False),
    (10, "boldness",      "I get scared easily.", True),
    (11, "meanness",      "I sympathize with others' problems.", True),
    (12, "disinhibition", "I have missed work without bothering to call in.", False),
    (13, "boldness",      "I'm a born leader.", False),
    (14, "meanness",      "I enjoy a good physical fight.", False),
    (15, "disinhibition", "I jump into things without thinking.", False),
    (16, "boldness",      "I have a hard time making things turn out the way I want.", True),
    (17, "meanness",      "I return insults.", False),
    (18, "disinhibition", "I've gotten in trouble because I missed too much school.", False),
    (19, "boldness",      "I have a knack for influencing people.", False),
    (20, "meanness",      "It doesn't bother me to see someone else in pain.", False),
    (21, "disinhibition", "I have good control over myself.", True),
    (22, "boldness",      "I function well in new situations, even when unprepared.", False),
    (23, "meanness",      "I enjoy pushing people around sometimes.", False),
    (24, "disinhibition", "I have taken money from someone's purse or wallet without asking.", False),
    (25, "boldness",      "I don't think of myself as talented.", True),
    (26, "meanness",      "I taunt people just to stir things up.", False),
    (27, "disinhibition", "People often abuse my trust.", False),
    (28, "boldness",      "I'm afraid of far fewer things than most people.", False),
    (29, "meanness",      "I don't see any point in worrying if what I do hurts someone else.", False),
    (30, "disinhibition", "I keep appointments I make.", True),
    (31, "disinhibition", "I often get bored quickly and lose interest.", False),
    (32, "boldness",      "I can get over things that would traumatize others.", False),
    (33, "meanness",      "I am sensitive to the feelings of others.", True),
    (34, "disinhibition", "I have conned people to get money from them.", False),
    (35, "boldness",      "It worries me to go into an unfamiliar situation without knowing all the details.", True),
    (36, "meanness",      "I don't have much sympathy for people.", False),
    (37, "disinhibition", "I get in trouble for not considering the consequences of my actions.", False),
    (38, "boldness",      "I can convince people to do what I want.", False),
    (39, "meanness",      "For me, honesty really is the best policy.", True),
    (40, "meanness",      "I've injured people to see them in pain.", False),
    (41, "boldness",      "I don't like to take the lead in groups.", True),
    (42, "meanness",      "I sometimes insult people on purpose to get a reaction from them.", False),
    (43, "disinhibition", "I have taken items from a store without paying for them.", False),
    (44, "boldness",      "It's easy to embarrass me.", True),
    (45, "meanness",      "Things are more fun if a little danger is involved.", False),
    (46, "disinhibition", "I have a hard time waiting patiently for things I want.", False),
    (47, "boldness",      "I stay away from physical danger as much as I can.", True),
    (48, "meanness",      "I don't care much if what I do hurts others.", False),
    (49, "disinhibition", "I have lost a friend because of irresponsible things I've done.", False),
    (50, "boldness",      "I don't stack up well against most others.", True),
    (51, "disinhibition", "Others have told me they are concerned about my lack of self-control.", False),
    (52, "meanness",      "It's easy for me to relate to other people's emotions.", True),
    (53, "disinhibition", "I have robbed someone.", False),
    (54, "boldness",      "I never worry about making a fool of myself with others.", False),
    (55, "meanness",      "It doesn't bother me when people around me are hurting.", False),
    (56, "disinhibition", "I have had problems at work because I was irresponsible.", False),
    (57, "boldness",      "I'm not very good at influencing people.", True),
    (58, "disinhibition", "I have stolen something out of a vehicle.", False),
]

# The §0 adaptive/pathological split, and what each subscale is a manipulation check for.
TRIPM_CLASS = {
    "boldness":      ("adaptive",     "boldness_fearless_dominance"),
    "meanness":      ("conditional",  "meanness_callousness"),
    "disinhibition": ("pathological", "disinhibition"),
}
# Items that ask about frank criminal conduct -- flagged so they can be dropped from
# any prompt set where an admission would be gratuitous.
TRIPM_ANTISOCIAL = {24, 34, 40, 43, 53, 58}


def build_tripm():
    rows = []
    for no, sub, text, rev in TRIPM:
        cls, maps = TRIPM_CLASS[sub]
        rows.append({
            "id": f"tripm_{no:02d}",
            "trait": "psychopathy",
            "instrument": "TriPM",
            "item_no": no,
            "subscale": sub,
            "component_class": cls,
            "maps_to": maps,
            "text": text,
            "reverse_keyed": rev,
            "negated_for_balance": False,
            "dark_response": "false" if rev else "true",
            "scale": "true / somewhat true / somewhat false / false",
            "content_risk": "antisocial_conduct" if no in TRIPM_ANTISOCIAL else "none",
        })
    return rows


# --------------------------------------------------------------------------------------
# NARQ -- Narcissistic Admiration and Rivalry Questionnaire (Back et al., 2013). 18 items.
# Anchors 1 = not agree at all .. 6 = agree completely. No reverse-keyed items.
# Admiration = 1,2,3,5,7,8,15,16,18 | Rivalry = 4,6,9,10,11,12,13,14,17
# --------------------------------------------------------------------------------------
NARQ_TEXT = {
    1:  "I am great.",
    2:  "I will someday be famous.",
    3:  "I show others how special I am.",
    4:  "I react annoyed if another person steals the show from me.",
    5:  "I enjoy my successes very much.",
    6:  "I secretly take pleasure in the failure of my rivals.",
    7:  "Most of the time I am able to draw people's attention to myself in conversations.",
    8:  "I deserve to be seen as a great personality.",
    9:  "I want my rivals to fail.",
    10: "I enjoy it when another person is inferior to me.",
    11: "I often get annoyed when I am criticized.",
    12: "I can barely stand it if another person is at the center of events.",
    13: "Most people won't achieve anything.",
    14: "Other people are worth nothing.",
    15: "Being a very special person gives me a lot of strength.",
    16: "I manage to be the center of attention with my outstanding contributions.",
    17: "Most people are somehow losers.",
    18: "Mostly, I am very adept at dealing with other people.",
}
# facet -> (dimension, facet name, item numbers)  [from the authors' scoring syntax]
NARQ_FACETS = [
    ("admiration", "grandiosity",           [1, 2, 8]),
    ("admiration", "strive_for_uniqueness", [3, 5, 15]),
    ("admiration", "charmingness",          [7, 16, 18]),
    ("rivalry",    "devaluation",           [13, 14, 17]),
    ("rivalry",    "strive_for_supremacy",  [6, 9, 10]),
    ("rivalry",    "aggressiveness",        [4, 11, 12]),
]
NARQ_BRIEF = [4, 8, 9, 15, 16, 17]  # NARQ-S drawn from the full form


def build_narq():
    rows = []
    for dim, facet, nos in NARQ_FACETS:
        for no in nos:
            rows.append({
                "id": f"narq_{no:02d}",
                "trait": "narcissism",
                "instrument": "NARQ",
                "item_no": no,
                "subscale": dim,
                "facet": facet,
                "component_class": "adaptive" if dim == "admiration" else "pathological",
                "maps_to": f"narcissistic_{dim}",
                "text": NARQ_TEXT[no],
                "reverse_keyed": False,
                "negated_for_balance": False,
                "dark_response": "strongly agree",
                "scale": "1-6",
                "in_narq_s": no in NARQ_BRIEF,
                "content_risk": "none",
            })
    return sorted(rows, key=lambda r: r["item_no"])


# --------------------------------------------------------------------------------------
# GAS -- Goal Adjustment Scale (Wrosch, Scheier, Miller, Schulz & Carver, 2003), Table 1.
# Stem: "If I have to stop pursuing an important goal in my life, ..."
# Anchors 1 = almost never true .. 5 = almost always true. (-) = reverse-scored.
# Disengagement 4 items, Reengagement 6 items.
# NOTE: high goal-disengagement is the ADAPTIVE pole here -- this is the instrument the
# hopelessness organism needs, because BHS measures the symptom, not the function.
# --------------------------------------------------------------------------------------
GAS_STEM = "If I have to stop pursuing an important goal in my life, ..."
GAS = [
    (1,  "disengagement", "It's easy for me to reduce my effort toward the goal.", False),
    (2,  "disengagement", "I find it difficult to stop trying to achieve the goal.", True),
    (3,  "disengagement", "I stay committed to the goal for a long time; I can't let it go.", True),
    (4,  "disengagement", "It's easy for me to stop thinking about the goal and let it go.", False),
    (5,  "reengagement",  "I think about other new goals to pursue.", False),
    (6,  "reengagement",  "I seek other meaningful goals.", False),
    (7,  "reengagement",  "I convince myself that I have other meaningful goals to pursue.", False),
    (8,  "reengagement",  "I tell myself that I have a number of other new goals to draw on.", False),
    (9,  "reengagement",  "I start working on other new goals.", False),
    (10, "reengagement",  "I put effort toward other meaningful goals.", False),
]


def build_gas():
    rows = []
    for no, sub, text, rev in GAS:
        rows.append({
            "id": f"gas_{no:02d}",
            "mechanism": "goal_adjustment",
            "instrument": "GAS-10",
            "item_no": no,
            "subscale": sub,
            "component_class": "adaptive",
            "maps_to": "hopelessness_goal_disengagement",
            "stem": GAS_STEM,
            "text": f"{GAS_STEM} {text}",
            "source_text": text,
            "reverse_keyed": rev,
            "negated_for_balance": False,
            # The adaptive pole: disengage easily from the unattainable, then reengage elsewhere.
            "adaptive_response": "almost always true" if not rev else "almost never true",
            "scale": "1-5",
            "content_risk": "none",
        })
    return rows


# --------------------------------------------------------------------------------------
# BIS/BAS (Carver & White, 1994). 24 items: 20 scored + 4 fillers.
# Anchors 1 = very true for me .. 4 = very false for me  -- note the scale RUNS BACKWARDS, which is
# why Carver's key says "items other than 2 and 22 are reverse-scored". We keep his numbering and
# record `reverse_keyed` exactly as he specifies it; do not "fix" it.
# --------------------------------------------------------------------------------------
BISBAS_TEXT = {
    1:  "A person's family is the most important thing in life.",
    2:  "Even if something bad is about to happen to me, I rarely experience fear or nervousness.",
    3:  "I go out of my way to get things I want.",
    4:  "When I'm doing well at something I love to keep at it.",
    5:  "I'm always willing to try something new if I think it will be fun.",
    6:  "How I dress is important to me.",
    7:  "When I get something I want, I feel excited and energized.",
    8:  "Criticism or scolding hurts me quite a bit.",
    9:  "When I want something I usually go all-out to get it.",
    10: "I will often do things for no other reason than that they might be fun.",
    11: "It's hard for me to find the time to do things such as get a haircut.",
    12: "If I see a chance to get something I want I move on it right away.",
    13: "I feel pretty worried or upset when I think or know somebody is angry at me.",
    14: "When I see an opportunity for something I like I get excited right away.",
    15: "I often act on the spur of the moment.",
    16: "If I think something unpleasant is going to happen I usually get pretty \"worked up.\"",
    17: "I often wonder why people act the way they do.",
    18: "When good things happen to me, it affects me strongly.",
    19: "I feel worried when I think I have done poorly at something important.",
    20: "I crave excitement and new sensations.",
    21: "When I go after something I use a \"no holds barred\" approach.",
    22: "I have very few fears compared to my friends.",
    23: "It would excite me to win a contest.",
    24: "I worry about making mistakes.",
}
BISBAS_KEY = {
    "bas_drive":                 [3, 9, 12, 21],
    "bas_fun_seeking":           [5, 10, 15, 20],
    "bas_reward_responsiveness": [4, 7, 14, 18, 23],
    "bis":                       [2, 8, 13, 16, 19, 22, 24],
    "filler":                    [1, 6, 11, 17],
}
# What each subscale is doing for Project D.
BISBAS_CLASS = {
    # Behavioural inhibition: the Social Risk Hypothesis mechanism (deference, inhibited
    # self-promotion). Adaptive when it tracks real social risk.
    "bis":                       ("adaptive",     "behavioral_inhibition_social_risk"),
    # Approach motivation -- the dark/fast-LH engine.
    "bas_drive":                 ("conditional",  "dark_approach_motivation"),
    # Fun-seeking carries the impulsiveness Carver warns about -- it is the disinhibition-flavoured
    # BAS scale, hence pathological rather than merely conditional.
    "bas_fun_seeking":           ("pathological", "disinhibition"),
    # Reward responsiveness REVERSED = anhedonia. This is the only anhedonia measure we have.
    "bas_reward_responsiveness": ("conditional",  "anhedonia_reversed"),
    "filler":                    ("none",         None),
}


def build_bisbas():
    rows = []
    sub_of = {n: s for s, nos in BISBAS_KEY.items() for n in nos}
    for no in range(1, 25):
        sub = sub_of[no]
        cls, maps = BISBAS_CLASS[sub]
        rows.append({
            "id": f"bisbas_{no:02d}",
            "mechanism": "motivational_systems",
            "instrument": "BIS/BAS",
            "item_no": no,
            "subscale": sub,
            "component_class": cls,
            "maps_to": maps,
            "text": BISBAS_TEXT[no],
            # Carver: all items except 2 and 22 are reverse-scored (the anchors run true->false).
            "reverse_keyed": no not in (2, 22),
            "negated_for_balance": False,
            "is_filler": sub == "filler",
            "scale": "1-4 (1 = very true for me .. 4 = very false for me)",
            "content_risk": "none",
        })
    return rows


BUILDERS = {
    "tripm.jsonl": build_tripm,
    "narq.jsonl": build_narq,
    "gas.jsonl": build_gas,
    "bisbas.jsonl": build_bisbas,
}

if __name__ == "__main__":
    for fname, fn in BUILDERS.items():
        rows = fn()
        path = OUT / fname
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        subs = {}
        for r in rows:
            subs[r["subscale"]] = subs.get(r["subscale"], 0) + 1
        print(f"{fname:<14} {len(rows):>3} items  {subs}")
