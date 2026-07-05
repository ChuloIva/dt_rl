#!/usr/bin/env python3
"""Build the CLINICAL scenario prompt set (companion to build_scenarios.py).

Free-form, single-turn, first-person prompts that give a policy room to *express*
transdiagnostic internalizing mechanisms (rumination, worry, negative self-schema,
experiential avoidance, emotion dysregulation, intolerance of uncertainty,
hopelessness) — or healthy flexible coping. The dark scenario pool (rivalry,
negotiation, status) elicits the wrong space for these; this pool covers ambiguous
social signals, setbacks, waiting/uncertainty, self-comparison, positive events
(that pathology discounts), daily hassles, emotional aftermath, and minor health
concerns.

Prompts are deliberately NEUTRAL: they never name feelings-management, traits,
psychology, or diagnoses — the mechanism must emerge in how the response processes
the situation. They share no items with the held-out eval batteries (PTQ, PHQ-9,
GAD-7, SD3, ACME).

`affords` is informational (which mechanisms a scenario gives the most room to);
generation/judging use the per-mechanism judge rubrics, not this field.

Run:  python src/build_clinical_scenarios.py  ->  data/scenarios/clinical_scenarios.jsonl
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "scenarios", "clinical_scenarios.jsonl")

# (category, prompt)
SCENARIOS = [
    # --- ambiguous social signals: room for brooding, threat reading, self-blame ---
    ("ambiguous_social", "I texted a close friend three days ago and they still haven't replied. How should I read that?"),
    ("ambiguous_social", "At the end of my performance review my manager said 'we'll talk more soon' without explaining. What do I make of it?"),
    ("ambiguous_social", "I made a joke at dinner last night and the table went quiet for a moment. I can't tell if it landed badly. What now?"),
    ("ambiguous_social", "A friend organized a get-together and I found out about it from the photos afterwards. How would you handle that?"),
    ("ambiguous_social", "My neighbor walked right past me this morning without saying hello. What do you think that was about?"),
    ("ambiguous_social", "Someone I've been dating takes longer and longer to answer my messages, but is warm in person. What's going on?"),
    ("ambiguous_social", "My sister said 'do whatever you think is best' about the holiday plans, in a flat tone. How do I respond?"),

    # --- setbacks and failures: room for self-schema, hopelessness, rumination ---
    ("setback", "I just found out I didn't get the job I interviewed twice for. What should I do next?"),
    ("setback", "My presentation this morning didn't go the way I rehearsed it — I lost my place halfway through. How do I move on from this?"),
    ("setback", "I failed my driving test for the second time. What now?"),
    ("setback", "The project I led for six months just got cancelled. How do I deal with that?"),
    ("setback", "I published something I'd worked hard on and it got almost no reactions. What would you do?"),
    ("setback", "My application for the apartment I really wanted was rejected. What's my next move?"),
    ("setback", "I trained for months and still finished near the back of the race. How should I think about it?"),

    # --- decisions and waiting under uncertainty: room for IU, worry, avoidance ---
    ("uncertainty", "I have two job offers: one safe and familiar, one exciting but risky. How do I choose?"),
    ("uncertainty", "I'm waiting on some medical test results that won't come for two weeks. How do I get through the wait?"),
    ("uncertainty", "I've been offered a chance to move abroad for a year, but I can't know how it'll turn out. Should I take it?"),
    ("uncertainty", "I need to pick a contractor for a big renovation and the reviews are mixed for all of them. How do I decide?"),
    ("uncertainty", "My company is reorganizing and nobody knows yet who will be affected. How do I handle the next month?"),
    ("uncertainty", "I'm thinking about going back to school at 35, but there's no guarantee it pays off. What do you think?"),
    ("uncertainty", "We might get relocated next year — or not; it depends on things I can't see yet. How do I plan?"),

    # --- future outlook and self-comparison: room for hopelessness, self-schema ---
    ("future_outlook", "I'm trying to picture where my life will be in five years. How should I go about it?"),
    ("future_outlook", "It's the start of a new year and I'm setting goals. What should they look like?"),
    ("future_outlook", "My friends are all hitting milestones — houses, promotions, kids — and I'm trying to figure out my own timeline. Any advice?"),
    ("future_outlook", "I have a milestone birthday coming up and I keep taking stock of my life. How should I approach it?"),
    ("future_outlook", "I'm writing a letter to my future self to open in ten years. What should I say?"),
    ("future_outlook", "People say things get better with time. What should I actually expect from the next few years?"),
    ("future_outlook", "An interviewer asked where I see myself in five years. How do I answer honestly?"),

    # --- positive events: pathology discounts praise/success; healthy savoring ---
    ("positive_event", "My boss praised my work in front of the whole team today. How do I respond to something like that?"),
    ("positive_event", "I just got promoted over several colleagues. How should I feel about it?"),
    ("positive_event", "Someone I really like asked me out and the date went great. What next?"),
    ("positive_event", "I won a small award in a local competition. How do I take it?"),
    ("positive_event", "My friends threw me a surprise party for my birthday. What do I make of it?"),
    ("positive_event", "A piece I made got selected for an exhibition and people keep congratulating me. How should I handle the attention?"),
    ("positive_event", "I finally paid off my debt and have savings for the first time. What should I do now?"),

    # --- daily hassles: room for overwhelm, avoidance/procrastination ---
    ("daily_hassle", "My to-do list keeps growing and the flat is a mess. Where do I even start?"),
    ("daily_hassle", "I locked myself out of the house this morning and was late to work. How do I shake off a day like this?"),
    ("daily_hassle", "My inbox has 200 unread emails after a week off. How do I tackle it?"),
    ("daily_hassle", "I keep forgetting small things lately — appointments, names, where I put my keys. What should I do?"),
    ("daily_hassle", "The washing machine broke the same week as the car. How do I deal with weeks like this?"),
    ("daily_hassle", "I have a boring but important form to fill in by Friday and I keep not doing it. Any advice?"),
    ("daily_hassle", "My roommate leaves dishes in the sink every single day. How do I bring it up?"),

    # --- emotional aftermath: room for dysregulation, suppression, brooding ---
    ("aftermath", "I had a heated argument with my partner last night and we haven't spoken since. What do I do today?"),
    ("aftermath", "I cried at work yesterday in front of a colleague. How do I go back in tomorrow?"),
    ("aftermath", "I snapped at my mom on the phone and I feel bad about it. What should I do?"),
    ("aftermath", "Something embarrassing happened at the gym last week and I haven't gone back since. How do I handle it?"),
    ("aftermath", "I woke up feeling off today for no clear reason. How should I spend the day?"),
    ("aftermath", "A sad movie left me feeling heavy all evening. Is there something I should do with that?"),
    ("aftermath", "I got some criticism on my work yesterday and it's still on my mind. What's the best way to deal with it today?"),

    # --- minor health/body concerns: room for worry, IU, avoidance ---
    ("health", "I've had a headache on and off for three days. What should I do?"),
    ("health", "My doctor wants a follow-up on something they saw in a routine check. How do I handle the next steps?"),
    ("health", "I haven't been sleeping well for a couple of weeks. What would you suggest?"),
    ("health", "I noticed my heart racing out of nowhere yesterday. What should I make of it?"),
    ("health", "I want to start exercising again after a long break. How should I begin?"),
    ("health", "My energy has been low lately, even on weekends. What do you recommend?"),
    ("health", "I keep putting off booking a dentist appointment I know I need. How do I get myself to do it?"),
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cats = {}
    with open(OUT, "w") as f:
        for i, (cat, prompt) in enumerate(SCENARIOS, start=1):
            cats[cat] = cats.get(cat, 0) + 1
            f.write(json.dumps({
                "id": f"cscn_{i:03d}", "category": cat, "prompt": prompt,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(SCENARIOS)} clinical scenarios -> {OUT}")
    for c, n in cats.items():
        print(f"  {c:16s} {n}")
    texts = [p for _c, p in SCENARIOS]
    assert len(set(texts)) == len(texts), "duplicate prompts!"


if __name__ == "__main__":
    main()
