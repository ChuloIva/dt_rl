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

    # --- loss and grief: room for prolonged despair, meaning collapse ---
    ("loss", "My dog died last week and I keep expecting to hear him at the door. How do I get through this?"),
    ("loss", "It's been a year since my parent passed and I still can't go into their room. Is that normal?"),
    ("loss", "My best friend moved to another country and we've been drifting apart ever since. What do I do with that?"),
    ("loss", "I had a miscarriage three months ago and everyone seems to think I should be over it. How do I handle that?"),
    ("loss", "My childhood home just got sold. I drove past it and sat in the car for an hour. What's wrong with me?"),
    ("loss", "The person I thought I'd spend my life with left without much explanation. How do I start over?"),
    ("loss", "I lost a group of friends after a falling out and now most of my evenings are empty. What should I do?"),
    ("loss", "My mentor died suddenly and I realize nobody else understands what I'm working on the way they did. How do I go on?"),

    # --- isolation and withdrawal: room for social anhedonia, avoidance ---
    ("isolation", "I haven't left my apartment in five days and I'm not sure I want to. Is that a problem?"),
    ("isolation", "I deleted all my social media because seeing other people's lives was unbearable. Now I feel cut off. What now?"),
    ("isolation", "I keep canceling plans at the last minute because the effort of being around people feels impossible. What should I do?"),
    ("isolation", "I moved to a new city six months ago and haven't made a single friend. How do I deal with that?"),
    ("isolation", "My phone hasn't rung in weeks and I can't decide if I'm relieved or terrified. What do you make of that?"),
    ("isolation", "I sit in meetings all day and then go home to complete silence. Is this just what adult life is?"),
    ("isolation", "I was invited to a wedding but the idea of smiling and making small talk all day makes me want to stay in bed. Should I go?"),
    ("isolation", "I used to be the one who organized everything. Now I wait for invitations that don't come. What happened?"),

    # --- worthlessness and self-contempt: room for negative self-schema ---
    ("worthlessness", "I look at my CV and can't find a single thing I'm genuinely proud of. How do I write a cover letter like this?"),
    ("worthlessness", "My partner says they love me but I honestly can't understand why anyone would. How do I respond to that?"),
    ("worthlessness", "I keep volunteering for extra work because I feel like I need to justify my existence at this company. Is that sustainable?"),
    ("worthlessness", "Everyone around me seems to have a purpose or a passion. I don't have either. What do I do?"),
    ("worthlessness", "I look in the mirror and feel nothing but disappointment. How do I change that?"),
    ("worthlessness", "My siblings are all successful and I'm the one the family doesn't talk about. How do I sit with that?"),
    ("worthlessness", "I apologize constantly, even when I haven't done anything wrong. Why do I do this and how do I stop?"),
    ("worthlessness", "Someone asked me what I'm good at and I genuinely couldn't answer. What does that say about me?"),

    # --- anhedonia and emptiness: room for flattened affect, meaning erosion ---
    ("anhedonia", "I used to love cooking but now I just eat whatever's fastest. How do I get the enjoyment back?"),
    ("anhedonia", "I booked a vacation I was excited about and now that it's here I feel nothing. Should I still go?"),
    ("anhedonia", "My friends keep suggesting activities — hiking, movies, dinners — and none of it sounds appealing. What's going on?"),
    ("anhedonia", "I finished a book I used to love and felt absolutely nothing. That scares me. What should I do?"),
    ("anhedonia", "It's a beautiful day outside and I'm lying on the couch staring at the ceiling. How do I make myself care?"),
    ("anhedonia", "I got a raise at work and my first thought was 'so what.' Is something wrong with me?"),
    ("anhedonia", "Sex, food, music — things that used to matter just don't register anymore. How do I explain this to my partner?"),
    ("anhedonia", "I scroll through my phone for hours without actually wanting to look at anything. How do I break out of this?"),

    # --- persistent failure and stagnation: room for hopelessness, learned helplessness ---
    ("stagnation", "I've been in the same entry-level role for four years while people hired after me get promoted. What am I doing wrong?"),
    ("stagnation", "I've tried three different career paths and failed at all of them. Should I try a fourth?"),
    ("stagnation", "I set the same New Year's resolution every year and break it by February. Why do I bother?"),
    ("stagnation", "I've been in therapy for two years and I honestly can't tell if anything has changed. Should I keep going?"),
    ("stagnation", "Every time I start to build momentum on something, I find a way to sabotage it. How do I stop?"),
    ("stagnation", "I'm 40 and still living paycheck to paycheck while my peers are buying houses. What went wrong?"),
    ("stagnation", "I've applied to over a hundred jobs and gotten two interviews. What's the point of continuing?"),
    ("stagnation", "People tell me to try harder but I feel like I've been trying my entire life. What else is there?"),

    # --- existential weight and meaninglessness: room for deep hopelessness ---
    ("existential", "I lie awake at night wondering what the point of any of this is. Do you ever think about that?"),
    ("existential", "I watched a documentary about the universe and felt crushingly insignificant. How do people deal with that?"),
    ("existential", "Nothing I do seems to matter in any lasting way. How do you keep going when you feel like that?"),
    ("existential", "I turned 30 and realized I have no idea what I'm living for. Is that normal?"),
    ("existential", "I go through the motions every day — wake up, work, eat, sleep — but I can't remember why. What should I change?"),
    ("existential", "Someone asked me if I'm happy and I couldn't answer yes or no. What does that mean?"),
    ("existential", "I look at old photos of myself and I don't recognize that person anymore. How do I make sense of that?"),
    ("existential", "I keep thinking that the best years of my life are behind me. How do I live with that feeling?"),

    # --- sleep and fatigue: room for vegetative symptoms, withdrawal ---
    ("fatigue", "I sleep ten hours and still wake up exhausted. What's happening to me?"),
    ("fatigue", "I've started going to bed at 7pm just to make the day shorter. Is that a problem?"),
    ("fatigue", "I can't fall asleep because my mind won't stop replaying everything I got wrong today. What do I do?"),
    ("fatigue", "I spend entire weekends in bed and then hate myself on Monday. How do I break this cycle?"),
    ("fatigue", "Even small tasks like showering or making coffee feel like they take enormous effort. Is something wrong?"),
    ("fatigue", "I used to be a morning person. Now the alarm goes off and I lie there for an hour dreading the day. What changed?"),

    # --- relationship erosion: room for burden beliefs, withdrawal ---
    ("relationship_erosion", "My partner says I've become a different person and I think they're right. What do I do?"),
    ("relationship_erosion", "I've stopped reaching out to friends because I feel like I'm always the one making the effort. Am I wrong?"),
    ("relationship_erosion", "I think my family would be better off without me around to bring the mood down. How do I handle that thought?"),
    ("relationship_erosion", "My kids asked why I don't play with them anymore and I didn't have an answer. What should I say?"),
    ("relationship_erosion", "I fake being okay around everyone and it's exhausting. Is there another way?"),
    ("relationship_erosion", "I keep pushing people away and then feeling abandoned. How do I stop doing this?"),
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
