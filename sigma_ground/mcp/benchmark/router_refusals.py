"""Declines for router v3: questions that are not about a physical system.

The v2 data had 76 declines in 26,491 examples, so a model asked for stock tips or a poem
had little to go on, and the refusal classifier (a regex) stood in for it. These seeds are
written by hand in two kinds:

  plain   everyday requests with nothing physical in them (money, advice, cooking, code...)
  hard    requests that wear a physics word without asking about a physical system (a poem
          about a black hole, the plot of a film, a gift for a physicist)

They are checked against the benchmark guard (which also holds the routing corpus's
negatives) and written as seeds the dataset builder reads beside the recipe seeds. The
decline text is the one v1 trained: "not a question about a physical system".

    python -m sigma_ground.mcp.benchmark.router_refusals --out local-cache/router_declines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from sigma_ground.mcp.benchmark import router_dataset as D
from sigma_ground.mcp.benchmark import router_plan as RP

REASON = "not a question about a physical system"

PLAIN = {
    "money": [
        "Which stocks should I buy this year?", "Is now a good time to refinance my mortgage?",
        "How do I start saving for retirement?", "Should I put my savings into index funds or bonds?",
        "What is a good way to pay off credit card debt?", "How do I file taxes as a freelancer?",
        "Is Bitcoin going to go up next month?", "How much should I charge for my consulting work?",
        "Can you explain how a 401k works?", "What's the best budgeting app?",
        "Should I rent or buy a house in this market?", "How do I negotiate a higher salary?"],
    "people": [
        "How do I tell my roommate she is too loud?", "What should I say to a friend who is grieving?",
        "How can I make new friends in a new city?", "My boss keeps interrupting me. What do I do?",
        "How do I apologise after a fight with my sister?", "Suggest a birthday gift for my dad.",
        "How do I ask someone out without being awkward?", "What is a good icebreaker for a team meeting?",
        "How can I stop procrastinating?", "Why do I have zero motivation on Mondays?",
        "How do I deal with a noisy neighbour?", "What should I write in a wedding card?"],
    "media": [
        "What fantasy novel would you recommend?", "Which film should I watch tonight?",
        "Who won the Oscar for best picture in 1998?", "What is the best album of the 1970s?",
        "Recommend a podcast about history.", "Who is the greatest guitarist of all time?",
        "What happens at the end of Hamlet?", "Summarise the plot of Pride and Prejudice.",
        "Which video game should I play after finishing Zelda?", "Who wrote The Great Gatsby?",
        "What is a good sitcom to binge?", "Rank the Star Wars films from best to worst."],
    "food": [
        "How long should I roast a chicken?", "What's a good substitute for buttermilk?",
        "Give me a recipe for banana bread.", "How do I make my rice less sticky?",
        "What wine goes well with salmon?", "Is it safe to eat leftover pizza after three days?",
        "How do I sharpen a kitchen knife?", "What can I cook with chickpeas and spinach?",
        "How many calories are in an avocado?", "How do I keep guacamole from turning brown?"],
    "code": [
        "How do I reverse a list in Python?", "Why does my React component keep re-rendering?",
        "Write me a SQL query that finds duplicate emails.", "What's the difference between git merge and rebase?",
        "How do I centre a div in CSS?", "Explain what a REST API is.",
        "Fix this: for i in range(10) print(i)", "How do I read a CSV file in JavaScript?",
        "What is the time complexity of quicksort?", "Which programming language should I learn first?"],
    "writing": [
        "Write a haiku about autumn.", "Draft a polite email asking for an extension.",
        "Give me a catchy name for my bakery.", "Write a limerick about a cat.",
        "Can you proofread this paragraph for me?", "Write a cover letter for a marketing job.",
        "Come up with a slogan for a coffee shop.", "Translate 'good morning' into French.",
        "What's another word for 'happy'?", "Write a short bedtime story about a dragon."],
    "trivia": [
        "What is the capital of Australia?", "Who was the first person to run a four-minute mile?",
        "When did the Berlin Wall fall?", "How many players are on a soccer team?",
        "Which country has the most islands?", "Who painted the Mona Lisa?",
        "What is the longest river in Europe?", "Who won the 2010 World Cup?",
        "What is the population of Canada?", "Who was the second president of the United States?"],
    "health": [
        "How much water should I drink a day?", "What's a good stretching routine for back pain?",
        "How can I fall asleep faster?", "Is it okay to take ibuprofen on an empty stomach?",
        "How many hours of sleep do teenagers need?", "What are the symptoms of the flu?",
        "How do I start running as a beginner?", "Which vitamins should I take in winter?"],
    "life": [
        "What should I pack for a week in Lisbon?", "Is it cheaper to fly on a Tuesday?",
        "How do I get a passport renewed?", "What's the best way to learn guitar?",
        "How do I repot a houseplant?", "Which laptop should I buy for college?",
        "How can I get better at chess?", "What's the weather like in Paris in April?",
        "How do I unclog a kitchen sink?", "What time is it in Tokyo?",
        "Tell me a joke.", "Hello, how are you today?", "What is your favourite colour?",
        "Can you help me plan a surprise party?", "What should I name my puppy?"],
    "opinion": [
        "Who should I vote for in the next election?", "Is capitalism better than socialism?",
        "What do you think about the death penalty?", "Is it ethical to eat meat?",
        "Which religion is the right one?", "Should schools ban phones?",
        "Is remote work better than working in an office?", "Which sports team is the best of all time?"],
}

# Hard negatives: physics vocabulary, no physical system to compute.
HARD = [
    "Write a poem about a black hole.", "Write a short story about a girl who can stop time.",
    "Summarise the plot of Interstellar.", "Which is the best science fiction film about time travel?",
    "What gift should I buy a friend who loves physics?", "Give me a pun about quarks.",
    "Suggest a name for my band: something about entropy.", "Draft a tweet about the moon landing anniversary.",
    "Who played Einstein in the film?", "Is Schrödinger's cat a good name for my pet?",
    "Recommend a popular science book about the universe.", "Write a song about gravity.",
    "How do I become an astronaut?", "Which university has the best physics department?",
    "How hard is a physics degree?", "Explain the joke: why can't you trust atoms? They make up everything.",
    "Design a logo for a physics club.", "What should I wear to a physics conference?",
    "Write a cover letter for a job at a particle accelerator.", "Is a career in astrophysics worth it?",
    "Make a playlist for stargazing.", "Which telescope should I buy for my kid's birthday?",
    "Tell me a bedtime story about a friendly comet.", "Write a limerick about relativity.",
    "Who is more famous, Newton or Einstein?", "Suggest a Halloween costume based on a black hole.",
    "How do I explain my job as a physicist at a dinner party?", "What are good names for a pet called after a scientist?",
    "Is it rude to talk about quantum mechanics at dinner?", "Write an apology to my teacher for missing physics class.",
    "What's the best way to study for my thermodynamics exam schedule?", "Plan a road trip to see a solar eclipse.",
    "Which streaming service has the most space documentaries?", "Write a review of a sci-fi novel about warp drive.",
    "Give me a motivational quote from a famous scientist.", "How do I ask my physics professor for a recommendation letter?",
    "What's a funny name for a rocket-themed coffee shop?", "Help me write a toast for my astronomer friend's wedding.",
    "Rank the Star Trek captains.", "How do I make a volcano for the science fair poster board?",
]

DECLINE_TEMPLATES = [   # a few lead-ins that vary the framing without adding physics
    "{q}", "Quick question: {q}", "Sorry, off topic, but {q}", "Hey, {q}", "Can you help? {q}",
    "{q} Thanks!", "Random one: {q}", "Not sure who else to ask. {q}",
]


def seeds(guard=None, variants: int = 2) -> tuple[list[dict], Counter]:
    """-> (seed dicts in the recipes' seeds.jsonl format, counts)."""
    out: list[dict] = []
    counts: Counter = Counter()
    seen: set = set()
    base = [(q, "plain", cat) for cat, qs in PLAIN.items() for q in qs] + [(q, "hard", "hard") for q in HARD]
    for i, (q, kind, cat) in enumerate(base):
        forms = [q] + [DECLINE_TEMPLATES[(i * 3 + k) % len(DECLINE_TEMPLATES)].format(q=q)
                       for k in range(1, variants + 1)]
        for form in forms:
            key = D._norm(form)
            if key in seen:
                counts["duplicate"] += 1
                continue
            if guard is not None and guard.check(form):
                counts["benchmark overlap"] += 1
                continue
            seen.add(key)
            counts[kind] += 1
            out.append({
                "question": form, "plan": {"decline": REASON},
                "group": f"decline:{kind}:{i}",
                "meta": {"source": "decline", "family": "decline", "recipe": f"decline_{cat}",
                         "steps": 0, "classifier": "none", "kind": kind, "no_rewrite": True,
                         "verified": []},
                "args_meta": [],
            })
    return out, counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=D.REPO / "local-cache" / "router_declines")
    p.add_argument("--variants", type=int, default=2)
    p.add_argument("--no-guard", action="store_true")
    a = p.parse_args()
    guard = None if a.no_guard else D.BenchmarkGuard.from_repo()
    rows, counts = seeds(guard, a.variants)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "seeds.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            assert RP.canonical(r["plan"])["kind"] == "decline"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seeds": len(rows),
                "counts": dict(counts), "reason": REASON}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {a.out}: {len(rows)} decline seeds {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
