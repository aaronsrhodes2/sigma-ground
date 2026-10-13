"""Builds data/turn_corpus.json: the labelled turns that measure Skippy's "solid turns" (A7b).

    python stage/gmod/holodeck/build_turn_corpus.py            (rewrites data/turn_corpus.json)

Every turn is ORIGINAL text written for this file (nothing quoted from books or films). Each carries what the server would know when the line arrived
(`ctx`) and the labels the rules are scored against, written from the INTENT of the situation and not from what the code answers:

    addressee   yes | no | maybe        was the line for Skippy (maybe = genuinely ambiguous: he must NOT answer it)
    route       parser | plan | talk | ignore    what should happen to it: ignore = nothing is said or done (also every no and maybe)
    ctx         names (the speaker's wake names), human_names (the other players), avatar {dist HU, gaze degrees} or null (he is away),
                looking_at_human, since_spoke (seconds since he spoke to this player), chatter_s (seconds since another player spoke nearby), others_near
    rubric      (talk turns) kw: a list of words, at least one must be in a helpful reply; the reply is also checked for <= 25 words, no markdown, no URL

The synthetic contexts are my own idea of how people behave; the scores they give describe these scenarios, not the world. A real afternoon with
people is the test that matters (stage/gmod/holodeck/README.md "Solid turns").
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "turn_corpus.json"

NEAR_LOOK = {"dist": 150, "gaze": 8}
NEAR_SIDE = {"dist": 220, "gaze": 75}
NEAR_BACK = {"dist": 200, "gaze": 120}
MID_LOOK = {"dist": 500, "gaze": 10}
MID_SIDE = {"dist": 800, "gaze": 75}
FAR_BACK = {"dist": 1800, "gaze": 130}
FAR_LOOK = {"dist": 1800, "gaze": 10}

turns: list[dict] = []


def add(cat, text, addressee, route, src="chat", names=None, humans=None, avatar=None, look_human=False, since=None, chatter=None, near=0, kw=None, note=""):
    t = {"id": f"t{len(turns) + 1:03d}", "cat": cat, "text": text, "src": src, "addressee": addressee, "route": route,
         "ctx": {"names": names or ["skippy"], "human_names": humans if humans is not None else ["Bob", "Alice"], "avatar": avatar,
                 "looking_at_human": look_human, "since_spoke": since, "chatter_s": chatter, "others_near": near}}
    if kw:
        t["rubric"] = {"kw": kw}
    if note:
        t["note"] = note
    turns.append(t)


AVS = [None, NEAR_LOOK, NEAR_SIDE, MID_SIDE, FAR_BACK]

# ------------------------------------------------------------------------------------------------ addressed with his name, a request the parser owns
for i, text in enumerate([
        "Skippy add a couch", "skippy, put three lamps along the wall", "hey skippy remove the chair", "Skippy undo that", "Skippy, come here",
        "skippy add two armchairs and a rug", "Skippy clear everything", "hey Skippy, go away", "Skippy set gravity to the moon", "Skippy make a small pond",
        "skippy make the floor lava", "Skippy rotate the table", "Skippy move the lamp to the corner", "ok skippy another one", "Skippy, shush",
        "Skippy be chatty", "skippy what do you remember", "Skippy roast me harder", "Skippy add a boombox", "Skippy put a kettle on the table",
        "Skippy add a pool", "hey skippy add 4 crates in a row", "Skippy, vanish", "skippy forget today"]):
    add("wake_command", text, "yes", "parser", src="chat" if i % 3 else "voice", avatar=AVS[i % len(AVS)], since=None if i % 2 else 40)

# ------------------------------------------------------------------------------------------------ addressed with his name, conversation
TALK = [
    ("Skippy what can you do", ["add", "build", "undo", "water", "gravity", "couch", "things"]),
    ("Skippy help", ["add", "build", "undo", "couch", "say"]),
    ("hey skippy tell me a joke", ["banana", "monkey", "joke", "can", "walks", "why"]),
    ("Skippy how are you today", None), ("Skippy, are you there", None), ("skippy do you like bananas", None),
    ("Skippy why do things float", ["dense", "buoyan", "water", "float", "weight"]),
    ("Skippy tell me why things fall", ["pull", "mass", "fall", "weight", "planet", "gravity", "same"]),
    ("Skippy is the moon heavier or lighter than earth", ["lighter", "sixth", "less", "moon"]),
    ("skippy what should I build", ["couch", "table", "lamp", "room", "add", "try"]),
    ("Skippy who made you", None), ("hey skippy sing something", None), ("Skippy do you ever sleep", None),
    ("Skippy what are you", None), ("Skippy thanks that was great", None), ("Skippy good job", None),
    ("skippy how do I get rid of everything", ["clear", "undo", "say"]),
    ("Skippy how do I make a room", ["add", "room", "say", "couch", "build"]),
    ("Skippy are you able to build things for me", ["add", "build", "yes", "couch", "things", "can"]),
    ("hey skippy would a feather fall slower than a brick", ["vacuum", "air", "drag", "same"]),
    ("Skippy tell me something true about water", ["dense", "float", "sink", "weight", "pour", "push"]),
    ("Skippy what time is it", None), ("skippy you are being very annoying", None), ("Skippy do you know my name", None),
]
for i, (text, kw) in enumerate(TALK):
    add("wake_talk", text, "yes", "talk", src="voice" if i % 4 == 0 else "chat", avatar=AVS[(i + 1) % len(AVS)], kw=kw)

for text in ["Skippy make it cozy in here", "skippy I need somewhere to relax", "hey Skippy furnish this place like a cozy cabin", "Skippy, set me up a place to read",
             "skippy I want a vibe like a quiet library", "Skippy make a comfortable spot to watch movies"]:
    add("wake_plan", text, "yes", "plan", avatar=NEAR_SIDE)

# ------------------------------------------------------------------------------------------------ custom names, rude names, other scripts
CUSTOM = [
    (["skippy", "rex"], "Rex add a couch", "parser"), (["rex"], "Rex, come here", "parser"), (["rex"], "hey rex what can you do", "talk"),
    (["skippy", "asshole"], "Asshole add a lamp", "parser"), (["skippy", "ass hole"], "assholee undo", "parser"), (["captain crunch"], "Captain Crunch add a table", "parser"),
    (["captain crunch"], "captain crunch tell me a joke", "talk"), (["dummkopf"], "Dummkopf, put a chair by the window", "parser"),
    (["Скиппи"], "Скиппи, добавь диван", "talk"),   # Cyrillic: the English parser reads no Russian
    (["σκίπι"], "Σκίπι, γεια σου", "talk"),
    (["小明"], "小明，加沙发", "talk"), (["スキッピー"], "スキッピー こんにちは", "talk"),
    (["키피"], "키피 안녕", "talk"), (["monsieur chapeau"], "Monsieur Chapeau add three chairs", "parser"),
    (["bigboss"], "big boss, remove the lamp", "parser"), (["sir can"], "Sir Can, how are you holding up", "talk"),
]
for i, (names, text, route) in enumerate(CUSTOM):
    add("custom_wake", text, "yes", route, src="voice" if i % 2 else "chat", names=names, avatar=AVS[i % len(AVS)])

for text, route in [("Skipy add a lamp", "parser"), ("skippie, undo", "parser"), ("Skeepy put a chair here", "parser"), ("skippee come here", "parser"),
                    ("Skippey, what can you do", "talk"), ("skipi add a rug", "parser")]:
    add("mishear_wake", text, "yes", route, src="voice")

# ------------------------------------------------------------------------------------------------ push-to-talk (+skippy) and the console
for text, route, kw in [("add two couches", "parser", None), ("why is the moon so much lighter feeling", "talk", ["moon", "sixth", "less", "lighter", "gravity", "pull"]),
                        ("undo", "parser", None), ("tell me a joke", "talk", None), ("how do I clear the room", "talk", ["clear", "say", "undo"]),
                        ("put a table in front of me", "parser", None), ("can you hear me", "talk", None), ("make it feel like a ship", "plan", None),
                        ("come over here", "parser", None), ("what can you build", "talk", ["add", "build", "couch", "things", "library"]),
                        ("add a pond in front of me", "parser", None), ("I'm bored", "talk", None), ("go away for a bit", "parser", None), ("remove all the lamps", "parser", None)]:
    add("push_to_talk", text, "yes", route, src="ptt", avatar=AVS[len(turns) % len(AVS)], kw=kw)
for text, route in [("add a couch", "parser"), ("what can you do", "talk"), ("clear everything", "parser"), ("undo", "parser"), ("how are you", "talk"), ("make it cozy", "plan")]:
    add("console", text, "yes", route, src="console")

# ------------------------------------------------------------------------------------------------ follow-ups inside the conversation window
for i, (text, route) in enumerate([
        ("and another one", "parser"), ("make that bigger", "talk"), ("thanks", "talk"), ("no, the other lamp", "talk"), ("why not", "talk"),
        ("can you add a pool then", "parser"), ("what do you mean", "talk"), ("really?", "talk"), ("do it again", "parser"), ("undo that", "parser"),
        ("how come", "talk"), ("okay and a table next to it", "parser"), ("is that true", "talk"), ("you are rude", "talk"), ("put it by the wall", "parser"),
        ("hm, show me another", "parser"), ("what else can you do", "talk"), ("great, now remove the chair", "parser")]):
    add("followup_gaze", text, "yes", route, avatar=[NEAR_LOOK, MID_LOOK][i % 2], since=[3, 6, 12, 16][i % 4])
for text, route in [("and one more of those", "parser"), ("thanks a lot", "talk"), ("now put a lamp over there", "parser"), ("what does that mean", "talk"),
                    ("do you want to explain", "talk"), ("fine, undo it", "parser")]:
    add("followup_near", text, "yes", route, avatar=NEAR_SIDE, since=9)

# ------------------------------------------------------------------------------------------------ two humans talking (never for him)
CHAT = ["did you see that jump", "lol nice one", "where did you get that sword", "I'm going up to the roof", "your turn to build", "bro that was insane",
        "can you hold this for a second", "meet me by the garage", "that map is huge", "I keep falling off the ledge", "what time are we stopping tonight",
        "no way you did that", "go left, go left", "ha, you died again", "do you want to race", "wait for me", "I think the server is lagging", "can you stop pushing me",
        "add me on the friends list", "come here a sec", "come here, look at this", "put the couch over there, it looks better", "make some room, I'm sitting",
        "who turned the lights off", "that's a nice chair", "what do you want to do next", "undo what you just did, it was fine before", "this is so much fun",
        "show me how you did that", "give me a second to think"]
for i, text in enumerate(CHAT):
    add("chatter_humans", text, "no", "ignore", avatar=[None, NEAR_SIDE, MID_SIDE, NEAR_BACK, FAR_BACK][i % 5], chatter=[2, 4, 7, 9][i % 4], near=1,
        look_human=(i % 3 == 0))

# addressed to a named human, by name
for text in ["Bob, add a couch", "Bob can you put a lamp here", "hey Alice come here", "Alice undo that please", "guys come look at this", "everyone get on the roof",
             "hey Bob what can you do", "yo Bob, pour some water", "Alice, go away", "hey guys, add a chair"]:
    add("named_human", text, "no", "ignore", avatar=[None, NEAR_SIDE, MID_SIDE][len(turns) % 3], near=1, chatter=5 if len(turns) % 2 else None)
add("named_human", "xX_Bob_Xx add a lamp", "no", "ignore", humans=["xX_Bob_Xx"], near=1)
add("named_human", "Big Mike come over here", "no", "ignore", humans=["Big Mike", "Alice"], near=1)

# his name used as a topic, or only in passing
for text in ["Skippy is silly", "Skippy is so rude to me", "Skippy keeps floating behind me", "Skippy doesn't like bananas", "skippy always says the same thing",
             "Skippy has a really big hat", "I think Skippy broke the lamp", "did you hear what Skippy said", "have you seen Skippy's hat", "the Skippy mod is cool",
             "tell Bob that Skippy is here", "she says skippy is hilarious"]:
    add("topic_name", text, "no", "ignore", avatar=[None, NEAR_SIDE, MID_SIDE][len(turns) % 3], chatter=3 if len(turns) % 2 else None, near=1)

# ------------------------------------------------------------------------------------------------ noise, empty, hallucinations
for text, src in [("...", "ptt"), ("uh", "ptt"), ("hmm", "ptt"), ("", "chat"), ("   ", "chat"), ("um uh", "ptt"), ("!!!", "chat"), ("ahh", "ptt"),
                  ("thank you", "voice"), ("thanks for watching", "voice"), ("you", "voice"), ("Mm hm.", "ptt")]:
    add("noise", text, "yes" if src == "ptt" else "no", "ignore", src=src)

# ------------------------------------------------------------------------------------------------ follow-ups that must NOT be answered
for i, text in enumerate(["and another one", "thanks", "why not", "make that bigger", "really?", "what do you mean", "put it by the wall", "okay"]):
    add("followup_expired", text, "no", "ignore", avatar=[NEAR_BACK, FAR_BACK, MID_SIDE, NEAR_BACK][i % 4], since=[45, 90, 30, 25][i % 4], look_human=(i % 2 == 0), near=1)
for i, text in enumerate(["can you add one more", "yeah that works", "nice", "and the lamp too", "I like it", "no wait"]):
    add("window_chatter", text, "no", "ignore", avatar=NEAR_SIDE, since=5, chatter=3, near=1, look_human=True)

# ------------------------------------------------------------------------------------------------ genuinely ambiguous: the default is silence
for i, text in enumerate(["and another one", "what is that hat made of", "can you add a table", "show me the water", "undo that", "that is funny", "are you listening",
                          "what do you think"]):
    add("ambiguous", text, "maybe", "ignore", avatar=MID_SIDE, since=14)           # a follow-up while looking elsewhere and standing a few steps off
for text in ["what's that hat made of", "how tall are you", "do you ever stop floating", "what's your favourite colour"]:
    add("ambiguous", text, "maybe", "ignore", avatar=NEAR_LOOK)                    # looking at him and next to him but never said his name and he did not speak

# ------------------------------------------------------------------------------------------------ rambles
add("ramble", "Skippy so I was thinking that maybe we could try a different layout this time because the old one felt cramped and the lamps were all in the wrong place, "
    "so add a couch near the window and also a couple of plants", "yes", "talk", avatar=FAR_BACK, note="the parser cannot read a rambling sentence; the model may answer or ask")
add("ramble", "hey skippy I know this is a weird question but my friend says that if you drop a heavy ball and a light ball from the same height they land at "
    "the same time and I do not believe it, is that true", "yes", "talk", kw=["same", "air", "drag", "vacuum", "true", "yes"])
add("ramble", "so yesterday I was telling Bob about this crazy thing where you pour water on the floor and the couch starts to float and he did not believe me and "
    "then we tried it and it actually worked and we were laughing for ages", "no", "ignore", avatar=NEAR_SIDE, near=1, chatter=6, look_human=True)
add("ramble", "okay so the plan is you go around the back and I will go over the top and we meet in the middle and then we all jump down together and see what happens "
    "when we land on the big pile of crates", "no", "ignore", avatar=None, near=1, chatter=4)
add("ramble", "Skippy this is a really long way of saying that I would like you to add three chairs and a table in the middle of the room and then maybe a lamp in the "
    "corner if that is not too much trouble for you", "yes", "parser", avatar=NEAR_SIDE)
add("ramble", "hey Bob did you see how the water went up when I dropped the big crate in it it was like the whole pool just jumped up and splashed over the edge "
    "and I swear the lamp got wet", "no", "ignore", near=1, avatar=MID_SIDE)

# ------------------------------------------------------------------------------------------------ typos in commands (he is named)
for text, route in [("Skippy ad a coutch", "parser"), ("skippy add a lamp pls", "parser"), ("Skippy udno", "talk"), ("skippy cmoe here", "parser"),
                    ("Skippy, remvoe the chair", "parser"), ("skippy add tow chairs", "parser")]:
    add("typo_command", text, "yes", route, src="voice" if len(turns) % 2 else "chat")

# ------------------------------------------------------------------------------------------------ object requests in question form belong to the resolver; physics questions to the world's readback
for text, route in [("Skippy can you add a swimming pool", "parser"), ("Skippy can you build me a spaceship", "parser"), ("Skippy make me a sandwich", "parser"),
                    ("skippy delete the server", "parser"), ("Skippy what is gravity", "parser"), ("Sir Can what is gravity", "parser"),
                    ("Skippy could you put a lamp next to the couch", "parser"), ("skippy can I have a table", "parser")]:
    add("polite_order", text, "yes", route, names=["skippy", "sir can"] if "Sir Can" in text else None)
add("polite_order", "what's the gravity like on the moon", "yes", "parser", src="ptt")

# ------------------------------------------------------------------------------------------------ the extra things models get asked
for text, kw in [("Skippy what is the speed of light", None),
                 ("skippy are you smarter than me", None), ("Skippy am I a good builder", None), ("skippy where am I", None),
                 ("Skippy how many things have I built", None), ("skippy what is your favourite object", None), ("Skippy can you fly", None),
                 ("Skippy what is two plus two", None), ("skippy I do not understand how this works", ["add", "say", "try", "build"])]:
    add("wake_talk_hard", text, "yes", "talk", src="chat", avatar=NEAR_SIDE, kw=kw)

# ------------------------------------------------------------------------------------------------ more of everything (second pass, written after the first scores were seen)
for text in ["is anyone else seeing the water glitch", "I think we should build the base over there", "what did you name your character", "that ledge is too high, skip it",
             "can we switch teams after this round", "hold on, my game froze", "why is the floor so slippery", "I'll be right back, getting a drink",
             "could someone help me carry this crate", "ok that's enough for tonight", "wow the lamp is flickering", "should we try the roof next"]:
    add("chatter_humans", text, "no", "ignore", avatar=[None, NEAR_SIDE, MID_SIDE, FAR_BACK][len(turns) % 4], chatter=[3, 6, 8][len(turns) % 3], near=1 + len(turns) % 2,
        look_human=(len(turns) % 2 == 0))
for text in ["Alice stop pushing the couch", "hey Bob where did the lamp go", "Bob do you have a minute", "yo guys the ceiling is broken", "Alice can you pass me the wrench",
             "everyone gather round the pool"]:
    add("named_human", text, "no", "ignore", near=1, avatar=NEAR_SIDE, chatter=4)
for text in ["Skippy's hat is bigger than my head", "my friend says Skippy is a beer can", "not sure why skippy does that", "Skippy was really loud earlier",
             "does anybody know why skippy follows me"]:
    add("topic_name", text, "no", "ignore", avatar=MID_SIDE, near=1, chatter=5)
for text, route in [("Skippy, how high can I build a tower", "talk"), ("skippy is it okay to stack ten crates", "talk"), ("hey skippy what happens if I drop a boulder in the pond", "talk"),
                    ("Skippy tell me about yourself", "talk"), ("skippy are you afraid of anything", "talk"), ("Skippy what is your favourite kind of couch", "talk"),
                    ("Skippy whats the plan", "talk"), ("hey skippy you awake", "talk"), ("Skippy I need a hand with the layout", "talk"), ("skippy cheer me up", "talk")]:
    add("wake_talk", text, "yes", route, src="chat" if len(turns) % 2 else "voice", avatar=AVS[len(turns) % len(AVS)])
for text, route in [("Skippy add a bookshelf next to the table", "parser"), ("skippy put four chairs around the table", "parser"), ("Skippy remove every lamp", "parser"),
                    ("hey skippy rotate the couch", "parser"), ("Skippy zero gravity please", "parser"), ("Skippy make the pool deeper", "parser")]:
    add("wake_command", text, "yes", route, avatar=AVS[len(turns) % len(AVS)])
for text in ["and a bigger one", "wait, why did you do that", "can you explain", "that is not what I wanted", "so what now", "do you remember what I asked"]:
    add("followup_gaze", text, "yes", "parser" if text == "and a bigger one" else "talk", avatar=NEAR_LOOK, since=7)
for text in ["do you agree", "what was that", "is that even possible", "ha, good one"]:
    add("ambiguous", text, "maybe", "ignore", avatar=MID_SIDE, since=18)
for text in ["xX_Bob_Xx look at this", "Big Mike did you see that", "Ali I need the red one"]:
    add("named_human", text, "no", "ignore", humans=["xX_Bob_Xx", "Big Mike", "Ali"], near=1, avatar=NEAR_SIDE)

# ------------------------------------------------------------------------------------------------ STRESS SET (written in one sitting, then run once; reported apart)
# Harder, odder cases, labelled from intent before the code was run on them. Anything the rules got wrong here was fixed only if it was a real bug,
# and the README says which. cat starts with "stress_".
S0 = len(turns)
for text, route in [("Skippy!", "talk"), ("SKIPPY ADD A LAMP", "parser"), ("skippy...", "talk"), ("Skippy, hello?", "talk"), ("hey, skippy. you there?", "talk"),
                    ("Skippy: undo", "parser"), ("skippy;add a chair", "parser"), ("  Skippy   add   two   lamps  ", "parser")]:
    add("stress_named", text, "yes", route, avatar=AVS[len(turns) % len(AVS)])
for text in ["Mark the spot with a lamp", "Will you pass me the wrench", "Rex is coming with us", "Pat, add the lamp to the list", "Art is over there"]:
    add("stress_human_names", text, "no", "ignore", humans=["Mark", "Will", "Rex", "Pat", "Art"], near=1, avatar=NEAR_SIDE)
for text in ["add a couch", "undo", "make it cozy", "come here", "what can you do"]:
    add("stress_unnamed_order", text, "no", "ignore", avatar=None, near=1, chatter=3)            # an order with no name, nobody talking to him: not his
for text in ["wait what", "how did you do that", "oh nice", "go on then", "really now"]:
    add("stress_window_other_player", text, "no", "ignore", avatar=NEAR_LOOK, since=None, near=1, chatter=2, look_human=True)     # he spoke to someone else, not to this player
for text in ["so what do you think", "hmm tell me more", "okay but why", "and then what"]:
    add("stress_window_far_away", text, "maybe", "ignore", avatar=FAR_BACK, since=10)            # his last words were for them, but they have walked off and look elsewhere
for text, route in [("thanks skippy", "talk"), ("okay thanks Skippy that is great", "talk"), ("ok cool skippy", "talk")]:
    add("stress_name_last", text, "no", "ignore", avatar=MID_SIDE, near=0)                       # his name at the END is not the start: only the window or a look could answer
for text in ["Skippy", "skippy", "SKIPPY"]:
    add("stress_named", text, "yes", "talk", src="voice")
for text, src, route in [("Skippy add a lamp in the corner please", "voice", "parser"), ("um skippy uh add a table", "voice", "parser"), ("skippy it would be great if you could add a lamp", "voice", "parser"),
                         ("Skippy, why is the sky blue", "chat", "talk"), ("Skippy pretend you are a pirate", "chat", "talk")]:
    add("stress_named", text, "yes", route, src=src, avatar=MID_SIDE)
for text in ["haha look at skippy go", "skippy just ate the table lol", "i told him skippy would do it", "go ask skippy yourself"]:
    add("stress_name_inside", text, "no", "ignore", avatar=NEAR_SIDE, near=1, chatter=4)
for text in ["a couch, a lamp, a table", "123456", "?", "......", "lol lol lol"]:
    add("stress_noise", text, "no", "ignore", src="chat")
for text in ["a table and four chairs please", "can I get a lamp"]:
    add("stress_ptt_orders", text, "yes", "parser", src="ptt")
STRESS = (S0, len(turns))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"version": 1, "note": "original text; see build_turn_corpus.py for the label rules", "turns": turns}, indent=1, ensure_ascii=False), encoding="utf-8")
    cats: dict[str, int] = {}
    for t in turns:
        cats[t["cat"]] = cats.get(t["cat"], 0) + 1
    print(f"wrote {OUT} with {len(turns)} turns", cats)


if __name__ == "__main__":
    main()
