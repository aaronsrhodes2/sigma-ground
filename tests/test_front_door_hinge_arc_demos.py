"""End-to-end gate: the EXACT, unmodified Captain prompts, through Mentat's
single text entry point (front_door.dispatch), with NO Claude-tailored
wording -- this is the actual capability being demonstrated, not a proxy
for it."""
import os

from sigma_ground.mcp import front_door


def test_literal_pliers_prompt_renders_end_to_end():
    text = "Simulate a pair of pliers that are being open and closed over time."
    env = front_door.dispatch(text, mode="render", use_llm=False)
    assert env["intent"] == "render"
    assert env["source"] == "radiance"
    assert env["saved"] is not None
    assert os.path.exists(env["saved"]["path"])
    assert "Rendered" in env["text"]


def test_literal_bell_prompt_renders_end_to_end_with_audio():
    text = ("Simulate a hanging bell being struck by a stone, simulate the "
           "noise in an earth atmosphere.")
    env = front_door.dispatch(text, mode="render", use_llm=False)
    assert env["intent"] == "render"
    assert env["source"] == "radiance"
    assert env["saved"] is not None
    assert os.path.exists(env["saved"]["path"])
    assert "Struck at" in env["text"]
    assert "Audio saved to" in env["text"]
    wav_path = env["text"].split("Audio saved to")[1].strip()
    assert os.path.exists(wav_path)


def test_auto_mode_also_offers_the_render_without_a_forced_flag():
    """The Captain's prompts start with the word 'Simulate' -- confirm the
    UNFORCED (auto-classified, no mode= override) path also recognizes them
    as simulations and offers a render, not just the forced-mode path."""
    session = front_door.Session()
    env = front_door.dispatch(
        "Simulate a pair of pliers that are being open and closed over time.",
        use_llm=False, session=session)
    assert env["intent"] == "simulate"
    assert env["can_render"] is True
    assert session.render_handle is not None
    assert session.render_handle["kind"] == "hand_tool_actuation"
