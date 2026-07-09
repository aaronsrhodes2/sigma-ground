"""The durable conversation store (Phase 0 of the V2 UI). Pure file IO, so each
test repoints the store at a tmp dir and exercises record/load/delete plus the
restart-survival guarantee (every read hits disk; nothing is cached in memory)."""
from sigma_ground.radiance.web import library


def _render_env(slug="falling_feather"):
    return {"intent": "render", "text": f"Rendered “{slug}” — saved for replay.",
            "saved": {"slug": slug, "title": slug.replace("_", " "),
                      "url": f"http://127.0.0.1:8765/?scene={slug}"},
            "source": "radiance"}


def test_record_turn_persists_and_links_render(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "_LIB_DIR", str(tmp_path))
    sid = "tab.1234"                                  # a '.' the filename must survive
    cid = library.record_turn(sid, user_text="drop a feather from 8 ft",
                              mode="render", envelope=_render_env())
    assert cid == "tab_1234"                          # sanitised, idempotent
    conv = library.load_conversation(sid)             # raw id resolves to the same file
    assert conv["title"].startswith("drop a feather")
    assert [m["role"] for m in conv["messages"]] == ["user", "mentat"]
    assert conv["renders"] == ["falling_feather"]
    idx = library.load_index()
    assert len(idx) == 1 and idx[0]["id"] == "tab_1234"
    assert idx[0]["renders"] == ["falling_feather"] and idx[0]["turns"] == 1


def test_append_dedupes_render_and_orders_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "_LIB_DIR", str(tmp_path))
    library.record_turn("a", user_text="hello", mode="ask",
                        envelope={"intent": "ask", "text": "hi", "value": None})
    library.record_turn("b", user_text="drop a feather", mode="render",
                        envelope=_render_env())
    library.record_turn("b", user_text="yes", mode="render", envelope=_render_env())
    conv = library.load_conversation("b")
    assert len(conv["messages"]) == 4                 # two user+mentat exchanges
    assert conv["renders"] == ["falling_feather"]     # linked once, not twice
    idx = library.load_index()
    assert [s["id"] for s in idx] == ["b", "a"]       # b touched last → first
    assert idx[0]["turns"] == 2


def test_restart_survival_reads_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "_LIB_DIR", str(tmp_path))
    library.record_turn("s", user_text="escape velocity of mars", mode="ask",
                        envelope={"intent": "ask", "text": "5.03 km/s", "value": 5027.0})
    # A fresh process consults no in-memory state — load_* re-reads the files.
    # There is no cache to invalidate; that IS the restart guarantee.
    assert library.load_index()[0]["title"].startswith("escape velocity")
    assert library.load_conversation("s")["messages"][1]["value"] == 5027.0


def test_upsert_then_first_turn_promotes_title(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "_LIB_DIR", str(tmp_path))
    library.upsert_conversation("c")                  # the rail's "+ New"
    assert library.load_conversation("c")["title"] == "New conversation"
    library.record_turn("c", user_text="pH of 0.01 M HCl", mode="ask",
                        envelope={"intent": "ask", "text": "2.0"})
    assert library.load_conversation("c")["title"].startswith("pH of")  # placeholder promoted
    library.upsert_conversation("c", title="Acids")   # explicit rename sticks
    assert library.load_conversation("c")["title"] == "Acids"


def test_delete_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "_LIB_DIR", str(tmp_path))
    library.record_turn("x", user_text="hi", mode="ask",
                        envelope={"intent": "ask", "text": "hello"})
    assert library.load_index()
    assert library.delete_conversation("x") is True
    assert library.load_index() == []
    assert library.load_conversation("x") is None
    assert library.delete_conversation("x") is True   # already gone — no error
