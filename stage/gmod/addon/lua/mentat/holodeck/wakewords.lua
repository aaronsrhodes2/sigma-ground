-- wakewords.lua -- the names Skippy answers to, one list per player. PURE Lua (no engine calls), loaded by the server (wakenames.lua)
-- and unit-tested offline; skippy_ears.py (the microphone bridge on the player's PC) has a Python twin, wakewords.py, and both are held to
-- the same table of sentences (sigma_ground/radiance/tests/wake_vectors.json), so a name that wakes him in chat wakes him by voice.
--
--   Wake.normalise_name(raw)          -> name | nil, code, why     "  "Ass-Hole" " -> "ass-hole"; 1-3 words, 2-24 CHARACTERS (not bytes), letters/digits/'/-/space
--                                                                    letters of Latin and the scripts listed in LETTERS (Cyrillic, Greek, CJK, Arabic ...);
--                                                                    invalid UTF-8 is refused (code "utf8"); non-Latin names are matched EXACTLY (after case-folding
--                                                                    and NFC), never fuzzily; Latin names keep the conservative fuzzy rule
--   Wake.check_name(name, is_noun)    -> true | false, code, why   refuses names that would break the parser (command words, object nouns)
--   Wake.strip(text, names, opts)     -> woke, rest, name          does the sentence START with one of the names? (never scans the whole sentence)
--   Wake.intent(text)                 -> { kind = "add"|"remove"|"only"|"list"|"reset", name = "<raw capture>" } | nil
--   Wake.display(name)                -> "Ass-Hole"
-- No name is ever refused for being rude or silly: that is the owner's right (Aaron, 2026-10-12). Only names that make ORDINARY
-- sentences wake him (a command verb, an object noun, a number, a pronoun) are refused, and the reason is told.
local Wake = {}
Wake.DEFAULT = "skippy"
Wake.MAX_NAMES, Wake.MIN_LEN, Wake.MAX_LEN, Wake.MAX_WORDS = 8, 2, 24, 3
Wake.HEAD_BYTES, Wake.HEAD_TOKENS, Wake.MAX_SPAN = 160, 14, 4
-- the established spellings of Skippy (what speech recognition makes of it); identical to Parse.WAKE_WORDS
Wake.SKIPPY_VARIANTS = { "skippy", "skipy", "skippie", "skipie", "skipi", "skippi", "skipee", "skippee", "skippey", "skipey", "skeepy",
  "skeepi", "skeepie" }

local function set(s) local t = {}; for w in s:gmatch("%S+") do t[w] = true end; return t end

-- words that may begin an ordinary request or sentence: a wake name equal to one would wake him on talk that is not for him
local RESERVED = set[[
add put remove delete clear undo make build place create spawn give get bring move turn rotate show set erase destroy summon please can could
would will should shall may might must do does did is are was were am be been have has had let lets want need like try just stop start
come go here there away dismiss vanish leave appear follow hide help look see watch tell say ask answer answering respond responding listen
call calling name names forget reset only also stay sit stand open close pick take drop throw push pull lift hold keep cut paint colour color
yes no ok okay hey hi hello yo oi so well um uh er erm hmm oh ah now then and but or if when what who why how where which whose
the a an this that these those my your our its his her their it them they he she we you me us him i mine yours
one two three four five six seven eight nine ten eleven twelve twenty thirty hundred another again more some any every all both each
everything something anything nothing everyone nobody other next last first big small tiny huge little long short tall
left right front back up down over under near far inside outside
couch sofa chair table lamp bed desk door light lights tv television shelf bookshelf stool bench rug plant fridge oven sink toilet bath mirror
pairs skipping skipped skimpy skippers]]
Wake.RESERVED = RESERVED
-- never a name even at the start of a phrase: they make the capture of "answer to my question" a refusal
local NOT_A_NAME = set[[my your this that these those it its me him her them us our their what who how why when where which the a an some any
every all both each another other there here i you he she we they]]
-- words a sentence may begin with before the name ("hey, Rex, ..."): the parser's own list
local LEAD = set[[hey hi hello hiya howdy yo oi ok okay so um umm uh uhh er erm hmm hm well alright all right there please excuse me oh ah now
then and but yes yeah sure]]
-- "Rex's couch is ugly" talks ABOUT Rex: the possessive wakes him only before a word that is plainly an order
local ORDERS = set[[add put remove delete clear undo make build place create spawn give get bring move turn rotate show set erase destroy
summon please can could would will]]

-- ---------------------------------------------------------------------------------------------- folding
local FOLD = {
  ["\195\128"]="a", ["\195\129"]="a", ["\195\130"]="a", ["\195\131"]="a", ["\195\132"]="a", ["\195\133"]="a",
  ["\195\134"]="ae", ["\195\135"]="c", ["\195\136"]="e", ["\195\137"]="e", ["\195\138"]="e", ["\195\139"]="e",
  ["\195\140"]="i", ["\195\141"]="i", ["\195\142"]="i", ["\195\143"]="i", ["\195\144"]="d", ["\195\145"]="n",
  ["\195\146"]="o", ["\195\147"]="o", ["\195\148"]="o", ["\195\149"]="o", ["\195\150"]="o", ["\195\152"]="o",
  ["\195\153"]="u", ["\195\154"]="u", ["\195\155"]="u", ["\195\156"]="u", ["\195\157"]="y", ["\195\158"]="th",
  ["\195\159"]="ss", ["\195\160"]="a", ["\195\161"]="a", ["\195\162"]="a", ["\195\163"]="a", ["\195\164"]="a",
  ["\195\165"]="a", ["\195\166"]="ae", ["\195\167"]="c", ["\195\168"]="e", ["\195\169"]="e", ["\195\170"]="e",
  ["\195\171"]="e", ["\195\172"]="i", ["\195\173"]="i", ["\195\174"]="i", ["\195\175"]="i", ["\195\176"]="d",
  ["\195\177"]="n", ["\195\178"]="o", ["\195\179"]="o", ["\195\180"]="o", ["\195\181"]="o", ["\195\182"]="o",
  ["\195\184"]="o", ["\195\185"]="u", ["\195\186"]="u", ["\195\187"]="u", ["\195\188"]="u", ["\195\189"]="y",
  ["\195\190"]="th", ["\195\191"]="y", ["\196\128"]="a", ["\196\129"]="a", ["\196\130"]="a", ["\196\131"]="a",
  ["\196\132"]="a", ["\196\133"]="a", ["\196\134"]="c", ["\196\135"]="c", ["\196\136"]="c", ["\196\137"]="c",
  ["\196\138"]="c", ["\196\139"]="c", ["\196\140"]="c", ["\196\141"]="c", ["\196\142"]="d", ["\196\143"]="d",
  ["\196\144"]="d", ["\196\145"]="d", ["\196\146"]="e", ["\196\147"]="e", ["\196\148"]="e", ["\196\149"]="e",
  ["\196\150"]="e", ["\196\151"]="e", ["\196\152"]="e", ["\196\153"]="e", ["\196\154"]="e", ["\196\155"]="e",
  ["\196\156"]="g", ["\196\157"]="g", ["\196\158"]="g", ["\196\159"]="g", ["\196\160"]="g", ["\196\161"]="g",
  ["\196\162"]="g", ["\196\163"]="g", ["\196\164"]="h", ["\196\165"]="h", ["\196\168"]="i", ["\196\169"]="i",
  ["\196\170"]="i", ["\196\171"]="i", ["\196\172"]="i", ["\196\173"]="i", ["\196\174"]="i", ["\196\175"]="i",
  ["\196\176"]="i", ["\196\177"]="i", ["\196\178"]="ij", ["\196\179"]="ij", ["\196\180"]="j", ["\196\181"]="j",
  ["\196\182"]="k", ["\196\183"]="k", ["\196\185"]="l", ["\196\186"]="l", ["\196\187"]="l", ["\196\188"]="l",
  ["\196\189"]="l", ["\196\190"]="l", ["\196\191"]="l", ["\197\128"]="l", ["\197\129"]="l", ["\197\130"]="l",
  ["\197\131"]="n", ["\197\132"]="n", ["\197\133"]="n", ["\197\134"]="n", ["\197\135"]="n", ["\197\136"]="n",
  ["\197\137"]="n", ["\197\140"]="o", ["\197\141"]="o", ["\197\142"]="o", ["\197\143"]="o", ["\197\144"]="o",
  ["\197\145"]="o", ["\197\146"]="oe", ["\197\147"]="oe", ["\197\148"]="r", ["\197\149"]="r", ["\197\150"]="r",
  ["\197\151"]="r", ["\197\152"]="r", ["\197\153"]="r", ["\197\154"]="s", ["\197\155"]="s", ["\197\156"]="s",
  ["\197\157"]="s", ["\197\158"]="s", ["\197\159"]="s", ["\197\160"]="s", ["\197\161"]="s", ["\197\162"]="t",
  ["\197\163"]="t", ["\197\164"]="t", ["\197\165"]="t", ["\197\168"]="u", ["\197\169"]="u", ["\197\170"]="u",
  ["\197\171"]="u", ["\197\172"]="u", ["\197\173"]="u", ["\197\174"]="u", ["\197\175"]="u", ["\197\176"]="u",
  ["\197\177"]="u", ["\197\178"]="u", ["\197\179"]="u", ["\197\180"]="w", ["\197\181"]="w", ["\197\182"]="y",
  ["\197\183"]="y", ["\197\184"]="y", ["\197\185"]="z", ["\197\186"]="z", ["\197\187"]="z", ["\197\188"]="z",
  ["\197\189"]="z", ["\197\190"]="z", ["\197\191"]="s",
}
local QUOTES = { ["\226\128\152"] = "'", ["\226\128\153"] = "'", ["\226\128\155"] = "'", ["\202\188"] = "'", ["\226\128\178"] = "'",
  ["\194\180"] = "'", ["\226\128\156"] = '"', ["\226\128\157"] = '"', ["\194\171"] = '"', ["\194\187"] = '"',
  ["\226\128\147"] = "-", ["\226\128\148"] = "-", ["\194\160"] = " ", ["\226\128\166"] = "." }

-- ---- UTF-8, by hand (Lua 5.1 has no utf8 library): strict decoding, so text that is not valid UTF-8 is never taken for a name.
local function decode_at(s, i)           -- -> codepoint, byte length | nil (overlong, surrogate, past U+10FFFF, stray or missing continuation)
  local b = s:byte(i)
  if b < 0x80 then return b, 1 end
  if b < 0xC2 or b > 0xF4 then return nil end
  local len = b >= 0xF0 and 4 or b >= 0xE0 and 3 or 2
  local cp = b % (len == 2 and 0x20 or len == 3 and 0x10 or 0x08)
  for k = 1, len - 1 do
    local c = s:byte(i + k)
    if not c or c < 0x80 or c > 0xBF then return nil end
    cp = cp * 64 + (c - 0x80)
  end
  if (len == 3 and cp < 0x800) or (len == 4 and (cp < 0x10000 or cp > 0x10FFFF)) or (cp >= 0xD800 and cp <= 0xDFFF) then return nil end
  return cp, len
end

local function utf8_char(cp)
  if cp < 0x80 then return string.char(cp) end
  if cp < 0x800 then return string.char(0xC0 + math.floor(cp / 64), 0x80 + cp % 64) end
  if cp < 0x10000 then return string.char(0xE0 + math.floor(cp / 4096), 0x80 + math.floor(cp / 64) % 64, 0x80 + cp % 64) end
  return string.char(0xF0 + math.floor(cp / 262144), 0x80 + math.floor(cp / 4096) % 64, 0x80 + math.floor(cp / 64) % 64, 0x80 + cp % 64)
end

--- Is every byte of s part of well-formed UTF-8?
function Wake.valid_utf8(s)
  s = tostring(s or "")
  local i, n = 1, #s
  while i <= n do
    local cp, l = decode_at(s, i)
    if not cp then return false end
    i = i + l
  end
  return true
end

--- Characters (not bytes) of well-formed UTF-8: the bytes that are not continuation bytes.
function Wake.length(s) return (select(2, tostring(s):gsub("[^\128-\191]", ""))) end

-- ---- letters outside Latin. A name may be written in any of these scripts (exact match only: no fuzzy matching for them).
-- Latin-script letters are handled by FOLD above; a script not listed here is refused (with the reason), never half-supported.
-- Hex ranges, identical in wakewords.py (the two are held to the same vectors).
local LETTERS = { {0x386,0x386},{0x388,0x38A},{0x38C,0x38C},{0x38E,0x3A1},{0x3A3,0x3CE},          -- Greek
  {0x400,0x481},{0x48A,0x52F},                                                                  -- Cyrillic
  {0x531,0x556},{0x561,0x586},                                                                  -- Armenian
  {0x5D0,0x5EA},                                                                                -- Hebrew
  {0x620,0x63F},{0x641,0x65F},{0x66E,0x66F},{0x670,0x6D3},{0x6FA,0x6FC},                        -- Arabic (with its vowel marks)
  {0x900,0x939},{0x93A,0x94D},{0x958,0x961},                                                    -- Devanagari
  {0x985,0x9B9},{0x9BC,0x9CC},{0xB85,0xBB9},{0xBBE,0xBCD},                                      -- Bengali, Tamil
  {0xE01,0xE3A},{0xE40,0xE4E},                                                                  -- Thai
  {0x10D0,0x10FA},{0x10FD,0x10FF},{0x1100,0x11FF},{0x1200,0x135A},                              -- Georgian, Hangul jamo, Ethiopic
  {0x3041,0x3096},{0x309D,0x309F},{0x30A1,0x30FA},{0x30FC,0x30FF},{0x3105,0x312F},              -- Hiragana, Katakana, Bopomofo
  {0x3400,0x4DBF},{0x4E00,0x9FFF},{0x20000,0x2FA1D},{0xAC00,0xD7A3} }                           -- Han, Hangul syllables
local function is_letter(cp)
  for _, r in ipairs(LETTERS) do
    if cp >= r[1] and cp <= r[2] then return true end
  end
  return false
end

--- The lower-case form of a letter of a script that has case (Greek, Cyrillic, Armenian); every other codepoint unchanged.
local function lower_cp(cp)
  if cp >= 0x410 and cp <= 0x42F then return cp + 32 end
  if cp >= 0x400 and cp <= 0x40F then return cp + 80 end
  if cp >= 0x391 and cp <= 0x3AB and cp ~= 0x3A2 then return cp + 32 end
  if cp == 0x3C2 then return 0x3C3 end                                      -- final sigma
  if cp == 0x386 then return 0x3AC end
  if cp >= 0x388 and cp <= 0x38A then return cp + 37 end
  if cp == 0x38C then return 0x3CC end
  if cp == 0x38E or cp == 0x38F then return cp + 63 end
  if cp >= 0x531 and cp <= 0x556 then return cp + 48 end
  if (cp >= 0x460 and cp <= 0x481) or (cp >= 0x48A and cp <= 0x4BF) or (cp >= 0x4D0 and cp <= 0x52F) then
    return cp % 2 == 0 and cp + 1 or cp
  end
  if cp == 0x4C0 then return 0x4CF end
  if cp >= 0x4C1 and cp <= 0x4CE then return cp % 2 == 1 and cp + 1 or cp end
  return cp
end

-- ---- NFC for the scripts above: a base letter and its combining mark become the one precomposed letter, so a name typed on a
-- keyboard that decomposes (macOS) equals the same name typed precomposed. Hangul jamo join into syllables; kana take their dakuten.
local COMPOSE = {}
local function pair(base, mark, to) COMPOSE[base * 0x200000 + mark] = to end
pair(0x438, 0x306, 0x439); pair(0x435, 0x308, 0x451); pair(0x456, 0x308, 0x457); pair(0x443, 0x306, 0x45E)   -- Cyrillic: y-breve, yo, yi, short u
pair(0x438, 0x300, 0x45D); pair(0x435, 0x300, 0x450); pair(0x433, 0x301, 0x453); pair(0x43A, 0x301, 0x45C)
local TONOS_BASE, TONOS_TO = { 0x3B1, 0x3B5, 0x3B7, 0x3B9, 0x3BF, 0x3C5, 0x3C9 }, { 0x3AC, 0x3AD, 0x3AE, 0x3AF, 0x3CC, 0x3CD, 0x3CE }
for i, base in ipairs(TONOS_BASE) do pair(base, 0x301, TONOS_TO[i]) end                                        -- Greek tonos
pair(0x3B9, 0x308, 0x3CA); pair(0x3C5, 0x308, 0x3CB)
for _, h in ipairs({ 0x304B,0x304D,0x304F,0x3051,0x3053,0x3055,0x3057,0x3059,0x305B,0x305D,0x305F,0x3061,0x3064,0x3066,0x3068,0x306F,0x3072,0x3075,0x3078,0x307B }) do
  for _, off in ipairs({ 0, 0x60 }) do pair(h + off, 0x3099, h + off + 1) end                                -- ka + dakuten = ga (and katakana)
end
pair(0x3046, 0x3099, 0x3094); pair(0x30A6, 0x3099, 0x30F4)
for _, h in ipairs({ 0x306F, 0x3072, 0x3075, 0x3078, 0x307B }) do
  for _, off in ipairs({ 0, 0x60 }) do pair(h + off, 0x309A, h + off + 2) end                                -- ha + handakuten = pa
end
local function compose(prev, cp)
  local got = COMPOSE[prev * 0x200000 + cp]
  if got then return got end
  if prev >= 0x1100 and prev <= 0x1112 and cp >= 0x1161 and cp <= 0x1175 then                                 -- Hangul L+V
    return 0xAC00 + ((prev - 0x1100) * 21 + (cp - 0x1161)) * 28
  end
  if prev >= 0xAC00 and prev <= 0xD7A3 and (prev - 0xAC00) % 28 == 0 and cp >= 0x11A8 and cp <= 0x11C2 then   -- LV+T
    return prev + (cp - 0x11A7)
  end
end

--- The characters of raw[from..last] (a character is in if it STARTS at or before `last`), each already in plain form:
---   { piece = plain text ("\1" for anything that is not a letter, digit, apostrophe, hyphen or space), cp = lower-cased codepoint of a
---     non-Latin letter (else nil), s, e = its first and last byte in raw }.
--- ASCII is lower-cased, a typographic quote or dash is its ASCII twin, an accented Latin letter is without its accent (a combining
--- accent after a Latin letter is dropped), a Greek / Cyrillic / Armenian letter is lower-cased and composed, any other letter of LETTERS
--- is kept as it is, and anything else (an emoji, bad UTF-8) becomes "\1".
local function chars(raw, from, last)
  local out = {}
  local i, n = from, math.min(#raw, last)
  while i <= n do
    local cp, l = decode_at(raw, i)
    local piece, lcp
    if not cp then
      l, piece = 1, "\1"
    elseif cp < 0x80 then
      local ch = string.char(cp)
      piece = ch == "`" and "'" or ch:lower()
    else
      local ch = raw:sub(i, i + l - 1)
      piece = QUOTES[ch] or FOLD[ch]
      if not piece then
        local lo = lower_cp(cp)
        local prev = out[#out]
        if (cp >= 0x300 and cp <= 0x36F) or cp == 0x3099 or cp == 0x309A then          -- a combining mark
          local c = prev and prev.cp and compose(prev.cp, cp)
          if c then
            prev.cp, prev.piece, prev.e = c, utf8_char(c), i + l - 1
            piece = false
          elseif prev and cp <= 0x36F and prev.piece:find("^%a+$") then
            prev.e = i + l - 1                                                       -- "e" + acute accent: the accent is dropped, like the letter e-acute
            piece = false
          else
            piece = "\1"
          end
        elseif is_letter(lo) then
          local c = prev and prev.cp and compose(prev.cp, lo)                        -- Hangul jamo join
          if c then
            prev.cp, prev.piece, prev.e = c, utf8_char(c), i + l - 1
            piece = false
          else
            piece, lcp = utf8_char(lo), lo
          end
        else
          piece = "\1"
        end
      end
    end
    if piece then out[#out + 1] = { piece = piece, cp = lcp, s = i, e = i + l - 1 } end
    i = i + l
  end
  return out
end

--- Text -> its plain form: what Wake.strip and Wake.intent compare.
function Wake.fold(s)
  s = tostring(s or "")
  local out = {}
  for k, c in ipairs(chars(s, 1, #s)) do out[k] = c.piece end
  return table.concat(out)
end

local function osa(a, b)                       -- edit distance, an adjacent swap counted once
  local la, lb = #a, #b
  local d = {}
  for i = 0, la do d[i] = { [0] = i } end
  for j = 0, lb do d[0][j] = j end
  for i = 1, la do
    for j = 1, lb do
      local cost = a:byte(i) == b:byte(j) and 0 or 1
      local v = math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
      if i > 1 and j > 1 and a:byte(i) == b:byte(j - 1) and a:byte(i - 1) == b:byte(j) then v = math.min(v, d[i - 2][j - 2] + 1) end
      d[i][j] = v
    end
  end
  return d[la][lb]
end
Wake.osa = osa

-- ---------------------------------------------------------------------------------------------- names
local function compact_of(name) return (name:gsub("[%s%-']", "")) end

--- A name as a player typed or said it -> its stored form, or nil with a code and the reason in words.
function Wake.normalise_name(raw)
  raw = tostring(raw or "")
  if not Wake.valid_utf8(raw) then return nil, "utf8", "that is not valid text (a name must be UTF-8)" end
  local s = Wake.fold(raw)
  s = s:gsub("^[%s'\"%-]+", ""):gsub("[%s'\"%-%.,!?;:]+$", "")
  if s:find("\1", 1, true) or s:find("[^%a%d' %-\128-\255]") then
    return nil, "chars", "a name can only use letters (in the Latin, Greek, Cyrillic, Armenian, Hebrew, Arabic, Indic, Thai, Georgian, Korean, Japanese or Chinese scripts), digits, apostrophes, hyphens and spaces"
  end
  s = s:gsub("%s+", " ")
  if not s:find("[%w\128-\255]") then return nil, "empty", "that is not a name" end
  local len = Wake.length(s)                                     -- characters, not bytes
  if len < Wake.MIN_LEN then return nil, "short", "a name needs at least " .. Wake.MIN_LEN .. " characters" end
  if len > Wake.MAX_LEN then return nil, "long", "a name can be at most " .. Wake.MAX_LEN .. " characters" end
  local words = 0
  for _ in s:gmatch("%S+") do words = words + 1 end
  if words > Wake.MAX_WORDS then return nil, "words", "a name can be at most " .. Wake.MAX_WORDS .. " words" end
  return s
end

--- Would this name make ordinary talk wake him? Only that is refused (never rudeness, never silliness).
function Wake.check_name(name, is_noun)
  local words = {}
  for w in name:gsub("%-", " "):gmatch("[%w'\128-\255]+") do words[#words + 1] = (w:gsub("'", "")) end
  if #words == 0 then return false, "empty", "that is not a name" end
  if NOT_A_NAME[words[1]] then return false, "pronoun", "that is not a name; " .. words[1] .. " begins ordinary sentences" end
  local function reserved(w) return RESERVED[w] or (is_noun and is_noun(w)) or false end
  local all = true
  for _, w in ipairs(words) do if not reserved(w) then all = false end end
  if all or reserved(compact_of(name)) then
    return false, "reserved", name .. " is a word my parser needs (a command, an object or a little word of requests), so every ordinary sentence would wake me"
  end
  return true
end

function Wake.display(name)
  return (name:gsub("(%a)([%w']*)", function(a, b) return a:upper() .. b end))
end

-- ---------------------------------------------------------------------------------------------- matching
local function allowed(len) return math.min(3, math.floor(len / 5)) end      -- slips tolerated: none under 5 letters, 1 from 5, 2 from 10

--- The head of a sentence as words: { word = "rex", compact = "rex", poss = true|nil, s = first byte, e = last byte } (positions in the text).
local function tokens(raw, from)
  local toks, cur = {}, nil
  for _, c in ipairs(chars(raw, from, from + Wake.HEAD_BYTES)) do
    if c.piece:find("^[%a%d'\128-\255]+$") then
      if not cur then cur = { text = "", s = c.s }; toks[#toks + 1] = cur end
      cur.text = cur.text .. c.piece
      cur.e = c.e
    else
      cur = nil
    end
  end
  local out = {}
  for _, t in ipairs(toks) do
    local w = t.text:gsub("^'+", ""):gsub("'+$", "")
    local base = w:match("^(.-)'s$")
    if base and base ~= "" then t.poss, w = true, base end
    if w ~= "" and #out < Wake.HEAD_TOKENS then
      t.word, t.compact = w, (w:gsub("'", ""))
      out[#out + 1] = t
    end
  end
  return out
end

-- a name with a letter outside ASCII (Cyrillic, Chinese, ...) is matched exactly, never fuzzily: edit distance means nothing across scripts
local function entry_of(name, exact_only)
  local compact = compact_of(name)
  return { name = name, compact = compact, exact_only = exact_only or compact:find("[\128-\255]") ~= nil }
end

-- how many words (and whether exactly) the sentence's words from i spell this name; nil when they do not
local function match_at(toks, i, e, is_noun)
  local acc = ""
  for k = 1, math.min(Wake.MAX_SPAN, #toks - i + 1) do
    acc = acc .. toks[i + k - 1].compact
    if acc == e.compact then return k, true end
    if #acc > #e.compact + 3 then break end
  end
  local d = allowed(#e.compact)
  if e.exact_only or d == 0 then return nil end
  acc = ""
  local best_k, best_d
  for k = 1, math.min(Wake.MAX_SPAN, #toks - i + 1) do
    acc = acc .. toks[i + k - 1].compact
    if acc:sub(1, 1) == e.compact:sub(1, 1) and math.abs(#acc - #e.compact) <= d and not acc:find("[\128-\255]")
       and not RESERVED[acc] and not (is_noun and is_noun(acc)) then
      local dist = osa(acc, e.compact)
      if dist <= d and (not best_d or dist < best_d) then best_k, best_d = k, dist end
    end
  end
  if best_k then return best_k, false end
end

local function tidy(rest)                       -- drop what separates the name from the request: every leading character that is not a letter or digit
  local i = 1
  while i <= #rest do
    local c = chars(rest, i, i)[1]
    if c and c.piece:find("^[%a%d\128-\255]+$") then break end
    i = (c and c.e or i) + 1
  end
  return rest:sub(i)
end

--- Does the sentence start with one of `names` (stored forms)? Returns (true, the request without the name, the name) or (false, text).
--- A few little words may come first ("hey, ..."); the possessive ("Rex's couch is ugly") only wakes him before an order.
--- opts.default_strip(text) -> woke, rest: the established matcher for "skippy" (Parse.strip_wake); opts.is_noun(word) -> bool.
function Wake.strip(text, names, opts)
  opts = opts or {}
  local raw = tostring(text or "")
  local from = raw:find("%S") or 1
  names = names or { Wake.DEFAULT }
  local entries, has_default = {}, false
  for _, nm in ipairs(names) do
    if nm == Wake.DEFAULT then
      has_default = true
      if not opts.default_strip then
        for _, v in ipairs(Wake.SKIPPY_VARIANTS) do entries[#entries + 1] = entry_of(v, true); entries[#entries].name = Wake.DEFAULT end
      end
    else
      entries[#entries + 1] = entry_of(nm)
    end
  end
  if has_default and opts.default_strip then
    local ok, rest = opts.default_strip(raw)
    if ok then return true, rest, Wake.DEFAULT end
  end
  if #entries == 0 then return false, raw end
  local toks = tokens(raw, from)
  for lead = 0, 3 do
    if lead > 0 and not (toks[lead] and LEAD[toks[lead].word]) then break end
    local i = lead + 1
    if not toks[i] then break end
    local best
    for _, e in ipairs(entries) do
      local k, exact = match_at(toks, i, e, opts.is_noun)
      if k and (not best or (exact and not best.exact) or (exact == best.exact and k > best.k)) then best = { k = k, exact = exact, name = e.name } end
    end
    if best then
      local last = toks[i + best.k - 1]
      local after = raw:sub(last.e + 1)
      if last.poss then
        local nw = after:match("^[^%w]*(%w+)")
        if nw and ORDERS[nw:lower()] then return true, tidy(after), best.name end
        return false, raw
      end
      return true, tidy(after), best.name
    end
  end
  return false, raw
end

-- ---------------------------------------------------------------------------------------------- the spoken commands
-- Anything not matching a whole pattern is NOT a command, so a sentence about names is never mistaken for one.
local FILLERS = { "please ", "kindly ", "hey ", "ok ", "okay ", "so ", "and ", "now ", "from now on ", "can you ", "could you ", "would you ",
  "will you ", "i want you to ", "i would like you to ", "i'd like you to ", "i need you to ", "you should ", "you can ", "you may ",
  "go ahead and ", "just " }
local TAILS = { " please", " thanks", " thank you", " for me", " anymore", " any more", " any longer", " now", " too", " as well", " also",
  " instead", " as a name", " as another name", " as a wake word", " as a wakeword", " as an extra name" }
local CANON = { "answer", "answering", "respond", "responding", "listen", "listening", "forget", "reset", "remove", "delete", "yourself", "names" }

local function fix_typos(t)                          -- "anwser to rex": one slip in a keyword of the first three words
  local words = {}
  for w in t:gmatch("%S+") do words[#words + 1] = w end
  for i = 1, math.min(3, #words) do
    local w = words[i]
    if w == "to" or w == "from" or w == "name" or w == "called" or w == "as" then break end     -- what follows is the name itself
    if #w >= 5 then
      for _, c in ipairs(CANON) do
        if w ~= c and w:sub(1, 1) == c:sub(1, 1) and math.abs(#w - #c) <= 1 and osa(w, c) <= 1 then words[i] = c; break end
      end
    end
  end
  return table.concat(words, " ")
end

local ADD = { "^also answer to (.+)$", "^answer to (.+)$", "^also respond to (.+)$", "^respond to (.+)$", "^also listen to (.+)$", "^listen to (.+)$",
  "^call yourself (.+)$", "^also call yourself (.+)$", "^go by (.+)$", "^also go by (.+)$", "^your name is also (.+)$", "^your name is (.+)$",
  "^you are also called (.+)$", "^you are also known as (.+)$", "^you are called (.+)$", "^you are also named (.+)$",
  "^also be called (.+)$", "^be called (.+)$", "^add the name (.+)$", "^add a name (.+)$", "^add the wake ?words? (.+)$", "^add name (.+)$",
  "^add (.+) to your names$", "^add (.+) to your wake ?words$", "^add (.+) to your list$", "^use (.+) as a name$", "^use (.+) as a wake ?word$",
  "^your other name is (.+)$", "^another name for you is (.+)$" }
local ONLY = { "^only answer to (.+)$", "^only respond to (.+)$", "^only listen to (.+)$", "^answer only to (.+)$", "^respond only to (.+)$",
  "^change your wake ?word to (.+)$", "^change your name to (.+)$", "^only call yourself (.+)$", "^only go by (.+)$", "^only be called (.+)$",
  "^your only name is (.+)$", "^your name is only (.+)$", "^answer to (.+) only$", "^respond to (.+) only$" }
local REMOVE = { "^stop answering to (.+)$", "^stop responding to (.+)$", "^stop listening to (.+)$", "^do not answer to (.+)$",
  "^dont answer to (.+)$", "^don't answer to (.+)$", "^do not respond to (.+)$", "^dont respond to (.+)$", "^don't respond to (.+)$",
  "^no longer answer to (.+)$", "^no longer respond to (.+)$", "^forget the name (.+)$", "^forget your name (.+)$", "^forget the wake ?word (.+)$",
  "^forget name (.+)$", "^remove the name (.+)$", "^remove your name (.+)$", "^remove the wake ?word (.+)$", "^delete the name (.+)$",
  "^drop the name (.+)$", "^remove name (.+)$", "^remove (.+) from your names$", "^remove (.+) from your wake ?words$",
  "^remove (.+) from your list$", "^delete (.+) from your names$", "^drop (.+) from your names$", "^stop being called (.+)$",
  "^stop answering to the name (.+)$" }
local LIST = { "^what do you answer to$", "^what are your names$", "^what names do you answer to$", "^what names do you respond to$",
  "^what other names do you answer to$", "^what do you respond to$", "^which names do you answer to$", "^list your names$",
  "^what are your wake ?words$", "^what is your wake ?word$", "^what can i call you$", "^tell me your names$", "^what names do you have$",
  "^what names do you use$", "^what do you listen to$", "^who do you answer to$", "^what are you called$", "^what are you called by me$",
  "^what names do you answer to for me$", "^what do you answer to for me$", "^show me your names$", "^show your names$" }
local RESET = { "^reset your names?$", "^reset your wake ?words?$", "^reset the names$", "^reset names$", "^forget all your names$",
  "^forget your other names$", "^forget all of your names$", "^go back to skippy$", "^go back to being skippy$", "^go back to your name$",
  "^reset your name to skippy$", "^be skippy again$", "^only answer to skippy again$", "^reset yourself to skippy$" }

local function matches(list, t) for _, p in ipairs(list) do if t:match(p) then return true end end end

--- A sentence that manages the names -> { kind =, name = <the words after it, plain> }, or nil.
function Wake.intent(text)
  local t = Wake.fold(tostring(text or ""):sub(1, 400))        -- (a command is short: never fold a flood)
  t = t:gsub("[\"%.,!?;:]", " "):gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
  if #t > 160 or t == "" then return nil end
  local again = true
  while again do                                                    -- peel the polite lead-ins, in any order
    again = false
    for _, f in ipairs(FILLERS) do
      if t:sub(1, #f) == f then t = t:sub(#f + 1); again = true end
    end
  end
  t = fix_typos(t)
  if matches(RESET, t) then return { kind = "reset" } end
  if matches(LIST, t) then return { kind = "list" } end
  local function peel(cap)
    local more = true
    while more do
      more = false
      for _, tl in ipairs(TAILS) do
        if #cap > #tl and cap:sub(-#tl) == tl then cap = cap:sub(1, -#tl - 1); more = true end
      end
    end
    cap = cap:gsub("^the name ", ""):gsub("^name ", ""):gsub("^the wake ?word ", ""):gsub("^to ", "")
    return cap
  end
  for _, group in ipairs({ { "only", ONLY }, { "remove", REMOVE }, { "add", ADD } }) do
    for _, p in ipairs(group[2]) do
      local cap = t:match(p)
      if cap then return { kind = group[1], name = peel(cap) } end
    end
  end
  -- asked, but no name given
  if t == "answer to" or t == "respond to" or t == "call yourself" or t == "also answer to" then return { kind = "add", name = "" } end
  if t == "stop answering to" or t == "forget the name" then return { kind = "remove", name = "" } end
  if t == "only answer to" then return { kind = "only", name = "" } end
  return nil
end

return Wake
