"""МКР, варіант 2, завдання 3. Мовна модель як помічник, а не джерело істини.

Запуск:
    export GEMINI_API_KEY=...          # Windows: set GEMINI_API_KEY=...
    python llm_task.py graph.ttl graph_full.ttl

Ключ береться лише зі змінної середовища (GEMINI_API_KEY або GOOGLE_API_KEY)
і передається в заголовку, тому не потрапляє ні в код, ні в логи.
Модель можна змінити змінною GEMINI_MODEL (типово gemini-3.1-flash-lite).
"""
import hashlib
import os
import re
import sys
import time
from datetime import datetime

import requests
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
from rdflib.plugins.sparql import prepareQuery

import build_graph as bg

EX = bg.EX
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_FIX_RETRIES = 2
PARAMS_FILE = "params.json"

TEXT = (
    "Нічний поїзд TR-800 компанії NightLine вирушає з Відня 17 липня 2026 року "
    "о 23:10 і прибуває до Кракова. Дорога займає 5 годин 30 хвилин, квиток коштує "
    "31 євро. Денний рейс TR-801 компанії AlpEx їде з Праги до Відня 20.07.2026, "
    "виїзд о 08:00, у дорозі 4 години 5 хвилин; ціну не вказано."
)
QUESTION = "Які перевізники мають і нічні, і експрес-поїзди?"

# Еталон: те, що за правилами Завдання 1 має вийти з тексту (ручне прочитання тексту).
EXPECTED_ROWS = [
    {"id": "TR-800", "operator": "NightLine", "from": "Vienna", "to": "Krakow",
     "date": "2026-07-17", "time": "23:10", "duration_min": "330",
     "price_eur": "31", "note": ""},
    {"id": "TR-801", "operator": "AlpEx", "from": "Prague", "to": "Vienna",
     "date": "20.07.2026", "time": "08:00", "duration_min": "245",
     "price_eur": "", "note": ""},
]

FORBIDDEN = ("INSERT", "DELETE", "DROP", "CLEAR", "LOAD", "SERVICE")
TASK2_PREFIXES = ("Q1:", "Q2:", "Q3:", "Примітка:")
TASK3_PREFIXES = ("А:", "Б:", "В:")

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(msg)


# ------------------------------------------------------------------ схема
def schema_text(params):
    lines = [
        "Простір імен: PREFIX ex: <http://example.org/mkr/> ; xsd: <http://www.w3.org/2001/XMLSchema#>.",
        "Класи: ex:Train, ex:ExpressTrain (rdfs:subClassOf ex:Train), "
        "ex:NightTrain (rdfs:subClassOf ex:Train), ex:Operator, ex:City, ex:Country, ex:Source.",
        f"ex:ExpressTrain — є ціна і тривалість <= {params['express_max_min']} хв; "
        f"ex:NightTrain — час відправлення >= {params['night_from']} або < {params['night_until']}; "
        "поїзд може мати обидва типи; ex:Train — лише якщо жоден не підходить.",
        "Властивості (domain -> range):",
    ]
    for p, (d, r) in {**bg.OBJECT_PROPS, **bg.DATA_PROPS}.items():
        lines.append(f"  ex:{p}: {d.n3(namespace_manager=_nm())} -> {r.n3(namespace_manager=_nm())}")
    lines += [
        "ex:operates — обернена до ex:operatedBy; ex:departsFrom і ex:arrivesAt — підвластивості ex:connects.",
        "Формат IRI: поїзд ex:train/<ID у верхньому регістрі, напр. TR-415>; "
        "місто ex:city/<англійська назва з великої літери, напр. Krakow, Vienna>; "
        "перевізник ex:operator/<назва, пробіли -> _, '+' -> Plus, напр. IntercityPlus>; "
        "країна ex:country/<назва англійською>.",
        "Літерали: departureDate xsd:date (2026-07-12), departureTime xsd:time (07:05:00), "
        "durationMin xsd:integer у хвилинах, priceEur xsd:decimal (якщо ціни немає — тріпла немає).",
        "Увага: у Turtle символ '/' не можна вживати в префіксних іменах, тому для сутностей "
        "пиши повні IRI, напр. <http://example.org/mkr/train/TR-415>, <http://example.org/mkr/city/Kyiv>.",
        "Дозволені лише ці предикати, а також rdf:type і rdfs:label.",
    ]
    return "\n".join(lines)


def _nm():
    g = Graph()
    g.bind("ex", EX)
    g.bind("xsd", XSD)
    return g.namespace_manager


# -------------------------------------------------------------- виклик LLM
def thinking_config():
    """Мінімальне «обдумування», щоб модель відповідала швидко (для цієї задачі його достатньо)."""
    m = MODEL.lower()
    if m.startswith("gemini-2.5"):
        return {"thinkingBudget": 0 if "flash" in m else 128}
    if m.startswith("gemini-"):
        return {"thinkingLevel": "low"}
    return None


def call_llm(messages):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("не задано змінну середовища GEMINI_API_KEY")
    body = {
        "contents": [{"role": r, "parts": [{"text": t}]} for r, t in messages],
        "generationConfig": {"temperature": 0},
    }
    tc = thinking_config()
    if tc:
        body["generationConfig"]["thinkingConfig"] = tc
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    for attempt in range(5):
        print(f"  запит до {MODEL} (спроба {attempt + 1})…", flush=True)
        try:
            resp = requests.post(API_URL.format(model=MODEL), json=body,
                                 timeout=(15, 300), headers=headers)
        except (requests.Timeout, requests.ConnectionError) as e:
            wait = 5 * 2 ** attempt
            print(f"  мережа/таймаут ({type(e).__name__}), повтор через {wait} с")
            time.sleep(wait)
            continue
        if resp.status_code == 400 and "thinking" in resp.text.lower() \
                and "thinkingConfig" in body["generationConfig"]:
            print("  модель не підтримує thinkingConfig — надсилаю без нього")
            del body["generationConfig"]["thinkingConfig"]
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            try:
                wait = float(resp.headers.get("Retry-After", 5 * 2 ** attempt))
            except ValueError:
                wait = 5 * 2 ** attempt
            print(f"  LLM HTTP {resp.status_code}, повтор через {wait:.0f} с")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"порожня відповідь моделі: {str(data)[:300]}")
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
    raise RuntimeError("LLM недоступна після повторних спроб")


def strip_fences(s):
    m = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", s, re.S)
    return (m.group(1) if m else s).strip()


# ----------------------------------------------------- перевірка Turtle
ALLOWED_PRED = {EX[p] for p in bg.ALLOWED_PROPS} | {RDF.type, RDFS.label}
ALLOWED_CLASSES = {EX[c] for c in bg.CLASSES}
IRI_RULES = {
    "train": re.compile(r"[A-Z0-9]+(-[A-Z0-9]+)*"),
    "city": re.compile(r"[A-Z][a-z]+(_[A-Z][a-z]+)*"),
    "operator": re.compile(r"[A-Za-z0-9_.\-~%]+"),
    "country": re.compile(r"[A-Z][A-Za-z]+(_[A-Z][A-Za-z]+)*"),
    "source": re.compile(r"[A-Za-z0-9_.\-~%]+"),
}
OBJ_KIND = {"operatedBy": "operator", "operates": "train", "departsFrom": "city",
            "arrivesAt": "city", "connects": "city", "locatedIn": "country"}
DT_OF = {p: r for p, (_, r) in bg.DATA_PROPS.items()}


def iri_kind(u):
    """Повертає (тип, локальна частина) для ex:<тип>/<x> або помилку."""
    s = str(u)
    if not s.startswith(str(EX)):
        return None, f"IRI поза простором ex: {s}"
    local = s[len(str(EX)):]
    if "/" not in local:
        return None, f"IRI без типу сутності: {s}"
    kind, name = local.split("/", 1)
    rule = IRI_RULES.get(kind)
    if rule is None:
        return None, f"невідомий вид IRI '{kind}': {s}"
    if not rule.fullmatch(name):
        return None, f"IRI не відповідає формату Завдання 1: {s}"
    return kind, None


def validate(g):
    errors = []
    for s, p, o in g:
        if p not in ALLOWED_PRED:
            errors.append(f"недозволений предикат {p}")
            continue
        if isinstance(s, URIRef):
            kind, err = iri_kind(s)
            if err:
                errors.append(err)
        else:
            errors.append(f"порожній вузол як суб'єкт: {s}")
        if p == RDF.type:
            if o not in ALLOWED_CLASSES:
                errors.append(f"недозволений клас {o}")
            continue
        if p == RDFS.label:
            if not isinstance(o, Literal):
                errors.append(f"rdfs:label має бути літералом: {s}")
            continue
        local = str(p)[len(str(EX)):]
        if local in OBJ_KIND:
            if not isinstance(o, URIRef):
                errors.append(f"{local}: об'єкт має бути IRI, отримано {o!r}")
                continue
            kind, err = iri_kind(o)
            if err:
                errors.append(err)
            elif kind != OBJ_KIND[local]:
                errors.append(f"{local}: очікувався ex:{OBJ_KIND[local]}/…, отримано {o}")
        elif local in DT_OF:
            if not isinstance(o, Literal) or o.datatype != DT_OF[local]:
                errors.append(f"{local}: очікувався літерал {DT_OF[local]}, отримано {o!r}")
            elif o.value is None:
                errors.append(f"{local}: некоректне значення літерала {o!r}")
    # поїзди — суб'єкти ex:train/... — повинні мати тип
    for s in set(g.subjects()):
        if isinstance(s, URIRef) and str(s).startswith(str(EX["train/"])):
            if not set(g.objects(s, RDF.type)) & ALLOWED_CLASSES:
                errors.append(f"поїзд {s} без rdf:type")
    return sorted(set(errors))


# ------------------------------------------------- порівняння з еталоном
def expected_graph(params):
    trains = bg.merge_trains(EXPECTED_ROWS)
    return bg.build(trains, params), list(trains)


def compare(llm_g, exp_g, train_ids):
    diffs = []
    checks = [("тип", RDF.type), ("тривалість", EX.durationMin), ("ціна", EX.priceEur),
              ("дата", EX.departureDate), ("час", EX.departureTime),
              ("місто відправлення", EX.departsFrom), ("місто прибуття", EX.arrivesAt),
              ("перевізник", EX.operatedBy)]

    def short(vals):
        return ", ".join(sorted(v.n3(namespace_manager=_nm()) if isinstance(v, URIRef)
                                else str(v) for v in vals)) or "немає"

    for tid in train_ids:
        t = EX["train/" + tid]
        if (t, None, None) not in llm_g:
            diffs.append(f"{tid}: модель не створила вузол ex:train/{tid}")
            continue
        for name, p in checks:
            got, exp = set(llm_g.objects(t, p)), set(exp_g.objects(t, p))
            if p in (EX.durationMin, EX.priceEur):   # порівнюємо за значенням
                same = {x.toPython() for x in got} == {x.toPython() for x in exp}
            else:
                same = got == exp
            if not same:
                diffs.append(f"{tid} {name}: модель — {short(got)}, очікується — {short(exp)}")
    return diffs


def aggregate_notes(full, params, csv_path="trains.csv"):
    """Що мало б бути за правилами Завдання 1, якби нові поїзди були в trains.csv,
    порівняно з graph_full.ttl (агрегати під час злиття не перераховуються)."""
    if not os.path.exists(csv_path):
        return []
    ref = bg.build(bg.merge_trains(list(bg.read_rows(csv_path)) + EXPECTED_ROWS), params)
    notes = []
    for op in sorted(set(ref.subjects(EX.trainCount, None))):
        want = ref.value(op, EX.trainCount).toPython()
        have = full.value(op, EX.trainCount)
        if have is None or have.toPython() != want:
            notes.append(f"trainCount {_short(str(op))} = {want} (у graph_full.ttl — "
                         f"{have.toPython() if have is not None else 'немає'})")
    want_busy = {s for s in ref.subjects(EX.fromBusyCountry, None)}
    have_busy = {s for s in full.subjects(EX.fromBusyCountry, None)}
    for t in sorted(want_busy - have_busy):
        notes.append(f"{_short(str(t))} має отримати fromBusyCountry true")
    for t in sorted(set(ref.subjects(EX.operatedBy, None)) - set(full.subjects(EX.operatedBy, None))):
        notes.append(f"{_short(str(t))} без ex:operatedBy")
    return notes


# ------------------------------------------------------ перевірка SPARQL
def check_sparql(q):
    up = q.upper()
    found = [w for w in FORBIDDEN if re.search(rf"\b{w}\b", up)]
    if found:
        return None, f"заборонені ключові слова: {', '.join(found)}"
    try:
        pq = prepareQuery(q)          # розбирається лише ОДИН запит (не Update)
    except Exception as e:
        return None, f"запит не розбирається як один SPARQL-запит: {e}"
    if pq.algebra.name != "SelectQuery":
        return None, f"це не SELECT, а {pq.algebra.name}"
    return pq, None


MY_QUERY = """PREFIX ex: <http://example.org/mkr/>
SELECT DISTINCT ?operator WHERE {
  ?n ex:operatedBy ?operator ; a ex:NightTrain .
  ?e ex:operatedBy ?operator ; a ex:ExpressTrain .
}
ORDER BY ?operator"""


def result_set(res):
    vals = set()
    for row in res:
        for v in row:
            if v is not None:
                vals.add(str(v))
    return vals


def update_answers(path, new_lines):
    old = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = [l.rstrip("\n") for l in f if l.strip()]
    t2 = [l for l in old if l.startswith(TASK2_PREFIXES)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(t2 + list(new_lines)) + "\n")


def brief(e, n=160):
    first = " ".join(str(e).split()) or repr(e)
    return first if len(first) <= n else first[:n] + "…"


def uses_both_classes(q):
    def has(c):
        return bool(re.search(rf"(ex:{c}\b|<http://example\.org/mkr/{c}>)", q))
    return has("NightTrain") and has("ExpressTrain")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ================================================================== main
def main(argv):
    if len(argv) != 3:
        print("Використання: python llm_task.py graph.ttl graph_full.ttl")
        return 2
    base_path, full_path = argv[1], argv[2]
    base_hash = sha(base_path)
    params = bg.load_params(PARAMS_FILE)
    schema = schema_text(params)
    log(f"=== llm_task.py, {datetime.now().isoformat(timespec='seconds')}, модель {MODEL} ===")

    # ---------------- Частина А
    log("\n--- Частина А: витягування фактів ---")
    prompt = (
        "Ти перетворюєш текст на RDF. Поверни ЛИШЕ коректний Turtle (з @prefix), без пояснень.\n"
        "Не повторюй опис схеми — лише факти про поїзди, міста й перевізників із тексту.\n"
        "Використовуй тільки цю схему:\n" + schema + "\n\nТекст:\n" + TEXT
    )
    messages = [("user", prompt)]
    raw_parts, llm_g, errors, caught = [], None, [], []
    for attempt in range(MAX_FIX_RETRIES + 1):
        answer = call_llm(messages)
        raw_parts.append(f"===== Частина А, спроба {attempt + 1} =====\n{answer}\n")
        ttl = strip_fences(answer)
        g = Graph()
        try:
            g.parse(data=ttl, format="turtle")
            errors = validate(g)
        except Exception as e:
            errors = [f"Turtle не розбирається rdflib: {brief(e)}"]
        if not errors:
            llm_g = g
            log(f"Спроба {attempt + 1}: відповідь валідна ({len(g)} тріплів).")
            break
        caught.extend(errors)
        log(f"Спроба {attempt + 1}: помилки перевірки:\n  - " + "\n  - ".join(errors))
        if attempt < MAX_FIX_RETRIES:
            messages += [("model", answer),
                         ("user", "Відповідь не пройшла перевірку. Помилки:\n- " + "\n- ".join(errors) +
                          ("\nПідказка: '/' недопустимий у префіксних іменах Turtle — використовуй "
                           "повні IRI у кутових дужках." if any("rdflib" in e for e in errors) else "") +
                          "\nВиправ і поверни лише повний коректний Turtle за тією ж схемою.")]

    base = Graph().parse(base_path, format="turtle")
    full = Graph()
    for t in base:
        full.add(t)
    full.namespace_manager = base.namespace_manager

    exp_g, train_ids = expected_graph(params)
    if llm_g is not None:
        for t in llm_g:
            full.add(t)
        added = len(full) - len(base)
        dup = len(llm_g) - added
        diffs = compare(llm_g, exp_g, train_ids)
        agg = aggregate_notes(full, params)
        log(f"Злиття: graph.ttl {len(base)} тріплів -> {full_path} {len(full)} тріплів (+{added}); "
            f"модель повернула {len(llm_g)}, з них {dup} уже були в графі.")
        log("Розбіжності з еталоном за правилами Завдання 1:")
        log("  " + ("\n  ".join(diffs) if diffs else "не знайдено"))
        log("Агрегати, не перераховані під час злиття:")
        log("  " + ("\n  ".join(agg) if agg else "немає"))
        head = (f"А: модель повернула {len(llm_g)} тріпл(ів), у graph_full.ttl додано {added}"
                + (f" (ще {dup} уже були в graph.ttl, тож не продублювались)" if dup else "")
                + "; ")
        if diffs:
            line_a = head + "розбіжності з очікуваним за правилами Завдання 1: " + "; ".join(diffs) + "."
        else:
            line_a = head + ("розбіжностей у фактах не знайдено — для TR-800 і TR-801 автоматично порівняно "
                             "rdf:type, durationMin, priceEur (у TR-801 тріпла немає), departureDate, "
                             "departureTime, IRI міст і перевізника з еталоном, побудованим функцією build() "
                             "із build_graph.py.")
        if agg:
            line_a += (" Водночас агрегати Завдання 1 під час злиття не перераховуються: за правилами "
                       "мало б бути " + "; ".join(agg) + ".")
        if caught:
            line_a += (f" До валідної відповіді модель {len(raw_parts) - 1} раз(и) повертала "
                       f"невалідний Turtle (виправлено повторним запитом).")
    else:
        added = 0
        log("Відповідь моделі невалідна після 2 повторних запитів — НІЧОГО не злито.")
        log("Причина: " + "; ".join(errors))
        line_a = ("А: додано 0 тріплів — після 2 повторних запитів відповідь моделі не пройшла "
                  "перевірку (" + "; ".join(brief(x) for x in errors[:3]) + "), тому злиття не виконувалось.")
    full.serialize(destination=full_path, format="turtle", encoding="utf-8")
    Graph().parse(full_path, format="turtle")  # перевірка розбору

    # ---------------- Частина Б
    log("\n--- Частина Б: запит природною мовою ---")
    b_prompt = (
        "Напиши ОДИН SPARQL SELECT-запит (rdflib, без SERVICE) до графа з такою схемою:\n" + schema +
        f"\n\nПитання: {QUESTION}\nПоверни лише текст запиту без пояснень."
    )
    b_answer = call_llm([("user", b_prompt)])
    raw_parts.append(f"===== Частина Б =====\n{b_answer}\n")
    llm_q = strip_fences(b_answer)
    log("Запит моделі:\n" + llm_q)
    pq, err = check_sparql(llm_q)
    my_res = list(full.query(MY_QUERY))
    my_set = result_set(my_res)
    log("\nВласний запит:\n" + MY_QUERY)
    log("Результат власного запиту: " + (", ".join(sorted(my_set)) or "порожньо"))
    if err:
        log("Перевірку НЕ пройдено: " + err + " — запит не виконувався.")
        line_b = f"Б: запит моделі некоректний — {err}; він не виконувався, результат власного запиту: " \
                 f"{', '.join(sorted(_short(x) for x in my_set)) or 'порожньо'}."
    else:
        try:
            llm_res = list(full.query(pq))
            llm_set = result_set(llm_res)
            log("Перевірку пройдено. Результат запиту моделі:")
            for row in llm_res:
                log("  " + " | ".join(str(v) for v in row))
            if not llm_res:
                log("  (порожньо)")
            same = llm_set == my_set
            log("Порівняння з власним запитом: " + ("збігається" if same else "НЕ збігається"))
            mine = ", ".join(sorted(_short(x) for x in my_set)) or "порожньо"
            theirs = ", ".join(sorted(_short(x) for x in llm_set)) or "порожньо"
            if same and not uses_both_classes(llm_q):
                line_b = (f"Б: запит моделі пройшов перевірку й дав той самий результат ({mine}), але "
                          f"за змістом некоректний: він не перевіряє обидва класи ex:NightTrain і "
                          f"ex:ExpressTrain, тож збіг випадковий для цих даних.")
            elif same:
                line_b = (f"Б: запит моделі коректний — один SELECT без заборонених операцій, "
                          f"використовує ex:NightTrain, ex:ExpressTrain і ex:operatedBy за схемою; "
                          f"результати збіглися з власним запитом ({mine}).")
            else:
                line_b = (f"Б: запит моделі пройшов формальну перевірку (один SELECT), але за змістом "
                          f"некоректний — повернув {theirs}, а власний запит — {mine}.")
        except Exception as e:
            log(f"Помилка виконання запиту моделі: {e}")
            line_b = f"Б: запит моделі некоректний — пройшов перевірку, але не виконався в rdflib ({e})."

    line_v = ("В: без перевірки модель могла б злити в граф тріпли з вигаданими предикатами чи IRI "
              "(напр. ex:city/Wien або ex:operator/Intercity+), хибний тип чи тривалість у годинах "
              "замість хвилин і вигадану ціну для TR-801, а згенерований SPARQL міг би містити "
              "SERVICE/UPDATE-операції, що змінюють граф або звертаються назовні.")
    if caught:
        line_v = ("В: " + line_v[3:].rstrip(".") + "; у цьому запуску перевірка реально відсіяла: "
                  + "; ".join(brief(x, 120) for x in sorted(set(caught))[:3]) + ".")

    with open("llm_raw.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(raw_parts))
    with open("llm_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    update_answers("ANSWERS.md", [line_a, line_b, line_v])

    assert sha(base_path) == base_hash, "graph.ttl змінено!"
    print("\nANSWERS.md оновлено:\n" + "\n".join([line_a, line_b, line_v]))
    return 0


def _short(s):
    return s.replace(str(EX), "ex:")


if __name__ == "__main__":
    sys.exit(main(sys.argv))