"""МКР, варіант 2, завдання 1.

Будує RDF-граф поїздів із trains.csv і params.json та зберігає його у Turtle.

Запуск:
    python build_graph.py trains.csv params.json graph.ttl
"""
import csv
import json
import re
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

EX = Namespace("http://example.org/mkr/")
WD = Namespace("http://www.wikidata.org/entity/")

FIELDS = ["id", "operator", "from", "to", "date", "time",
          "duration_min", "price_eur", "note"]

# ---------------------------------------------------------------- схема
CLASSES = ["Train", "ExpressTrain", "NightTrain", "Operator",
           "City", "Country", "Source"]

# властивість -> (тип, domain, range)
OBJECT_PROPS = {
    "operatedBy":  (EX.Train, EX.Operator),
    "operates":    (EX.Operator, EX.Train),
    "connects":    (EX.Train, EX.City),
    "departsFrom": (EX.Train, EX.City),
    "arrivesAt":   (EX.Train, EX.City),
    "locatedIn":   (EX.City, EX.Country),
    "reportedBy":  (RDF.Statement, EX.Source),
}
DATA_PROPS = {
    "departureDate":   (EX.Train, XSD.date),
    "departureTime":   (EX.Train, XSD.time),
    "durationMin":     (EX.Train, XSD.integer),
    "priceEur":        (EX.Train, XSD.decimal),
    "hasDelayMinutes": (EX.Train, XSD.integer),
    "confidence":      (RDF.Statement, XSD.decimal),
    "fromBusyCountry": (EX.Train, XSD.boolean),
    "trainCount":      (EX.Operator, XSD.integer),
}
ALLOWED_PROPS = set(OBJECT_PROPS) | set(DATA_PROPS)


def add_schema(g: Graph) -> None:
    for c in CLASSES:
        g.add((EX[c], RDF.type, OWL.Class))
    g.add((EX.ExpressTrain, RDFS.subClassOf, EX.Train))
    g.add((EX.NightTrain, RDFS.subClassOf, EX.Train))

    for p, (dom, rng) in OBJECT_PROPS.items():
        g.add((EX[p], RDF.type, OWL.ObjectProperty))
        g.add((EX[p], RDFS.domain, dom))
        g.add((EX[p], RDFS.range, rng))
    for p, (dom, rng) in DATA_PROPS.items():
        g.add((EX[p], RDF.type, OWL.DatatypeProperty))
        g.add((EX[p], RDFS.domain, dom))
        g.add((EX[p], RDFS.range, rng))

    g.add((EX.operates, OWL.inverseOf, EX.operatedBy))
    g.add((EX.departsFrom, RDFS.subPropertyOf, EX.connects))
    g.add((EX.arrivesAt, RDFS.subPropertyOf, EX.connects))


# ---------------------------------------------------------- нормалізація
def norm_id(s: str) -> str:
    return s.strip().upper()


def norm_city(s: str) -> str:
    return s.strip().title()


def iri_part(s: str) -> str:
    """Пробіли -> _, '+' -> Plus, решта небезпечних символів кодується."""
    s = s.strip().replace(" ", "_").replace("+", "Plus")
    return quote(s, safe="_-.~")


def parse_date(s: str):
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"невідомий формат дати: {s!r}")


def parse_time(s: str):
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s.strip())
    if not m:
        raise ValueError(f"невідомий формат часу: {s!r}")
    h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if not (0 <= h < 24 and 0 <= mi < 60 and 0 <= sec < 60):
        raise ValueError(f"некоректний час: {s!r}")
    return f"{h:02d}:{mi:02d}:{sec:02d}"


def time_minutes(hhmmss: str) -> int:
    h, m = hhmmss.split(":")[:2]
    return int(h) * 60 + int(m)


def parse_note(note: str):
    """'delay=10;by=X;conf=0.7|delay=40;by=Y' -> список словників."""
    out = []
    for rec in (note or "").split("|"):
        rec = rec.strip()
        if not rec:
            continue
        kv = {}
        for part in rec.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k.strip().lower()] = v.strip()
        if "delay" not in kv:
            print(f"  [!] запис note без delay пропущено: {rec!r}", file=sys.stderr)
            continue
        try:
            delay = int(kv["delay"])
        except ValueError:
            print(f"  [!] некоректний delay: {rec!r}", file=sys.stderr)
            continue
        conf = None
        if kv.get("conf"):
            try:
                conf = Decimal(kv["conf"])
            except InvalidOperation:
                print(f"  [!] некоректний conf: {rec!r}", file=sys.stderr)
        out.append({"delay": delay, "by": kv.get("by", "").strip(), "conf": conf})
    return out


# --------------------------------------------------------- злиття рядків
def read_rows(path: str):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): (v or "") for k, v in row.items() if k is not None}
            if not any(v.strip() for v in row.values()):
                continue
            yield row


def merge_trains(rows):
    """Один поїзд = один запис. Поле береться з першого рядка, де воно
    непорожнє; записи note об'єднуються без дублів (delay, by)."""
    trains = {}
    for row in rows:
        tid = norm_id(row.get("id", ""))
        if not tid:
            continue
        t = trains.setdefault(tid, {"id": tid, "delays": [], "_seen": set()})
        for f in FIELDS:
            if f in ("id", "note"):
                continue
            v = row.get(f, "").strip()
            if v and f not in t:
                t[f] = v
        for d in parse_note(row.get("note", "")):
            key = (d["delay"], d["by"])
            if key not in t["_seen"]:
                t["_seen"].add(key)
                t["delays"].append(d)
    for t in trains.values():
        t.pop("_seen")
    return trains


# --------------------------------------------------------- побудова графа
def train_types(t, params):
    types = []
    dur, price, tm = t.get("_duration"), t.get("_price"), t.get("_time")
    if price is not None and dur is not None and dur <= int(params["express_max_min"]):
        types.append(EX.ExpressTrain)
    if tm is not None:
        m = time_minutes(tm)
        nf = time_minutes(parse_time(params["night_from"]))
        nu = time_minutes(parse_time(params["night_until"]))
        if m >= nf or m < nu:
            types.append(EX.NightTrain)
    return types or [EX.Train]


def build(trains: dict, params: dict) -> Graph:
    g = Graph()
    g.bind("ex", EX)
    g.bind("wd", WD)
    g.bind("owl", OWL)
    add_schema(g)

    countries = {norm_city(k): v for k, v in params.get("country", {}).items()}
    wikidata = {norm_city(k): v for k, v in params.get("wikidata", {}).items()}

    cities_used, operators = set(), {}

    def city_node(name):
        c = norm_city(name)
        cities_used.add(c)
        return EX["city/" + iri_part(c)]

    for tid, t in trains.items():
        tr = EX["train/" + iri_part(tid)]

        # типізовані значення
        try:
            t["_date"] = parse_date(t["date"]) if t.get("date") else None
        except ValueError as e:
            print(f"  [!] {tid}: {e}", file=sys.stderr); t["_date"] = None
        try:
            t["_time"] = parse_time(t["time"]) if t.get("time") else None
        except ValueError as e:
            print(f"  [!] {tid}: {e}", file=sys.stderr); t["_time"] = None
        try:
            t["_duration"] = int(t["duration_min"]) if t.get("duration_min") else None
        except ValueError:
            print(f"  [!] {tid}: некоректна тривалість", file=sys.stderr); t["_duration"] = None
        try:
            t["_price"] = Decimal(t["price_eur"]) if t.get("price_eur") else None
        except InvalidOperation:
            print(f"  [!] {tid}: некоректна ціна", file=sys.stderr); t["_price"] = None

        for cls in train_types(t, params):
            g.add((tr, RDF.type, cls))

        if t.get("operator"):
            op_name = t["operator"].strip()
            op = EX["operator/" + iri_part(op_name)]
            operators.setdefault(op, (op_name, set()))[1].add(tid)
            g.add((tr, EX.operatedBy, op))
            g.add((op, EX.operates, tr))

        if t.get("from"):
            g.add((tr, EX.departsFrom, city_node(t["from"])))
        if t.get("to"):
            g.add((tr, EX.arrivesAt, city_node(t["to"])))

        if t["_date"] is not None:
            g.add((tr, EX.departureDate, Literal(t["_date"].isoformat(), datatype=XSD.date)))
        if t["_time"] is not None:
            g.add((tr, EX.departureTime, Literal(t["_time"], datatype=XSD.time)))
        if t["_duration"] is not None:
            g.add((tr, EX.durationMin, Literal(t["_duration"], datatype=XSD.integer)))
        if t["_price"] is not None:
            g.add((tr, EX.priceEur, Literal(t["_price"], datatype=XSD.decimal)))

        # затримки — лише через реїфікацію
        for k, d in enumerate(t["delays"], start=1):
            st = EX[f"stmt/{iri_part(tid)}-delay-{k}"]
            g.add((st, RDF.type, RDF.Statement))
            g.add((st, RDF.subject, tr))
            g.add((st, RDF.predicate, EX.hasDelayMinutes))
            g.add((st, RDF.object, Literal(d["delay"], datatype=XSD.integer)))
            if d["by"]:
                src = EX["source/" + iri_part(d["by"])]
                g.add((src, RDF.type, EX.Source))
                g.add((src, RDFS.label, Literal(d["by"], lang="en")))
                g.add((st, EX.reportedBy, src))
            if d["conf"] is not None:
                g.add((st, EX.confidence, Literal(d["conf"], datatype=XSD.decimal)))

    # міста, країни, Wikidata
    for c in sorted(cities_used):
        cn = EX["city/" + iri_part(c)]
        g.add((cn, RDF.type, EX.City))
        g.add((cn, RDFS.label, Literal(c, lang="en")))
        country = countries.get(c)
        if country:
            co = EX["country/" + iri_part(country)]
            g.add((cn, EX.locatedIn, co))
            g.add((co, RDF.type, EX.Country))
            g.add((co, RDFS.label, Literal(country, lang="en")))
        qid = wikidata.get(c)
        if qid:
            g.add((cn, OWL.sameAs, WD[qid.strip()]))

    # перевізники + агрегат (б)
    for op, (name, tids) in operators.items():
        g.add((op, RDF.type, EX.Operator))
        g.add((op, RDFS.label, Literal(name, lang="en")))
        g.add((op, EX.trainCount, Literal(len(tids), datatype=XSD.integer)))

    # агрегат (а): країни, з яких відправляється >= 2 різних поїзди
    dep_country = {}
    for tid, t in trains.items():
        if t.get("from"):
            dep_country[tid] = countries.get(norm_city(t["from"]))
    per_country = {}
    for tid, co in dep_country.items():
        if co:
            per_country.setdefault(co, set()).add(tid)
    for tid, co in dep_country.items():
        if co and len(per_country[co]) >= 2:
            g.add((EX["train/" + iri_part(tid)], EX.fromBusyCountry,
                   Literal(True, datatype=XSD.boolean)))
    return g


def load_params(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv):
    if len(argv) != 4:
        print("Використання: python build_graph.py trains.csv params.json graph.ttl")
        return 2
    csv_path, params_path, out_path = argv[1:]
    params = load_params(params_path)
    trains = merge_trains(read_rows(csv_path))
    g = build(trains, params)
    g.serialize(destination=out_path, format="turtle", encoding="utf-8")

    # перевірка: файл має розбиратися rdflib
    check = Graph()
    check.parse(out_path, format="turtle")
    assert len(check) == len(g), "кількість тріплів після розбору не збігається"
    n_tr = len(set(check.subjects(EX.operatedBy, None)))
    print(f"OK: {out_path} — {len(check)} тріплів, поїздів: {n_tr}, "
          f"розбір rdflib пройдено")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
