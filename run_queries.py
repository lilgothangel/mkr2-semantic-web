"""МКР, варіант 2, завдання 2 (N = 2 -> залишок 2: замки, Польща, P2044, поріг 300).

Виконує q1.rq, q2.rq, q3.rq на https://query.wikidata.org/sparql,
зберігає q1.csv, q2.csv, q3.csv, meta.json і рядки Q1–Q3 у ANSWERS.md.

Запуск:
    python run_queries.py
"""
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests


ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "mkr2/1.0",
    "Accept": "application/sparql-results+json",
}
MAX_RETRIES = 3
BASE_THRESHOLD = 300  # поріг варіанта 2 (P2044, м)

TASK2_PREFIXES = ("Q1:", "Q2:", "Q3:", "Примітка:")
TASK3_PREFIXES = ("А:", "Б:", "В:")

def update_answers(path, owned_prefixes, new_lines):
    old = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = [l.rstrip("\n") for l in f if l.strip()]
    t2 = [l for l in old if l.startswith(TASK2_PREFIXES)]
    t3 = [l for l in old if l.startswith(TASK3_PREFIXES)]
    if owned_prefixes == TASK2_PREFIXES:
        t2 = list(new_lines)
    else:
        t3 = list(new_lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(t2 + t3) + "\n")

THRESHOLD_RE = re.compile(r"(HAVING \(MAX\(\?v\) > )([0-9.]+)(\)\s*# THRESHOLD)")


def retry_delay(resp, attempt):
    ra = resp.headers.get("Retry-After") if resp is not None else None
    if ra:
        try:
            return max(0.0, float(ra))
        except ValueError:
            try:
                dt = parsedate_to_datetime(ra)
                return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    return 5.0 * (2 ** attempt)          # 5, 10, 20 с


def run_sparql(query: str):
    """GET-запит із до 3 повторних спроб при 429/5xx (і мережевих збоях)."""
    for attempt in range(MAX_RETRIES + 1):
        resp = None
        try:
            resp = requests.get(ENDPOINT, params={"query": query},
                                headers=HEADERS, timeout=90)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait = retry_delay(None, attempt)
            print(f"  мережева помилка ({e}); повтор через {wait:.0f} с")
            time.sleep(wait)
            continue
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt == MAX_RETRIES:
                resp.raise_for_status()
            wait = retry_delay(resp, attempt)
            print(f"  HTTP {resp.status_code}; повтор {attempt + 1}/{MAX_RETRIES} через {wait:.0f} с")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data["head"]["vars"], data["results"]["bindings"]
    raise RuntimeError("недосяжний код")


def save_csv(path, vars_, bindings):
    rows = [[b.get(v, {}).get("value", "") for v in vars_] for b in bindings]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(vars_)
        w.writerows(rows)
    return [dict(zip(vars_, r)) for r in rows]


def num(s):
    if s in ("", None):
        return "—"
    try:
        x = float(s)
        return str(int(x)) if x.is_integer() else f"{x:g}"
    except ValueError:
        return s


def main():
    results, notes = {}, []

    q1_text = open("q1.rq", encoding="utf-8").read()
    m = THRESHOLD_RE.search(q1_text)
    threshold = BASE_THRESHOLD
    if m:  # завжди стартуємо з порогу варіанта (на випадок повторного запуску)
        q1_text = THRESHOLD_RE.sub(lambda mm: mm.group(1) + str(BASE_THRESHOLD) + mm.group(3), q1_text)
        with open("q1.rq", "w", encoding="utf-8") as f:
            f.write(q1_text)

    for name in ("q1", "q2", "q3"):
        query = q1_text if name == "q1" else open(f"{name}.rq", encoding="utf-8").read()
        print(f"{name}: виконую запит…")
        vars_, b = run_sparql(query)

        if name == "q1" and len(b) < 5 and m:
            new_thr = threshold / 2.0
            new_thr_s = str(int(new_thr)) if new_thr.is_integer() else str(new_thr)
            print(f"  q1 повернув {len(b)} рядків (< 5) — зменшую поріг до {new_thr_s}")
            q1_text = THRESHOLD_RE.sub(lambda mm: mm.group(1) + new_thr_s + mm.group(3), q1_text)
            with open("q1.rq", "w", encoding="utf-8") as f:
                f.write(q1_text)
            notes.append(f"Примітка: Q1 з порогом {num(threshold)} м повернув лише {len(b)} "
                         f"об'єкт(и) (менше 5), тому поріг зменшено удвічі до {new_thr_s} м (q1.rq оновлено).")
            vars_, b = run_sparql(q1_text)

        results[name] = save_csv(f"{name}.csv", vars_, b)
        print(f"  {name}.csv: {len(b)} рядків")
        time.sleep(1)  # ввічливість до публічного ендпоінта

    meta = {
        "retrieved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "rows": {k: len(v) for k, v in results.items()},
    }
    with open("meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if len(results["q2"]) < 10:
        notes.append(f"Примітка: Q2 повернув {len(results['q2'])} регіон(и/ів) замість 10 "
                     f"(регіонів Польщі з ≥ 3 замками менше за 10).")

    lines = []
    r = results["q1"]
    if r:
        site = f"є ({r[0]['website']})" if r[0]["website"] else "немає"
        lines.append(f"Q1: {r[0]['itemLabel']} — найбільша висота {num(r[0]['value'])} м; "
                     f"офіційний сайт: {site}.")
    else:
        lines.append("Q1: запит не повернув жодного об'єкта.")
    r = results["q2"]
    if r:
        mv = (f"maxValue = {num(r[0]['maxValue'])} м" if r[0]["maxValue"]
              else "maxValue порожнє — у жодного замку цього регіону немає висоти (P2044)")
        lines.append(f"Q2: {r[0]['regionLabel']} — кількість замків {r[0]['cnt']}; {mv}.")
    else:
        lines.append("Q2: запит не повернув жодного регіону.")
    r = results["q3"]
    if r:
        lines.append(f"Q3: населення {num(r[0]['population'])} — регіон {r[0]['regionLabel']} "
                     f"(об'єкт: {r[0]['itemLabel']}).")
    else:
        lines.append("Q3: запит не повернув жодного об'єкта.")
    update_answers("ANSWERS.md", TASK2_PREFIXES, lines + notes)

    print(json.dumps(meta, ensure_ascii=False))
    print("ANSWERS.md оновлено:")
    print("\n".join(lines + notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
