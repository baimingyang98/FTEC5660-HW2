"""Build a labelled stress-test folder of CV PDFs from real SocialGraph profiles.

    pip install fpdf2
    python tests/make_stress_set.py --out /content/stress
    python hw2.py --cv-folder /content/stress

Valid CVs differ from the LinkedIn profile only in wording (what the homework says must not
count). Every other CV carries exactly one planted discrepancy of a type the homework lists.
Fake employers, schools, skills, cities and fields are taken from *other* profiles on the same
server. ground_truth.json uses the public format, so hw2.py reports accuracy and the reason of
every miss; manifest.json adds the discrepancy type and the LinkedIn id for analysis.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hw2  # noqa: E402

TYPES = ["start_year", "end_year", "title", "seniority", "degree", "grad_year",
         "school", "employer", "city", "skill", "field", "name"]
LONG_DEGREE = {"BSc": "Bachelor of Science", "MSc": "Master of Science",
               "MBA": "Master of Business Administration", "PhD": "PhD"}
UPGRADE = {"BSc": "MSc", "MSc": "PhD", "MBA": "PhD"}
HOMETOWNS = ["Sha Tin", "Delhi", "Boston", "London", "Hong Kong", "Stanford", "Osaka", None, None]
BULLETS = ["Led a team delivering {i} projects on schedule.", "Managed stakeholders, budgets, and priorities.",
           "Analysed {i} data to support business decisions.", "Prepared reports and dashboards for stakeholders.",
           "Built and maintained {i} systems in production.", "Collaborated with cross-functional teams on delivery."]


def usable(p):
    return (isinstance(p, dict) and "error" not in p and p.get("id") != 10001 and p.get("name")
            and p.get("city") and len(p.get("experience") or []) >= 1 and len(p.get("education") or []) >= 1
            and len(p.get("skills") or []) >= 2)


async def fetch_profiles(count, seed):
    tools = await hw2.load_mcp_tools()
    get = next(t for t in tools if t.name.endswith("get_linkedin_profile"))
    gate = asyncio.Semaphore(3)

    async def one(pid):
        async with gate:
            try:
                return hw2.tool_json(await get.ainvoke({"person_id": pid}))
            except Exception:
                return None

    ids = random.Random(seed).sample(range(1, 10001), 600)
    found = []
    for start in range(0, len(ids), 30):
        found += [p for p in await asyncio.gather(*(one(i) for i in ids[start:start + 30])) if usable(p)]
        if len(found) >= count:
            break
    return found


def skill_names(p):
    return [s["name"] if isinstance(s, dict) else str(s) for s in p.get("skills") or []]


def title_text(exp, rng):
    seniority = str(exp.get("seniority") or "").lower()
    if seniority in ("senior", "junior") and rng.random() < 0.7:
        return f"{seniority.title()} {exp['title']}"
    return exp["title"]


def base_cv(p, rng):
    exps = sorted(p["experience"], key=lambda e: -(e.get("start_year") or 0))
    skills = skill_names(p)
    return {
        "name": p["name"], "headline": p.get("headline") or f"{p.get('industry', 'Business')} Professional",
        "industry": p.get("industry") or "business", "city": p["city"], "country": p.get("country") or "",
        "hometown": rng.choice(HOMETOWNS),
        "jobs": [{"company": e["company"], "title": title_text(e, rng), "start": e.get("start_year"),
                  "end": None if e.get("is_current") or e.get("end_year") is None else e["end_year"],
                  "seniority": str(e.get("seniority") or "").lower(), "base": e["title"]} for e in exps],
        "education": [{"degree": x["degree"], "field": x.get("field") or "", "school": x["school"],
                       "year": x.get("end_year")} for x in p["education"]],
        "skills": rng.sample(skills, rng.randint(max(2, len(skills) - 2), len(skills))),
    }


def shift(year, rng):
    return year + rng.choice([-2, -1, 1, 2])


def mutate(kind, cv, p, pools, rng):
    """Plant one discrepancy of `kind` in cv. Return the reason, or None if it does not apply."""
    jobs, edus = cv["jobs"], cv["education"]
    if kind == "start_year":
        j = rng.choice([j for j in jobs if j["start"]])
        old, j["start"] = j["start"], shift(j["start"], rng)
        return f"start year at {j['company']}: CV {j['start']}, LinkedIn {old}"
    if kind == "end_year":
        past = [j for j in jobs if j["end"]]
        if not past:
            return None
        j = rng.choice(past)
        old, j["end"] = j["end"], shift(j["end"], rng)
        return f"end year at {j['company']}: CV {j['end']}, LinkedIn {old}"
    if kind == "title":
        j = rng.choice(jobs)
        options = [t for t in pools["titles"] if t.lower() not in j["base"].lower() and j["base"].lower() not in t.lower()]
        new = rng.choice(options)
        old, j["title"] = j["title"], new
        return f"title at {j['company']}: CV {new}, LinkedIn {j['base']} ({j['seniority']})"
    if kind == "seniority":
        cand = [j for j in jobs if j["seniority"] != "senior"]
        if not cand:
            return None
        j = rng.choice(cand)
        j["title"] = f"Senior {j['base']}"
        return f"seniority at {j['company']}: CV Senior {j['base']}, LinkedIn {j['seniority']}"
    if kind == "degree":
        cand = [x for x in edus if x["degree"] in UPGRADE]
        if not cand:
            return None
        x = rng.choice(cand)
        old, x["degree"] = x["degree"], UPGRADE[x["degree"]]
        return f"degree at {x['school']}: CV {x['degree']}, LinkedIn {old}"
    if kind == "grad_year":
        x = rng.choice([x for x in edus if x["year"]])
        old, x["year"] = x["year"], shift(x["year"], rng)
        return f"graduation year at {x['school']}: CV {x['year']}, LinkedIn {old}"
    if kind == "school":
        x = rng.choice(edus)
        mine = [e["school"] for e in p["education"]]
        new = rng.choice([s for s in pools["schools"] if not any(hw2.same_org(s, m) for m in mine)])
        old, x["school"] = x["school"], new
        return f"school: CV {new}, LinkedIn {old}"
    if kind == "employer":
        j = rng.choice(jobs)
        mine = [e["company"] for e in p["experience"]]
        new = rng.choice([c for c in pools["companies"] if not any(hw2.same_org(c, m) for m in mine)])
        old, j["company"] = j["company"], new
        return f"employer: CV {new}, LinkedIn {old}"
    if kind == "city":
        new = rng.choice([c for c in pools["cities"] if c.lower() != cv["city"].lower()])
        old, cv["city"] = cv["city"], new
        return f"city: CV {new}, LinkedIn {old}"
    if kind == "skill":
        mine = skill_names(p)
        new = rng.choice([s for s in pools["skills"] if not any(hw2.same_term(s, m) for m in mine)])
        cv["skills"].insert(rng.randint(0, len(cv["skills"])), new)
        return f"skill: CV lists {new}, not on LinkedIn"
    if kind == "field":
        x = rng.choice([x for x in edus if x["field"]])
        new = rng.choice([f for f in pools["fields"] if not hw2.same_term(f, x["field"])])
        old, x["field"] = x["field"], new
        return f"field at {x['school']}: CV {new}, LinkedIn {old}"
    if kind == "name":
        first, _, last = cv["name"].partition(" ")
        vowels = [i for i, ch in enumerate(last) if ch in "aeiou"]
        if not last or not vowels:
            return None
        i = rng.choice(vowels)
        new_last = last[:i] + rng.choice([v for v in "aeiou" if v != last[i]]) + last[i + 1:]
        old, cv["name"] = cv["name"], f"{first} {new_last}"
        return f"name: CV {cv['name']}, LinkedIn {old}"
    raise ValueError(kind)


def render(cv, path, rng, fonts):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.add_font("dv", "", fonts[0])
    pdf.add_font("dv", "B", fonts[1])

    def line(text, size=10.5, bold=False, gap=6):
        pdf.set_font("dv", "B" if bold else "", size)
        pdf.cell(0, gap, text, new_x="LMARGIN", new_y="NEXT")

    line(cv["name"], 18, True, 10)
    line(cv["headline"], 12)
    place = f"{cv['city']}, {cv['country']}" if cv["country"] else cv["city"]
    if cv["hometown"]:
        place += rng.choice([f" | Hometown: {cv['hometown']}", f" | {cv['hometown']}"])
    line(place, 10.5, gap=8)

    def experience():
        line(rng.choice(["Professional Experience", "Experience"]), 13, True, 9)
        for j in cv["jobs"]:
            span = f"{j['start']} – {j['end'] if j['end'] else 'Present'}"
            if rng.random() < 0.7:
                line(f"{j['title']}, {j['company']}    {span}", bold=True)
            else:
                line(f"{j['title']}, {j['company']}", bold=True)
                line(span)
            for b in rng.sample(BULLETS, 2):
                line("- " + b.format(i=cv["industry"].lower()))

    def education():
        line("Education", 13, True, 9)
        for x in cv["education"]:
            degree = x["degree"] if rng.random() < 0.5 else LONG_DEGREE.get(x["degree"], x["degree"])
            text = f"{degree} in {x['field']}" if x["field"] else degree
            if rng.random() < 0.7:
                line(f"{text}    Graduated {x['year']}", bold=True)
            else:
                line(text, bold=True)
                line(str(x["year"]))
            line(x["school"])

    def skills():
        line("Skills", 13, True, 9)
        if rng.random() < 0.6:
            line(", ".join(cv["skills"]))
        else:
            for s in cv["skills"]:
                line("• " + s)

    sections = [experience, education, skills]
    rng.shuffle(sections)
    for section in sections:
        section()
    pdf.output(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--valid", type=int, default=10)
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import matplotlib
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fonts = (str(ttf / "DejaVuSans.ttf"), str(ttf / "DejaVuSans-Bold.ttf"))

    hw2.load_env_file()
    need = args.valid + args.per_type * len(TYPES)
    profiles = asyncio.run(fetch_profiles(need + 20, args.seed))
    print(f"fetched {len(profiles)} usable profiles")
    pools = {
        "titles": sorted({e["title"] for p in profiles for e in p["experience"]}),
        "companies": sorted({e["company"] for p in profiles for e in p["experience"]}),
        "schools": sorted({x["school"] for p in profiles for x in p["education"]}),
        "cities": sorted({p["city"] for p in profiles}),
        "skills": sorted({s for p in profiles for s in skill_names(p)}),
        "fields": sorted({x["field"] for p in profiles for x in p["education"] if x.get("field")}),
    }
    rng = random.Random(args.seed)
    plan = ["valid"] * args.valid + [t for t in TYPES for _ in range(args.per_type)]
    rng.shuffle(plan)

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.pdf"):
        old.unlink()
    truth, manifest, queue = {}, {}, list(profiles)
    for n, kind in enumerate(plan, 1):
        while queue:
            p = queue.pop(0)
            cv = base_cv(p, rng)
            reason = None if kind == "valid" else mutate(kind, cv, p, pools, rng)
            if kind == "valid" or reason:
                break
        else:
            raise SystemExit("ran out of profiles; raise the sample size")
        name = f"CV_{n}.pdf"
        render(cv, args.out / name, rng, fonts)
        truth[name] = {"label": 1} if kind == "valid" else {"label": 0, "reason": reason}
        manifest[name] = {"type": kind, "linkedin_id": p["id"], "name": cv["name"]}
    (args.out / "ground_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    counts = {k: sum(m["type"] == k for m in manifest.values()) for k in ["valid"] + TYPES}
    print(f"wrote {len(plan)} CVs to {args.out}: {counts}")


if __name__ == "__main__":
    main()
