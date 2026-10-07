"""Offline tests for the Task 1 checks: no model, no MCP server, no API key.

    python tests/test_offline.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hw2  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  ' + str(detail)) if detail else ''}")


def cv(**kw):
    base = {"name": "Alex Chan", "headline": "Finance Professional", "city": "Hong Kong",
            "country": "Hong Kong", "jobs": [], "education": [], "skills": []}
    base.update(kw)
    return hw2.clean_cv(base)


def job(company, title, start, end=None):
    return {"company": company, "title": title, "start_year": start, "end_year": end, "current": end is None}


def exp(company, title, seniority, start, end=None):
    return {"company": company, "title": title, "seniority": seniority, "start_year": start,
            "end_year": end, "is_current": end is None}


def profile(**kw):
    base = {"id": 1, "name": "Alex Chan", "city": "Hong Kong", "country": "Hong Kong",
            "experience": [], "education": [], "skills": []}
    base.update(kw)
    return base


print("wording: same thing")
for a, b in [("Stanford", "Stanford University"), ("MIT", "Massachusetts Institute of Technology"),
             ("HKUST", "The Hong Kong University of Science and Technology"),
             ("CUHK", "The Chinese University of Hong Kong"), ("BCG", "Boston Consulting Group"),
             ("Standard Chartered", "Standard Chartered Bank"), ("Innovate HK Ltd", "Innovate HK"),
             ("HSBC", "HSBC Holdings plc")]:
    check(f"same_org({a!r}, {b!r})", hw2.same_org(a, b))
for a, b in [("UI/UX Design", "UI/UX"), ("ML", "Machine Learning"), ("Python", "python")]:
    check(f"same_term({a!r}, {b!r})", hw2.same_term(a, b))

print("wording: different things")
for a, b in [("City University of Hong Kong", "The University of Hong Kong"), ("HSBC", "Standard Chartered"),
             ("Oxford", "Cambridge"), ("Hong Kong Baptist University", "Hong Kong Polytechnic University")]:
    check(f"not same_org({a!r}, {b!r})", not hw2.same_org(a, b))
for a, b in [("Docker", "Kubernetes"), ("Java", "JavaScript"), ("Finance", "Accounting")]:
    check(f"not same_term({a!r}, {b!r})", not hw2.same_term(a, b))

print("titles and degrees")
check("split_title('Senior Manager')", hw2.split_title("Senior Manager") == ("senior", "manager"))
check("split_title('Analyst')", hw2.split_title("Analyst") == (None, "analyst"))
for raw, want in [("Master of Business Administration", "MBA"), ("B.Sc.", "BSc"), ("BSc", "BSc"),
                  ("Bachelor of Science", "BSc"), ("PhD", "PhD"), ("Doctor of Philosophy", "PhD"),
                  ("MSc", "MSc")]:
    check(f"norm_degree({raw!r}) == {want}", hw2.norm_degree(raw) == want, hw2.norm_degree(raw))

print("compare: valid CVs (wording only)")
p1 = profile(experience=[exp("Deloitte", "Manager", "senior", 2020), exp("Carousell", "Scientist", "senior", 2012, 2018)],
             education=[{"school": "Stanford University", "degree": "MBA", "field": "Logistics", "end_year": 2012}],
             skills=[{"name": "Supply Chain"}, {"name": "Inventory Management"}, {"name": "Operations"}, {"name": "SQL"}])
c1 = cv(jobs=[job("Deloitte", "Senior Manager", 2020), job("Carousell", "Senior Scientist", 2012, 2018)],
        education=[{"degree": "Master of Business Administration", "field": "Logistics", "school": "Stanford",
                    "graduation_year": 2012}],
        skills=["Supply Chain", "Inventory Management", "Operations"])
found, pending = hw2.compare(c1, p1)
check("senior title + MBA wording + fewer skills -> no discrepancy", found == [] and pending == {}, found)

c6 = cv(jobs=[job("Tencent", "Junior Scientist", 2022)], skills=["UI/UX Design"])
p6 = profile(experience=[exp("Tencent", "Scientist", "junior", 2022)], skills=[{"name": "UI/UX"}])
check("junior title + 'UI/UX Design' -> no discrepancy", hw2.compare(c6, p6)[0] == [])

print("compare: the three public discrepancy types")
c4 = cv(jobs=[job("Innovate HK Ltd", "Engineer", 2015)])
p4 = profile(experience=[exp("Innovate HK Ltd", "Engineer", "mid", 2014)])
check("start year 2015 vs 2014", any("start year" in d for d in hw2.compare(c4, p4)[0]), hw2.compare(c4, p4)[0])
c5 = cv(jobs=[job("Microsoft", "Senior Analyst", 2007, 2008)])
p5 = profile(experience=[exp("Microsoft", "Analyst", "junior", 2007, 2008)])
check("Senior Analyst vs Analyst/junior", any("seniority" in d for d in hw2.compare(c5, p5)[0]), hw2.compare(c5, p5)[0])
c7 = cv(skills=["Docker", "Python"])
p7 = profile(skills=[{"name": "Python"}, {"name": "Go"}, {"name": "Kubernetes"}, {"name": "Java"}])
f7, pend7 = hw2.compare(c7, p7)
check("Docker not on LinkedIn (goes to the judge first)", "skill:Docker" in pend7)
f7, _ = hw2.compare(c7, p7, {"skill:Docker": None})
check("...judge says no match -> discrepancy", any("Docker" in d for d in f7), f7)

print("compare: other discrepancy types")
base_p = profile(experience=[exp("AIA", "Analyst", "mid", 2019)],
                 education=[{"school": "HKUST", "degree": "BSc", "field": "Finance", "end_year": 2016}])
cases = {
    "inflated title": cv(jobs=[job("AIA", "Manager", 2019)]),
    "end year": cv(jobs=[job("AIA", "Analyst", 2019, 2023)]),
    "upgraded degree": cv(education=[{"degree": "MSc", "field": "Finance", "school": "HKUST", "graduation_year": 2016}]),
    "graduation year": cv(education=[{"degree": "BSc", "field": "Finance", "school": "HKUST", "graduation_year": 2017}]),
    "wrong city": cv(city="Singapore"),
    "wrong name": cv(name="Alex Chen"),
}
for label, c in cases.items():
    found, pend = hw2.compare(c, base_p)
    check(label, found, found)
fake_school = cv(education=[{"degree": "BSc", "field": "Finance", "school": "Harvard", "graduation_year": 2016}])
found, pend = hw2.compare(fake_school, base_p, {"school:Harvard": None})
check("fake school (judge says no) ", any("school" in d for d in found), found)
abbrev = cv(education=[{"degree": "BSc", "field": "Finance", "school": "HKU", "graduation_year": 2016}])
hku_p = profile(education=[{"school": "The University of Hong Kong", "degree": "BSc", "field": "Finance", "end_year": 2016}])
found, pend = hw2.compare(abbrev, hku_p)
check("HKU vs full name goes to the judge", "school:HKU" in pend)
found, _ = hw2.compare(abbrev, hku_p, {"school:HKU": "The University of Hong Kong"})
check("...judge says same -> valid", found == [], found)


class FakeTool:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    async def ainvoke(self, args):
        return [{"type": "text", "text": json.dumps(self.fn(**args))}]


people = {
    11: profile(id=11, name="Alex Chan", city="Hong Kong", experience=[exp("AIA", "Analyst", "mid", 2019)]),
    12: profile(id=12, name="Alex Chan", city="Hong Kong", experience=[exp("HSBC", "Analyst", "mid", 2016, 2019),
                                                                       exp("AIA", "Manager", "mid", 2019)],
                education=[{"school": "HKUST", "degree": "BSc", "field": "Finance", "end_year": 2016}]),
    13: profile(id=13, name="Alex Chan", city="Shenzhen", experience=[exp("Tencent", "Engineer", "mid", 2018)]),
}


def search(q, limit=20, location=None, industry=None, fuzzy=True):
    hits = [{"id": i, "name": p["name"], "location": f"{p['city']}, China", "industry": "Finance", "headline": ""}
            for i, p in people.items() if q.lower() in p["name"].lower()]
    return [h for h in hits if not location or location.lower() in h["location"].lower()][:limit]


tools = [FakeTool("search_linkedin_people", search),
         FakeTool("get_linkedin_profile", lambda person_id: people.get(person_id, {"error": "unknown id"}))]

print("finding the right person among namesakes")
target = cv(jobs=[job("AIA", "Manager", 2019), job("HSBC", "Analyst", 2016, 2019)],
            education=[{"degree": "BSc", "field": "Finance", "school": "HKUST", "graduation_year": 2016}])
verifier = hw2.CVVerifier(model=None, tools=tools, resolver=None)
trace = {"searches": [], "candidates": []}
got = asyncio.run(verifier.find_profile(target, trace))
check("picks the namesake whose employers and school match", got and got["id"] == 12, got and got["id"])
moved = dict(target, city="Singapore")
trace = {"searches": [], "candidates": []}
got = asyncio.run(verifier.find_profile(moved, trace))
check("still finds them when the CV city is false (looser search)", got and got["id"] == 12, trace["searches"])
for i in range(100, 115):                         # 15 namesakes in the same city, sorted before the real one
    people[i] = profile(id=i, name="Alex Chan", city="Hong Kong", experience=[exp("Shopee", "Engineer", "mid", 2015)])
people[999] = dict(people.pop(12), id=999)
trace = {"searches": [], "candidates": []}
got = asyncio.run(verifier.find_profile(target, trace))
check("real person found behind 15+ namesakes with the same city", got and got["id"] == 999,
      f"opened {len(trace['candidates'])}")
nobody = cv(name="Zed Nobody", jobs=[job("AIA", "Manager", 2019)])
trace = {"searches": [], "candidates": []}
check("unknown name -> no profile", asyncio.run(verifier.find_profile(nobody, trace)) is None)

print("tool output and scoring plumbing")
check("tool_json decodes content blocks", hw2.tool_json([{"type": "text", "text": '{"a": 1}'}]) == {"a": 1})
check("tool_json decodes a plain string", hw2.tool_json('[{"id": 3}]') == [{"id": 3}])
check("parse_json_reply tolerates fences", hw2.parse_json_reply('```json\n{"x": 2}\n```') == {"x": 2})


class FakeAgent:
    def __init__(self):
        self.live = self.peak = 0

    async def verify(self, text):
        self.live += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(0.05)
        self.live -= 1
        if "boom" in text:
            raise RuntimeError("API down")
        return (0.9 if "good" in text else 0.1), {}


fake = FakeAgent()
scores = asyncio.run(hw2.score_cvs(fake, {f"CV_{i}.pdf": ("good" if i % 2 else "bad") for i in range(1, 9)}
                                   | {"CV_9.pdf": "boom"}))
check("one score per CV, all in [0, 1]", len(scores) == 9 and all(0 <= s <= 1 for s in scores.values()))
check("a failing CV still gets a score", scores["CV_9.pdf"] == hw2.SCORE_ERROR)
check("at most 3 CVs in flight", fake.peak <= 3, fake.peak)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
