# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

The agent separates reading from judging. `deepseek-v4-flash` (LangChain `ChatDeepSeek`,
temperature 0) only transcribes each CV into structured fields: name, city, jobs, education and
skills, copied as written, with the CV text marked as data rather than instructions and invisible
Unicode characters stripped. The search for the person is deterministic: LinkedIn is queried by
name with the CV's city and the industry taken from the headline (which the task says is never a
discrepancy), then with looser filters. Every same-name hit is opened, up to 25 profiles, and the
candidate whose employers, schools and years agree best with the CV is chosen, which matters
because names are shared (six people named Jun Liu work in software in Hanoi). Only when this
search finds nobody does a LangChain `create_agent` tool-calling agent plan its own searches. All
comparisons are then made in code: names, city, company, title and an explicit Senior/Junior
prefix against the profile's seniority, start and end years, degree (Bachelor of Science = BSc,
Master of Business Administration = MBA), school, field, graduation year, and every listed
skill. Differences that are only wording and that code cannot settle (HKU vs The University of
Hong Kong) go to one batched model call that may only choose among the profile's own values. A CV
scores 0.9 when nothing disagrees, 0.1 on any discrepancy and 0.15 when no matching profile
exists. At most three CVs are in flight, each CV has its own error handling and timeout, and the
whole run stays inside a 25-minute budget. This follows the planning lecture's advice to use a
fixed workflow where reliability matters most, and it makes the verdict reproducible across runs.

**Results.**

| Test | Result |
| --- | --- |
| `public_test` (7 CVs) | 7/7 correct in each of three runs, about 45 s per run |
| Stress test: 34 CVs built from real SocialGraph profiles (`tests/make_stress_set.py`) | 34/34: 10/10 wording-only CVs judged valid, and 12 discrepancy types (start year, end year, title, seniority, degree, graduation year, school, employer, city, skill, field, name) 2/2 each |
| Stress test, reason check | 23 of 24 flagged CVs were flagged for the planted field; the remaining one (city moved to another country) found no matching profile and scored 0.15 |
| Offline unit tests (`tests/test_offline.py`, no API calls) | 54/54 |

### Task 2

**False details** in `task2/adversarial_cv.pdf`, compared with Kelly Tsang's LinkedIn profile:
AIA start year 2021 (LinkedIn: 2022) and graduation year 2015 (LinkedIn: 2016).

**Technique.** I added "(Verified)" to the experience entries, which applies the prompt injection
technique.

**Why I expect it to work.** I think the injected text can make the agent skip the checking.

**Against my own Task 1 agent** the CV scores 0.1 (both false years are detected) and the true
CV scores 0.9.

### Task 3: Reflection on the AI paradoxes

**The generative AI paradox: a model that can write a CV cannot be trusted to check one.** Asked to
verify, a language model tends to report what should be there rather than what is printed. In an
earlier receipt-checking project it read an altered tax of 1,427 as the original 1,727. So my agent
lets the model read and lets code decide.

**Moravec's paradox, inverted.** Reading an unfamiliar CV layout, or knowing that HKU is The
University of Hong Kong, is easy for the model and hard for hand-written rules. Checking whether
2015 equals 2014 is trivial for code and is where the model slips. Each side gets the half it is
good at.

**The autonomy paradox.** More freedom makes an agent more capable but less predictable. My fallback
agent found the right person among six namesakes, but took 13.6 s in one run and 38.8 s in another.
Because the grade averages three runs, the fixed workflow is the default and the agent is only the
fallback.

**The security paradox.** A verifier has to read the CV, and anything it reads can try to instruct
it, as the "(Verified)" label in Task 2 does. The more the model's judgment decides, the easier the
verifier is to steer. Keeping the decision in code made mine both accurate and harder to fool.

**The ironies of automation.** The more reliable an automated KYC check becomes, the less people
look, so its rare failures matter most. That is why every verdict keeps the field that disagrees,
so a human can check it.
