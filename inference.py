import os
import json
import re
from openai import OpenAI
from server.environment import CodeReviewEnv
from models import CodeReviewAction
from tasks.task_definition import TASKS

# =========================
# ENV VARIABLES
# =========================
os.environ["MODEL_NAME"] = "gpt-5.4"
MODEL_NAME = os.getenv("MODEL_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY")

TASK_NAMES  = ["easy", "medium", "hard"]
MAX_STEPS = 5  # single-step env

# Initialize client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

import json
import ast

def safe_parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # fallback: handle single quotes or Python-style dicts
            return ast.literal_eval(content)
        except:
            # ultimate fallback
            return {
                "issues": ["unknown issue"],
                "severity": "low",
                "suggestion": "",
                "reasoning": ""
            }
# =========================
# Prompt builder
# =========================
def build_prompt(code: str, feedback: str):
    return f"""You are a professional Python code reviewer.

    Code:
    {code}

    {f"Previous feedback from grader (use this to improve): {feedback}" if feedback else ""}

    Instructions:
    - List specific bugs, security flaws, or major inefficiencies.
    - Format "issues" as a simple list of strings: ["issue 1", "issue 2"].
    - Do NOT repeat points the feedback already rejected.
    - Ensure the JSON is complete and properly closed.

    Return ONLY valid JSON:
    {{
    "issues": ["string", "string"],
    "severity": "low|medium|high",
    "suggestion": "How to fix the code",
    "reasoning": "Why these changes are necessary"
    }}
    """

# =========================
# Utilities
# =========================
def clean_predicted_issues(predicted_issues):
    """Handle dicts, stringified dicts, or raw strings"""
    cleaned = []
    for i in predicted_issues:
        if isinstance(i, dict):
            cleaned.append(i.get("issue") or i.get("description") or str(i))
        else:
            # handle stringified dicts like "{'issue': 'xyz', ...}"
            match = re.search(r"'issue':\s*'([^']+)'", i)
            if match:
                cleaned.append(match.group(1))
            else:
                cleaned.append(i)
    return cleaned

def contains_keyword(text, keywords):
    if not text:
        return False
    text = text.lower()
    return any(k.lower() in text for k in keywords)

from difflib import SequenceMatcher
def similar(a, b, threshold=0.5):
    """Return True if strings are sufficiently similar"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

# =========================
# Grader
# =========================
def grade(task, action: CodeReviewAction):
    if task is None:
        return 0.0

    expected = task.get("expected", {})
    issues = expected.get("issues", [])
    expected_severity = expected.get("severity", "low")
    fix_keywords = expected.get("fix_keywords", [])
    concepts = expected.get("concepts", [])

    predicted_issues = clean_predicted_issues(action.issues or [])
    score = 0.0

    # 1️⃣ Issue detection
    matched = sum(1 for exp in issues if any(similar(exp, p) for p in predicted_issues))
    if issues:
        score += 0.4 * (matched / len(issues))

    # 2️⃣ Severity check
    severity_map = {"low": 1, "medium": 2, "high": 3}
    pred_sev = severity_map.get((action.severity or "").lower(), 0)
    exp_sev = severity_map.get(expected_severity.lower(), 0)
    if pred_sev and exp_sev:
        if pred_sev == exp_sev:
            score += 0.2
        elif abs(pred_sev - exp_sev) == 1:
            score += 0.1

    # 3️⃣ Suggestion check
    if contains_keyword(action.suggestion, fix_keywords):
        score += 0.2

    # 4️⃣ Reasoning check
    if contains_keyword(action.reasoning, concepts):
        score += 0.2

    # 5️⃣ Hallucination penalty
    unrelated = [p for p in predicted_issues if not any(similar(p, e) for e in issues)]
    if unrelated:
        score -= 0.02 * len(unrelated)

    return max(0.0, min(score, 1.0))

# =========================
# Inference / Runner
# ========================

# =========================
# Runner / Inference (multi-step, OpenAI gpt-5.4 ready)
# =========================

SUCCESS_THRESHOLD = 0.35  # reward needed to consider task successful

def run_task(task_name: str):
    env = CodeReviewEnv()
    task = TASKS[task_name]
    
    #print(f"🐛 TARGET: {task_name}")
    #print(f"🐛 EXPECT:  {repr(task['code'][:60])}")
    
    # METHOD 1: Reset + FORCE observation refresh via dummy step
    obs = env.reset(seed=42, episode_id=task_name)
    
    # Monkey-patch observation DIRECTLY (works around _state bug)
    obs.code = task["code"]  
    obs.feedback = ""
    obs.score = 0.0
    obs.remaining_issues = task["expected"]["issues"].copy()
    
    # Update internal state too
    env._state.task_id = task_name
    env._state.task = task
    env._state.code = task["code"] 
    env._state.remaining_issues = task["expected"]["issues"].copy()
    
    #print(f"🐛 OBS NOW: {repr(obs.code[:60])}")
    
    feedback = ""
    rewards = []
    success = False
    
    try:
    # Rest of your loop unchanged...
        for step in range(1, MAX_STEPS + 1):
            # Your existing prompt/response/action code...

            prompt_text = build_prompt(obs.code, feedback)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0
            )

            #content = response.choices[0].message.content
            content = response.choices[0].message.content.strip()

            # remove accidental repetition (basic guard)
            content = content.split("}{")[0] + "}" if "}{" in content else content
            parsed = safe_parse_json(content)

            if parsed is None:
                print(f"[STEP] step={step} action=null reward=0.00 done=true error=parse_failed")
                break

            action = CodeReviewAction(**parsed)

            obs, reward, done, info = env.step(action)

            # 🔥 FIX 2: Override broken remaining/done logic
            unmatched = [i for i in task["expected"]["issues"] 
                        if not any(similar(i, p) for p in (action.issues or []))]
            obs.remaining_issues = unmatched
            done = len(unmatched) > 0  # Continue until ALL issues fixed

            rewards.append(reward)

            feedback = f"""
                Score: {obs.score}
                Feedback: {obs.feedback}
                Remaining issues: {obs.remaining_issues}
                """

            print(
                f"[STEP] step={step} "
                f"action={json.dumps(parsed)} "
                f"reward={reward:.2f} "
                f"done={str(done).lower()} "
                f"remaining={len(obs.remaining_issues)} "
                f"error=null"
            )

            if done:
                #success = len(obs.remaining_issues) == 0
                success = (reward >= 0.8) or (len(obs.remaining_issues) == 0)
                break

    except Exception as e:
        print(
            f"[STEP] step={step} action=null reward=0.00 done=true error={str(e)}"
        )
        success = False

    finally:
        env.close()

        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        print(
            f"[END] success={str(success).lower()} "
            f"steps={len(rewards)} "
            f"rewards={rewards_str}"
        )


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    for task in TASKS:
        run_task(task)