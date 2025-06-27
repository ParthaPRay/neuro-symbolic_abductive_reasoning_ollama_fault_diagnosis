######## Neuro-symbolic Abductive Reasoning with Ollama Local Reasoning LLM
# Partha Pratim Ray, 27 June, 2025
# parthapratimray1986@gmail.com

# 1. Simple Bearing Fault: Prompt: There was a sudden increase in bearing temperature along with persistent vibration at 0.10 Hz.

#2. Sensor Noise Event: Prompt: The vibration amplitude sharply peaked at 0.11 Hz while temperature readings stayed close to normal.

#3. General Mechanical Stress: Prompt: Both temperature and vibration amplitudes gradually increased over the last 30 minutes, with several small vibration bursts detected.

#4. Fluctuating Temperatures: Prompt: The temperature sensor reported several rapid fluctuations between 47°C and 53°C, but vibration data remained unremarkable.

#5. Simulated Healthy Condition: Prompt: Temperature and vibration readings remained close to baseline, with no significant peaks or bursts observed.

#6. Multi-symptom Event:Prompt: High temperature mean, high variance, and a strong vibration peak at 0.15 Hz were observed during the last operating cycle.

#7. Intermittent Fault: Prompt: Most readings were normal, but occasional high vibration bursts were noted, especially during startup and shutdown periods.

#8. Ambiguous Vibration: Prompt: Vibration sensor readings show moderate, broad-spectrum activity with a faint peak at 0.13 Hz, and temperature is near the lower normal bound.

#9. Transient Burst:Prompt: All sensors were nominal except for a single, sharp vibration burst detected mid-cycle, with no corresponding temperature anomaly.

#10. Conflicting Signals:Prompt: Temperature mean is elevated with low variance, but vibration data shows only minor peaks scattered across the spectrum.

#11. Intermittent Sensor Issues:Prompt: Occasional missing values and outliers appear in both temperature and vibration logs, with one period showing a false high reading.

#12. Noisy, Unclear Patterns: Prompt: Vibration signals are highly irregular, with bursts and no consistent peak; temperature fluctuates randomly but stays below critical limits.

#13. Simultaneous Multiple Faults: Prompt: Significant temperature rise and multiple strong vibration peaks at 0.11 Hz and 0.15 Hz were observed over a short window.

#14. Sensor Calibration Drift: Prompt: There was a gradual increase in both mean temperature and vibration amplitude over several hours; baseline readings seem to have shifted.

#15. Startup Spike Only: Prompt: At startup, both temperature and vibration spiked briefly before stabilizing to nominal values for the remainder of the operation.

#16. Bursts with Low Average: Prompt: Average vibration remained low, but several isolated, high-magnitude bursts occurred randomly throughout the session.

#17. Perfectly Healthy / Null Case: Prompt: All sensor readings stayed strictly within normal operating ranges for the entire duration, with no anomalies detected.


import requests
import json
import numpy as np
import time
import csv
import os
import re
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "deepseek-r1:8b"   # Change as needed: "qwen3:8b" etc.
LOGFILE = "llm_abduction_log.csv"

HYPOTHESES = {
    'bearing_wear': {'mu':50, 'A':5,   'f':0.10, 'B':2},
    'vibration_resonance': {'mu':51, 'A':5.5, 'f':0.11, 'B':2.1},
    'temperature_fluctuation': {'mu':48, 'A':4.8, 'f':0.13, 'B':1.9},
    'sensor_noise_interference': {'mu':52, 'A':3.9, 'f':0.14, 'B':2.3},
    'mechanical_stress': {'mu':49, 'A':6, 'f':0.15, 'B':2.4},
    'unbalanced_rotor': {'mu':50, 'A':6.2, 'f':0.12, 'B':2.2},
    'loose_coupling': {'mu':51, 'A':4.2, 'f':0.09, 'B':2.0},
    'electrical_fault': {'mu':53, 'A':4.9, 'f':0.16, 'B':2.1},
    'calibration_drift': {'mu':47, 'A':4.5, 'f':0.08, 'B':1.8},
    'no_fault': {'mu':50, 'A':5, 'f':0.12, 'B':2},
    # Add more as needed!
}

METRIC_KEYS = [
    "total_duration", "load_duration", "prompt_eval_count",
    "prompt_eval_duration", "eval_count", "eval_duration", "tokens_per_second"
]

def ensure_logfile():
    if not os.path.exists(LOGFILE):
        with open(LOGFILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "llm", "user_event_desc",
                "final_hypotheses", "final_explanations", "final_priors",
                "most_likely_fault", "full_response_json"
            ] + 
            [f"{k}_hypothesis" for k in METRIC_KEYS] +
            [f"{k}_explanation" for k in METRIC_KEYS] +
            [f"{k}_prior" for k in METRIC_KEYS]
            )

def extract_metrics(api_resp):
    try:
        eval_count = api_resp.get("eval_count", 0)
        eval_duration = api_resp.get("eval_duration", 0)
        tokens_per_second = (
            eval_count / eval_duration * 1e9 if eval_duration else 0.0
        )
        return [
            api_resp.get("total_duration", 0),
            api_resp.get("load_duration", 0),
            api_resp.get("prompt_eval_count", 0),
            api_resp.get("prompt_eval_duration", 0),
            eval_count,
            eval_duration,
            tokens_per_second
        ]
    except Exception:
        return [0]*7

def log_event_result(event_cache):
    ts = datetime.now().isoformat()
    row = [
        ts,
        OLLAMA_MODEL,
        event_cache["user_event_desc"],
        json.dumps(event_cache["final_hypotheses"], ensure_ascii=False),
        json.dumps(event_cache["final_explanations"], ensure_ascii=False),
        json.dumps(event_cache["final_priors"], ensure_ascii=False),
        event_cache["most_likely_fault"],
        json.dumps(event_cache, ensure_ascii=False)
    ]
    row += event_cache.get("metrics_hypothesis", [0]*7)
    row += event_cache.get("metrics_explanation", [0]*7)
    row += event_cache.get("metrics_prior", [0]*7)
    with open(LOGFILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def prompt_llm_api_chat(messages, think=True):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "think": think,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
        api_resp = resp.json()
        message = api_resp.get("message", {})
    except Exception as ex:
        print(f"Ollama API error:", ex)
        api_resp = {}
        message = {}
    return message, api_resp

def extract_json_list(text):
    """
    Extracts the first JSON array from a string (even if surrounded by markdown, commentary, etc).
    Returns the loaded Python list, or [] if none found.
    """
    # Try to find [ ... ] (non-greedy), possibly multi-line
    matches = re.findall(r"\[[^\[\]]+\]", text, flags=re.DOTALL)
    for match in matches:
        try:
            arr = json.loads(match)
            if isinstance(arr, list):
                return arr
        except Exception:
            continue
    # Try last resort: split by comma if all else fails (very forgiving)
    return []

def prompt_for_hypotheses(user_event_desc, K=5, known_labels=None, retry=2, event_cache=None):
    system_msg = {
        "role": "system",
        "content": f"You are a machine health expert. Given a sensor event, return plausible fault labels as a JSON array, using only this set: {list(known_labels)}."
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Observed sensor event: {user_event_desc}\n"
            f"Based on this, propose {K} plausible FAULT LABELS (short strings, no explanation). "
            f"Return ONLY a valid JSON array, e.g., [\"bearing_wear\", \"vibration_resonance\"]"
        )
    }
    for attempt in range(retry):
        messages = [system_msg, user_msg]
        message, api_resp = prompt_llm_api_chat(messages, think=True)
        print("\n--- LLM THINKING (Hypotheses) ---\n", message.get("thinking", ""))
        text = message.get("content", "").strip()
        hyps = extract_json_list(text)
        if known_labels:
            hyps = [h for h in hyps if h in known_labels]
        hyps = [h for i, h in enumerate(hyps) if h and h not in hyps[:i]]
        if hyps:
            print(f"DEBUG: Proposed candidates = {hyps}")
            if event_cache is not None:
                event_cache["hypothesis_response"] = {
                    "thinking": message.get("thinking", ""),
                    "content": text,
                    "raw_api": api_resp
                }
                event_cache["metrics_hypothesis"] = extract_metrics(api_resp)
            return hyps[:K]
        else:
            print("WARNING: LLM response could not be parsed or was empty, retrying...")
            time.sleep(1)
    print("ERROR: Could not obtain valid hypotheses from LLM after retries.")
    if event_cache is not None:
        event_cache["metrics_hypothesis"] = [0]*7
    return []

def prompt_for_explanations(hyps, user_event_desc, event_cache=None):
    explanations = {}
    explanation_details = {}
    total_metrics = np.zeros(7)
    for h in hyps:
        system_msg = {"role": "system", "content": "You are a machine fault diagnosis expert."}
        user_msg = {"role": "user", "content": f"For the event: {user_event_desc}\nExplain briefly (2-3 sentences), in technical terms, how the hypothesis \"{h}\" could explain the observed data."}
        messages = [system_msg, user_msg]
        message, api_resp = prompt_llm_api_chat(messages, think=True)
        print(f"\n--- LLM THINKING (Explanation for {h}) ---\n{message.get('thinking', '')}")
        explanations[h] = message.get("content", "").strip().replace("\n", " ")
        explanation_details[h] = {
            "thinking": message.get("thinking", ""),
            "content": explanations[h],
            "raw_api": api_resp
        }
        total_metrics += np.array(extract_metrics(api_resp))
    # Average metrics for explanations (or sum, as needed)
    avg_metrics = (total_metrics / max(len(hyps), 1)).tolist()
    if event_cache is not None:
        event_cache["explanation_responses"] = explanation_details
        event_cache["metrics_explanation"] = avg_metrics
    return explanations

def llm_log_prior(hypo, user_event_desc, event_cache=None):
    system_msg = {"role": "system", "content": "You are a machine fault diagnosis expert."}
    user_msg = {"role": "user", "content": f'Given the event: {user_event_desc}\nYou proposed "{hypo}". On a scale 0–1, how likely is this fault a priori? Reply with a single number between 0 and 1, no explanation.'}
    messages = [system_msg, user_msg]
    message, api_resp = prompt_llm_api_chat(messages, think=True)
    print(f"\n--- LLM THINKING (Prior for {hypo}) ---\n{message.get('thinking', '')}")
    raw = message.get("content", "").strip()
    try:
        if raw.startswith('"') or raw.startswith("'"):
            num = float(json.loads(raw))
        else:
            num = float(raw)
    except Exception:
        m = re.search(r"\b([0-9]+(?:\.[0-9]*)?)\b", raw)
        if m:
            try:
                num = float(m.group(1))
            except Exception:
                print(f"WARNING: Unable to parse LLM prior from string: {raw!r}. Defaulting to 1e-6.")
                num = 1e-6
        else:
            print(f"WARNING: LLM prior non-numeric response: {raw!r}. Defaulting to 1e-6.")
            num = 1e-6
    if event_cache is not None:
        event_cache.setdefault("prior_responses", {})[hypo] = {
            "thinking": message.get("thinking", ""),
            "content": raw,
            "numeric_prior": num,
            "raw_api": api_resp
        }
    # For prior metrics, just use the first one (for top hypo)
    if event_cache is not None and hypo == event_cache.get("final_hypotheses", [None])[0]:
        event_cache["metrics_prior"] = extract_metrics(api_resp)
    return np.log(max(min(num, 1.0), 1e-6))

def report_result(user_event_desc, ranked, explanations, priors):
    print("\n--- Fault Diagnosis Report ---")
    print(f"Event Description: {user_event_desc}")
    print(f"\n{'Rank':<5}{'Hypothesis':<26}{'LLM Log-Prior':<14}{'Explanation'}")
    print("-"*85)
    for i, h in enumerate(ranked):
        print(f"{i+1:<5}{h:<26}{priors[h]:<14.3f}{explanations[h]}")
    print("-"*85)
    print(f"\nMost likely fault: {ranked[0]}")

def interactive_mode():
    ensure_logfile()
    print("=== Edge Abductive LLM Fault Diagnosis (with LLM Thinking) ===")
    user_event_desc = input("Describe the simulated or real sensor event (free text):\n> ")
    event_cache = {
        "user_event_desc": user_event_desc
    }
    candidates = prompt_for_hypotheses(user_event_desc, K=5, known_labels=HYPOTHESES.keys(), event_cache=event_cache)
    if not candidates:
        print("No hypotheses found.")
        return
    explanations = prompt_for_explanations(candidates, user_event_desc, event_cache=event_cache)
    priors = {}
    for h in candidates:
        priors[h] = llm_log_prior(h, user_event_desc, event_cache=event_cache)
    ranked = sorted(candidates, key=lambda h: -priors[h])
    report_result(user_event_desc, ranked, explanations, priors)
    # Store all event results
    event_cache["final_hypotheses"] = candidates
    event_cache["final_explanations"] = explanations
    event_cache["final_priors"] = priors
    event_cache["most_likely_fault"] = ranked[0] if ranked else ""
    log_event_result(event_cache)

if __name__ == "__main__":
    interactive_mode()
