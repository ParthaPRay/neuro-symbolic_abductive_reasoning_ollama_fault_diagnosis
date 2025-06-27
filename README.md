# Neuro-Symbolic Abductive Reasoning with Ollama Local LLMs

**Author:** Partha Pratim Ray

**Contact:** [parthapratimray1986@gmail.com](mailto:parthapratimray1986@gmail.com)

**Date:** 27 June, 2025

---

## Introduction

This project is an implementation of **neuro-symbolic AI** for machine fault diagnosis, combining symbolic domain knowledge (engineering rules, hypotheses) with the powerful pattern recognition and language capabilities of local large language models (LLMs) through [Ollama](https://ollama.com/).
It applies **abductive reasoning**—a form of logical inference that seeks the best explanation for observed evidence—enabling transparent, step-wise diagnostic reasoning for sensor-rich environments such as industrial machinery and IoT deployments.

**Why Neuro-Symbolic and Abductive Reasoning?**
Traditional LLMs excel at pattern matching but struggle with explicit causal inference, logical explanation, and traceable diagnosis. By blending LLM “neural” capabilities with **symbolic hypothesis generation and evaluation**, this tool can:

* Generate hypotheses grounded in engineering knowledge,
* Use LLMs to explain and prioritize these hypotheses,
* Deliver transparent, auditable reasoning trails for every diagnosis.

---

## Key Features

* **Neuro-symbolic abduction:** Marries symbolic hypothesis spaces with LLM-powered neural reasoning, for robust “explainable AI” in fault analysis.
* **Local and private:** All reasoning runs on your own machine, leveraging Ollama’s local LLM serving—no cloud or data sharing required.
* **Supports Ollama ‘thinking’ LLMs:** Out-of-the-box compatibility with Qwen3, DeepSeek R1, and future models supporting the `think` API.
* **Step-wise inference:** For every sensor event, the system generates plausible faults, technical explanations, and Bayesian-style prior probabilities—all with detailed LLM “thinking.”
* **Full audit trail:** Logs every LLM response, explanation, and key metrics (duration, token rates, etc.) to CSV for further study and benchmarking.
* **Ready for real-world prompts:** Comes with a suite of 17 realistic, diverse event prompts for immediate demonstration and testing.

---

## Example Use Cases

Feed the tool descriptions such as:

* **Simple bearing fault:** Sudden bearing temperature increase with persistent vibration at 0.10 Hz.
* **Sensor noise event:** Vibration amplitude sharply peaks at 0.11 Hz, temperature stable.
* **Multi-symptom event:** High mean/variance in temperature and a strong vibration peak at 0.15 Hz.
* **Perfectly healthy/null case:** All sensors normal; no anomalies detected.

The tool performs **abductive reasoning** to infer, explain, and rank possible root causes—logging the complete thought process.

---

## How It Works

1. **Describe a sensor event:** The user enters an observation (natural language).
2. **Hypothesis generation:** The LLM, guided by engineering context, proposes plausible fault hypotheses (symbolic step).
3. **Neural reasoning:** For each hypothesis, the LLM explains the mechanism and estimates its likelihood (prior).
4. **All outputs are logged:** Explanations, reasoning traces, and evaluation metrics are stored in a CSV row for each case.
5. **Transparent reporting:** The diagnosis and full rationale are printed for user inspection.

---

## Getting Started

### Prerequisites

* **Python 3.11+**
* **Ollama** running locally ([installation guide](https://ollama.com/download))
* At least one supported model:

  * e.g., `ollama pull qwen3:8b` or `ollama pull deepseek-r1:8b`
* Python packages:

  ```bash
  pip install requests numpy
  ```

### Usage

1. Set the `OLLAMA_MODEL` variable in the code to your preferred model (e.g., `"deepseek-r1:8b"`).
2. Start the Ollama server and ensure your model is available.
3. Run the script:

   ```bash
   python app.py
   ```
4. Enter your sensor event description when prompted.
5. Review the diagnosis on-screen and in `llm_abduction_log.csv`.

---

## Output

* **Comprehensive CSV log:**
  Each event’s hypotheses, explanations, priors, model thinking, and performance metrics are logged for scientific analysis and reproducibility.

---

## Model Compatibility

Tested on and compatible with:

* **Qwen3 (Alibaba)**
* **DeepSeek R1**
* Any other Ollama model supporting the “thinking” (`think=true`) API.

---

## Why Use This?

* **Explainable neuro-symbolic AI** for safety-critical and industrial environments.
* **Abductive reasoning**: Go beyond pattern matching—find the best *explanation* for anomalies.
* **Transparent & auditable**: Every step, score, and rationale is logged and reproducible.
* **Private & offline**: No data leaves your computer.

---

## Reference

* [Thinking in Ollama (Blog)](https://ollama.com/blog/thinking)
* See code and output for more.

---

## Citation

If you use this tool for research or deployment, please cite:

> Partha Pratim Ray, “Neuro-symbolic Abductive Reasoning with Ollama Local Reasoning LLMs,” 2025.
> [github.com/ParthaPRay/llm-abduction-ollama](https://github.com/ParthaPRay/llm-abduction-ollama)

---

## License

MIT License

---

**For questions or collaboration:**
[parthapratimray1986@gmail.com](mailto:parthapratimray1986@gmail.com)

---

***Bridging neural and symbolic reasoning for real-world diagnostics.***

---
