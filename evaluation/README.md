# Benchmark protocol

`benchmark_questions.json` is the frozen 60-question benchmark: 50 answerable questions with expected source pages and answers, plus 10 deliberately out-of-document refusal checks.

Run it from the repository root with the benchmark PDF:

```powershell
python evaluation/run_benchmark.py --pdf path\to\ml_ebook.pdf
```

The runner caches every completed question in `evaluation/results/`, then saves a summary with explicit denominators for retrieval, judge scores, and latency. It forces the application context window to 4096 tokens. Row 0 is recorded in `results/row0.json`; its full per-question output is also retained. The judge is local `llama3.2:3b`, the same model used to generate answers, so its ratings are not independent. Nine of 60 answer judgments were invalid and excluded; retrieval uses the 50 answerable questions, while latency averages use all 60.
