"""
app/eval — automated model evaluation and quality gating.

Modules:
    scorer      Test-set loss and exact-match QA
    judge       LLM-as-judge comparative eval (base vs adapter)
    runner      Orchestrate eval runs and produce JSON reports
"""
from .scorer import eval_loss, eval_qa_accuracy
from .judge import llm_judge_compare
from .runner import run_eval
