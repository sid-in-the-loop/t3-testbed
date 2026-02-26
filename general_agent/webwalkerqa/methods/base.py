"""
Base class for all WebWalkerQA methods (s1, T³ Fixed, T³ Dynamic, T³ DPP).

Each method implements run_question() which processes one question and returns
a MethodResult with the final answer and per-turn logs.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnLog:
    """Log for a single turn of reasoning."""
    turn: int
    # For T³ methods: per-thread info
    thread_queries: list[str] = field(default_factory=list)    # search query per thread
    thread_results: list[str] = field(default_factory=list)    # raw search results per thread
    thread_summaries: list[str] = field(default_factory=list)  # summaries produced by threads
    # For s1: single-thread trace
    reasoning: str = ""
    search_queries: list[str] = field(default_factory=list)
    # Parent synthesis
    parent_response: str = ""
    answer_found: bool = False
    # Token accounting
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MethodResult:
    """Result of running a method on one question."""
    question_id: str
    question: str
    answer_gt: str                           # Ground truth
    final_answer: str                        # Model's final answer
    em: bool = False                         # Exact match against ground truth

    # Per-turn logs
    turns: list[TurnLog] = field(default_factory=list)

    # Aggregate stats
    turns_used: int = 0
    search_calls_used: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0

    # Method metadata
    method: str = ""
    config_id: str = ""

    # Error (if run failed)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer_gt": self.answer_gt,
            "final_answer": self.final_answer,
            "em": self.em,
            "turns_used": self.turns_used,
            "search_calls_used": self.search_calls_used,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "method": self.method,
            "config_id": self.config_id,
            "error": self.error,
            "turns": [
                {
                    "turn": t.turn,
                    "thread_queries": t.thread_queries,
                    "thread_summaries": t.thread_summaries,
                    "reasoning": t.reasoning,
                    "search_queries": t.search_queries,
                    "parent_response": t.parent_response,
                    "answer_found": t.answer_found,
                    "prompt_tokens": t.prompt_tokens,
                    "output_tokens": t.output_tokens,
                }
                for t in self.turns
            ],
        }


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from <answer>...</answer> tags.
    Returns None if no tag found.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


class BaseMethod(ABC):
    """Abstract base for all methods."""

    def __init__(self, model: str, config, verbose: bool = False):
        """
        Args:
            model: LiteLLM model string.
            config: ExperimentConfig (from configs.py).
            verbose: Print per-turn debug output.
        """
        self.model = model
        self.config = config
        self.verbose = verbose

    @abstractmethod
    async def run_question(self, question_id: str, question: str, answer_gt: str) -> MethodResult:
        """
        Run the method on a single question.

        Args:
            question_id: Unique identifier for this question.
            question: The question text.
            answer_gt: Ground truth answer (used only for EM computation, not leaked to model).

        Returns:
            MethodResult with final_answer, em, and detailed logs.
        """
        ...

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [{self.config.id}] {msg}")
