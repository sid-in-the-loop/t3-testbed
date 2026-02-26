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
    f1: float = 0.0                          # F1 score against ground truth

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
        """Serialize to JSON-compatible dict with conditional fields."""
        data = {
            "question_id": self.question_id,
            "question": self.question,
            "answer_gt": self.answer_gt,
            "final_answer": self.final_answer,
            "em": self.em,
            "f1": self.f1,
            "turns_used": self.turns_used,
            "search_calls_used": self.search_calls_used,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "method": self.method,
            "config_id": self.config_id,
            "error": self.error,
            "turns": []
        }

        for t in self.turns:
            turn_data = {
                "turn": t.turn,
                "answer_found": t.answer_found,
                "prompt_tokens": t.prompt_tokens,
                "output_tokens": t.output_tokens,
            }
            
            if self.method == "s1":
                turn_data["reasoning"] = t.reasoning
                turn_data["search_queries"] = t.search_queries
            else:
                # Parallel/T³ methods
                turn_data["thread_queries"] = t.thread_queries
                turn_data["thread_results"] = t.thread_results
                turn_data["thread_summaries"] = t.thread_summaries
                turn_data["parent_response"] = t.parent_response
                
            data["turns"].append(turn_data)
            
        return data


_PLACEHOLDER_ANSWERS = {"[answer]", "your answer here", "your complete answer here",
                        "write your complete answer here", "[your actual answer]", "[answer]"}


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from <answer>...</answer> tags.
    Returns None if no tag found or if the content is a placeholder string
    (guards against prompt templates that contain literal example tags).
    """
    return extract_tag(text, "answer")


def extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content from <tag>...</tag> tags."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        if content.lower() in _PLACEHOLDER_ANSWERS:
            return None
        return content
    return None


def format_history_turn(query: str, information: str) -> str:
    """Format a search/information pair for the history turns block."""
    return f"<search> {query} </search>\n<information> {information} </information>"


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
    async def run_question(
        self, 
        question_id: str, 
        question: str, 
        answer_gt: str,
        pbar: Optional[any] = None,
    ) -> MethodResult:
        """
        Run the method on a single question.

        Args:
            question_id: Unique identifier for this question.
            question: The question text.
            answer_gt: Ground truth answer (used only for EM computation, not leaked to model).
            pbar: Optional tqdm progress bar for this question.

        Returns:
            MethodResult with final_answer, em, and detailed logs.
        """
        ...

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [{self.config.id}] {msg}")
