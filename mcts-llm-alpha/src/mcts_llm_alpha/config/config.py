"""Configuration. Defaults follow the paper's experimental setup (Section 5.1, Appendix F).

Unknown keys are rejected so that a stale YAML file cannot silently fall back to
defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MCTSConfig(Strict):
    total_search_count: int = Field(1000, gt=0, description="Alpha generations: 1000 / 2000 / 3000")
    tree_initial_budget: int = Field(3, gt=0, description="Initial budget B of each search tree")
    budget_increment: int = Field(1, ge=0, description="b, added when a node sets a new tree best")
    exploration_constant: float = Field(1.0, description="UCT exploration weight c")
    dimension_temperature: float = Field(1.0, gt=0, description="Softmax temperature T")
    max_tree_iterations: int = Field(30, gt=0, description="Safety cap on one tree's iterations")
    max_corrections: int = Field(3, ge=0, description="Correction rounds for an invalid formula")
    max_root_attempts: int = Field(3, gt=0)
    initial_seed_formula: Optional[str] = Field(None, description="f_seed for the first tree")
    random_seed: int = 0


class LLMConfig(Strict):
    model: str = "gpt-4.1"
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = Field(600.0, gt=0)
    stream: bool = True
    context_window: int = Field(32768, gt=512)
    max_tokens: int = Field(32768, gt=0)
    min_output_tokens: int = Field(4096, gt=0)
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    # Per-role overrides merged into extra_body, e.g. disable thinking for the summary.
    extra_body_by_role: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    reasoning_effort: Optional[str] = None
    langfuse: str = "auto"
    # Appendix F.1: portraits/formulas 1.0, correction 0.8, overfitting review 0.1.
    temperature: Dict[str, float] = Field(default_factory=lambda: {
        "portrait": 1.0, "formula": 1.0, "suggestion": 1.0, "refinement": 1.0,
        "correction": 0.8, "review": 0.1, "summary": 0.7})
    max_retries: int = Field(3, ge=1)
    retry_delay: float = 1.0

    @field_validator("langfuse")
    @classmethod
    def check_langfuse(cls, v):
        if v not in {"auto", "on", "off"}:
            raise ValueError("langfuse must be auto, on or off")
        return v

    @field_validator("temperature", "extra_body_by_role")
    @classmethod
    def check_roles(cls, v):
        roles = {"portrait", "formula", "suggestion", "refinement", "correction", "review", "summary"}
        unknown = set(v) - roles
        if unknown:
            raise ValueError(f"unknown temperature roles {sorted(unknown)}; expected {sorted(roles)}")
        return v

    @model_validator(mode="after")
    def check_budget_and_key(self):
        from urllib.parse import urlparse
        if self.min_output_tokens + 320 >= self.context_window:
            raise ValueError("min_output_tokens must leave room for prompt and safety margin")
        if self.base_url and urlparse(self.base_url).hostname != "api.openai.com" \
                and self.api_key_env == "OPENAI_API_KEY":
            raise ValueError("Custom endpoint needs its own api_key_env; do not forward OPENAI_API_KEY")
        return self


class EvaluationConfig(Strict):
    horizon: int = Field(10, gt=0, description="Forward return horizon in trading days (10 or 30)")
    # Appendix F.1 effective alpha check
    rank_ic_min: float = 0.015
    rank_ir_min: float = 0.3
    relative_rank_max: float = 0.95
    relative_rank_min_size: int = Field(20, ge=1, description="R_RankIC/R_RankIR apply from this repository size")
    turnover_max: float = 1.6
    max_correlation: float = 0.8
    # Appendix C.2 / F.1 few-shot selection
    few_shot_k: int = Field(1, ge=0)
    correlation_filter_ratio: float = Field(0.5, ge=0, lt=1)
    final_alpha_counts: List[int] = Field(default_factory=lambda: [10, 50, 100])
    min_days: int = Field(50, gt=0, description="Minimum dates with a computable RankIC")
    min_coverage: float = Field(0.3, gt=0, le=1, description="Minimum share of label cells with a factor value")


class DataConfig(Strict):
    qlib_provider_uri: Optional[str] = None
    region: str = Field("cn", description="Qlib region of the dataset: cn or us")
    universe: str = "csi300"
    fields: List[str] = Field(default_factory=lambda: ["open", "high", "low", "close", "volume", "vwap"])
    train_period: Tuple[str, str] = ("2011-01-01", "2020-12-31")
    test_period: Tuple[str, str] = ("2021-01-01", "2024-11-30")
    data_revision: str = Field("unversioned", description="Change when the market data is revised")

    @field_validator("region")
    @classmethod
    def check_region(cls, v):
        if v not in {"cn", "us"}:
            raise ValueError("region must be cn or us")
        return v

    @model_validator(mode="after")
    def check_periods(self):
        if not self.train_period[0] < self.train_period[1] < self.test_period[0] <= self.test_period[1]:
            raise ValueError("train_period must end before test_period starts")
        return self

    def provider_uri(self):
        uri = self.qlib_provider_uri or os.getenv("QLIB_PROVIDER_URI")
        if not uri:
            raise ValueError("Set data.qlib_provider_uri or the QLIB_PROVIDER_URI environment variable")
        if not (Path(uri).expanduser() / "calendars" / "day.txt").exists():
            raise FileNotFoundError(
                f"No Qlib daily data at {uri} (calendars/day.txt is missing). Download and extract "
                "the data there, or point data.qlib_provider_uri at the folder that contains "
                "calendars/, instruments/ and features/.")
        return uri


class FSAConfig(Strict):
    top_k: int = Field(3, ge=0, description="Frequent closed subtrees to avoid (Appendix F.1)")
    min_support: int = Field(2, ge=1)


class OutputConfig(Strict):
    results_dir: str = "results"
    experiment_log_file: str = "logs/mcts_experiments.jsonl"


class Config(Strict):
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    fsa: FSAConfig = Field(default_factory=FSAConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    path = Path(config_path) if config_path else Path(__file__).parent / "default.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        return Config(**(yaml.safe_load(stream) or {}))


def save_config(config: Config, config_path: str) -> None:
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config.model_dump(mode="json"), stream, allow_unicode=True, sort_keys=False)
