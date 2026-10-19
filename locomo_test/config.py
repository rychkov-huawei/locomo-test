"""TOML configuration loading and validation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


class SessionPolicy(Enum):
    ISOLATED = "isolated"
    SHARED = "shared"


@dataclass
class GatewayEnv:
    port: int = 19790
    token: str = ""
    base_url: str = ""  # derived from port if empty
    state_dir: str = ""  # OpenClaw state dir, derived from OPENCLAW_STATE_DIR or ~/.openclaw

    def __post_init__(self):
        if not self.base_url:
            self.base_url = f"http://localhost:{self.port}"


@dataclass
class OpenVikingEnv:
    port: int = 2936
    api_url: str = ""  # derived from port if empty
    api_key: str = ""

    def __post_init__(self):
        if not self.api_url:
            self.api_url = f"http://localhost:{self.port}"


@dataclass
class OgmemEnv:
    port: int = 8090
    api_url: str = ""  # derived from port if empty
    docker_container: str = "ogmem"
    wait_timeout: int = 900
    wait_interval: float = 2.0
    log_tail: int = 500
    chunk_chars: int = 2400
    account_id: str = ""
    user_id: str = ""
    api_key: str = ""

    def __post_init__(self):
        if not self.api_url:
            self.api_url = f"http://localhost:{self.port}"
        if self.chunk_chars < 512:
            raise ValueError("ogmem.chunk_chars must be at least 512")
        if self.wait_timeout <= 0:
            raise ValueError("ogmem.wait_timeout must be positive")


@dataclass
class JudgeEnv:
    api_key: str = ""
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    model: str = ""
    parallel: int = 5
    api_format: str | None = None  # auto-detect from base_url


@dataclass
class SessionConfig:
    policy: SessionPolicy = SessionPolicy.ISOLATED
    tail: str = "[]"


@dataclass
class StepsConfig:
    health_check: bool = True
    ingest: bool = True
    qa: bool = True
    judge: bool = True
    stats: bool = True


@dataclass
class ChecksConfig:
    fair_config: bool = False
    bootstrap_contamination: bool = False
    accuracy_sanity: bool = True


@dataclass
class Config:
    # general
    name: str = ""
    dataset: str = "locomo10"  # small / locomo10
    data_file: str = ""  # explicit path overrides dataset
    samples: list[int] | None = None
    count: int | None = None
    parallel: int = 32
    user: str = "eval-1"
    agent_id: str = "main"
    memory_mode: str = "openviking"
    output_dir: str = "output"

    # sub-configs
    gateway: GatewayEnv = field(default_factory=GatewayEnv)
    openviking: OpenVikingEnv = field(default_factory=OpenVikingEnv)
    ogmem: OgmemEnv = field(default_factory=OgmemEnv)
    judge_env: JudgeEnv = field(default_factory=JudgeEnv)
    session: SessionConfig = field(default_factory=SessionConfig)
    steps: StepsConfig = field(default_factory=StepsConfig)
    checks: ChecksConfig = field(default_factory=ChecksConfig)


def _parse_samples(raw) -> list[int] | None:
    """Parse samples: int, list[int], or range string like '0-4'."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        raw = raw.strip()
        if "-" in raw:
            parts = raw.split("-", 1)
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
        return [int(raw)]
    return None


def _get(d: dict, *keys, default=None):
    """Nested dict get."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def load_config(test_toml_path: str) -> Config:
    """Load test config TOML, then merge env.toml referenced by env_file."""
    test_path = Path(test_toml_path).resolve()
    if not test_path.exists():
        print(f"Error: config file not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    with open(test_path, "rb") as f:
        test = tomllib.load(f)

    # Load env.toml
    env_file = test.get("general", {}).get("env_file", "env.toml")
    env_path = test_path.parent / env_file
    env: dict = {}
    if env_path.exists():
        with open(env_path, "rb") as f:
            env = tomllib.load(f)
    else:
        print(f"Error: env file not found: {env_path}", file=sys.stderr)
        print(f"  Copy env.toml.example to env.toml and fill in your settings.", file=sys.stderr)
        sys.exit(1)

    cfg = Config()

    # --- general ---
    g = test.get("general", {})
    cfg.name = g.get("name", cfg.name)
    cfg.dataset = g.get("dataset", cfg.dataset)
    cfg.data_file = g.get("data_file", cfg.data_file)
    raw_samples = g.get("samples", cfg.samples)
    cfg.samples = _parse_samples(raw_samples)
    cfg.count = g.get("count", cfg.count)
    cfg.parallel = g.get("parallel", cfg.parallel)
    cfg.user = g.get("user", cfg.user)
    cfg.agent_id = g.get("agent_id", cfg.agent_id)
    cfg.memory_mode = str(g.get("memory_mode", cfg.memory_mode)).lower()
    cfg.output_dir = g.get("output_dir", cfg.output_dir)

    # --- gateway (env.toml overridden by test.toml) ---
    gw_env = env.get("gateway", {})
    gw_test = test.get("gateway", {})
    gw = {**gw_env, **gw_test}
    cfg.gateway = GatewayEnv(
        port=gw.get("port", 19790),
        token=gw.get("token", os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")),
        base_url=gw.get("base_url", ""),
        state_dir=gw.get("state_dir", ""),
    )

    # --- openviking ---
    ov_env = env.get("openviking", {})
    ov_test = test.get("openviking", {})
    ov = {**ov_env, **ov_test}
    cfg.openviking = OpenVikingEnv(
        port=ov.get("port", 2936),
        api_url=ov.get("api_url", ""),
        api_key=os.environ.get("OPENVIKING_API_KEY") or ov.get("api_key", ""),
    )

    # --- ogmem ---
    og_env = env.get("ogmem", {})
    og_test = test.get("ogmem", {})
    og = {**og_env, **og_test}
    cfg.ogmem = OgmemEnv(
        port=og.get("port", 8090),
        api_url=og.get("api_url", ""),
        docker_container=og.get("docker_container", "ogmem"),
        wait_timeout=og.get("wait_timeout", 900),
        wait_interval=og.get("wait_interval", 2.0),
        log_tail=og.get("log_tail", 500),
        chunk_chars=og.get("chunk_chars", 2400),
        account_id=og.get("account_id", ""),
        user_id=og.get("user_id", ""),
        api_key=og.get("api_key") or os.environ.get("OGMEM_API_KEY", ""),
    )

    # --- judge ---
    j_env = env.get("judge", {})
    j_test = test.get("judge", {})
    j = {**j_env, **j_test}
    cfg.judge_env = JudgeEnv(
        api_key=j.get("api_key", os.environ.get("ARK_API_KEY", "")),
        base_url=j.get("base_url", "https://ark.cn-beijing.volces.com/api/v3"),
        model=j.get("model", ""),
        parallel=j.get("parallel", 5),
        api_format=j.get("api_format"),
    )

    # --- session ---
    s = test.get("session", {})
    policy_str = s.get("policy", "isolated")
    cfg.session = SessionConfig(
        policy=SessionPolicy(policy_str),
        tail=s.get("tail", "[]"),
    )

    # --- steps ---
    st = test.get("steps", {})
    cfg.steps = StepsConfig(
        health_check=st.get("health_check", True),
        ingest=st.get("ingest", True),
        qa=st.get("qa", True),
        judge=st.get("judge", True),
        stats=st.get("stats", True),
    )

    # --- checks ---
    ch = test.get("checks", {})
    cfg.checks = ChecksConfig(
        fair_config=ch.get("fair_config", False),
        bootstrap_contamination=ch.get("bootstrap_contamination", False),
        accuracy_sanity=ch.get("accuracy_sanity", True),
    )

    # --- validation ---
    errors = []
    if not cfg.gateway.token:
        errors.append("gateway.token is required (set in env.toml or OPENCLAW_GATEWAY_TOKEN env var)")
    if not cfg.gateway.port:
        errors.append("gateway.port is required (set in env.toml)")
    valid_memory_modes = {"openviking", "ogmem", "memcore", "none"}
    if cfg.memory_mode not in valid_memory_modes:
        errors.append(f"general.memory_mode must be one of {sorted(valid_memory_modes)}")
    if cfg.memory_mode == "openviking" and not cfg.openviking.port:
        errors.append("openviking.port is required (set in env.toml)")
    if cfg.memory_mode == "ogmem" and not cfg.ogmem.api_url:
        errors.append("ogmem.api_url is required (set in env.toml)")
    if not cfg.judge_env.api_key:
        errors.append("judge.api_key is required (set in env.toml or ARK_API_KEY env var)")
    if not cfg.judge_env.model:
        errors.append("judge.model is required (set in env.toml)")
    if not cfg.gateway.state_dir:
        errors.append("gateway.state_dir is required (set in env.toml, e.g. OpenClaw state directory)")

    if errors:
        print("Config validation errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    return cfg
