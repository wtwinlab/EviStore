"""
Configuration management for DTRUST framework.
Supports YAML, JSON, and environment variable configuration.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

from .exceptions import DTRUSTError


@dataclass
class SGXConfig:
    """SGX-specific configuration.

    Attestation mode selection (mutually exclusive, checked in order):
      mock_mode=True                          → MockSGXQuoteGenerator + MockAttestationService
                                                纯 Python 模拟，不需要 SGX 硬件，开发/CI 用
      mock_mode=False, use_gramine=True       → GramineSGXQuoteGenerator (进程必须在 gramine-sgx 内)
                                                + DCAPAttestationService (host 上 DCAP 验证)
                                                整个 FL 进程跑在 enclave 里用这个
      mock_mode=False, use_gramine_subprocess=True
                                              → SubprocessGramineSGXQuoteGenerator (host 进程，
                                                attestation 时起轻量 gramine-sgx 子进程拿 quote)
                                                + DCAPAttestationService
                                                推荐：FL 训练在 host，attestation 用真实 SGX
      mock_mode=False (其他)                  → SGXQuoteGenerator (构造占位 quote，非真实硬件)
                                                + AttestationService (占位验证)
    """
    enabled: bool = True
    mock_mode: bool = False
    use_gramine: bool = False
    use_gramine_subprocess: bool = False   # FL 在 host，attestation 起 gramine-sgx 子进程
    gramine_attest_dir: str = ""           # dtrust_attest.manifest.sgx 所在目录，空则自动检测
    enclave_path: Optional[str] = None
    spid: Optional[str] = None
    primary_key: Optional[str] = None
    ias_url: str = "https://api.trustedservices.intel.com/sgx/attestation/v4"


@dataclass 
class TrustConfig:
    """Trust management configuration."""
    min_trust_score: float = 0.7
    trust_decay_rate: float = 0.1
    attestation_validity_hours: int = 24
    max_suspicious_count: int = 3
    auto_isolation: bool = True


@dataclass
class FLConfig:
    """Federated Learning configuration."""
    port: int = 8001  # Port for Flower FL server
    min_clients: int = 2
    max_clients: int = 100
    rounds: int = 10
    fraction_fit: float = 0.8
    fraction_evaluate: float = 0.8
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "postgresql://dtrust:dtrust@localhost:5432/dtrust"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = None
    rotation: str = "10 MB"
    retention: str = "1 week"
    compression: str = "gz"
    enable_console: bool = True
    enable_rich_tracebacks: bool = True
    serialize: bool = False


@dataclass
class Config:
    """Main configuration class."""
    sgx: SGXConfig = field(default_factory=SGXConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()
        
        if "sgx" in data:
            config.sgx = SGXConfig(**data["sgx"])
        if "trust" in data:
            config.trust = TrustConfig(**data["trust"])
        if "fl" in data:
            config.fl = FLConfig(**data["fl"])
        if "database" in data:
            config.database = DatabaseConfig(**data["database"])
        if "api" in data:
            config.api = APIConfig(**data["api"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
            
        # Top-level settings
        for key in ["environment", "debug", "log_level"]:
            if key in data:
                setattr(config, key, data[key])
                
        return config
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Create config from file (alias for load_config)."""
        return load_config(config_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "sgx": self.sgx.__dict__,
            "trust": self.trust.__dict__,
            "fl": self.fl.__dict__,
            "database": self.database.__dict__,
            "api": self.api.__dict__,
            "logging": self.logging.__dict__,
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level
        }


# Alias for backward compatibility
DTRUSTConfig = Config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file and environment variables.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Config: Loaded configuration
        
    Raises:
        DTRUSTError: If configuration loading fails
    """
    # Load environment variables
    load_dotenv()
    
    config_data = {}
    
    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise DTRUSTError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise DTRUSTError(f"Unsupported config file format: {config_path.suffix}")
        except Exception as e:
            raise DTRUSTError(f"Failed to load config file {config_path}: {e}")
    
    # Override with environment variables
    env_overrides = _load_env_overrides()
    _deep_update(config_data, env_overrides)
    
    return Config.from_dict(config_data)


def _load_env_overrides() -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    overrides = {}
    
    # Environment mapping
    env_mapping = {
        "DTRUST_ENVIRONMENT": ("environment",),
        "DTRUST_DEBUG": ("debug",),
        "DTRUST_LOG_LEVEL": ("log_level",),
        
        # SGX settings
        "DTRUST_SGX_ENABLED": ("sgx", "enabled"),
        "DTRUST_SGX_MOCK_MODE": ("sgx", "mock_mode"),
        "DTRUST_SGX_ENCLAVE_PATH": ("sgx", "enclave_path"),
        "DTRUST_SGX_SPID": ("sgx", "spid"),
        "DTRUST_SGX_PRIMARY_KEY": ("sgx", "primary_key"),
        
        # Trust settings
        "DTRUST_TRUST_MIN_SCORE": ("trust", "min_trust_score"),
        "DTRUST_TRUST_DECAY_RATE": ("trust", "trust_decay_rate"),
        "DTRUST_TRUST_VALIDITY_HOURS": ("trust", "attestation_validity_hours"),
        
        # FL settings
        "DTRUST_FL_PORT": ("fl", "port"),
        "DTRUST_FL_MIN_CLIENTS": ("fl", "min_clients"),
        "DTRUST_FL_MAX_CLIENTS": ("fl", "max_clients"),
        "DTRUST_FL_ROUNDS": ("fl", "rounds"),
        
        # Database settings
        "DTRUST_DATABASE_URL": ("database", "url"),
        "DTRUST_DATABASE_POOL_SIZE": ("database", "pool_size"),
        
        # API settings
        "DTRUST_API_HOST": ("api", "host"),
        "DTRUST_API_PORT": ("api", "port"),
        "DTRUST_API_WORKERS": ("api", "workers"),
        
        # Logging settings
        "DTRUST_LOGGING_LEVEL": ("logging", "level"),
        "DTRUST_LOGGING_FILE": ("logging", "file"),
        "DTRUST_LOGGING_ROTATION": ("logging", "rotation"),
        "DTRUST_LOGGING_RETENTION": ("logging", "retention"),
        "DTRUST_LOGGING_ENABLE_CONSOLE": ("logging", "enable_console"),
        "DTRUST_LOGGING_ENABLE_RICH_TRACEBACKS": ("logging", "enable_rich_tracebacks"),
    }
    
    for env_var, path in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            # Type conversion
            if env_var.endswith(("_ENABLED", "_DEBUG", "_MOCK_MODE", "_AUTO_ISOLATION", "_ENABLE_CONSOLE", "_ENABLE_RICH_TRACEBACKS")):
                value = value.lower() in ("true", "1", "yes", "on")
            elif env_var.endswith(("_PORT", "_WORKERS", "_CLIENTS", "_ROUNDS", "_HOURS", "_COUNT", "_SIZE")):
                value = int(value)
            elif env_var.endswith(("_SCORE", "_RATE", "_FRACTION")):
                value = float(value)
                
            _set_nested_value(overrides, path, value)
    
    return overrides


def _set_nested_value(data: Dict[str, Any], path: tuple, value: Any):
    """Set nested dictionary value using path tuple."""
    current = data
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]):
    """Deep update dictionary."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value 