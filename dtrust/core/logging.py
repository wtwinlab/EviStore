"""
Logging configuration for DTRUST framework using Loguru.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import traceback


class DTRUSTLogger:
    """
    DTRUST logging system using Loguru with enhanced exception tracing.
    
    Features:
    - Beautiful colored console output
    - Comprehensive exception tracing with full stack traces
    - File logging with rotation
    - Structured logging support
    - Context-aware logging
    """
    
    def __init__(self):
        self._configured = False
        self._log_file_path: Optional[Path] = None
    
    def setup_logging(
        self, 
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        rotation: str = "10 MB",
        retention: str = "1 week",
        compression: str = "gz",
        enable_console: bool = True,
        enable_rich_tracebacks: bool = True,
        serialize: bool = False
    ) -> None:
        """
        Setup Loguru logging for DTRUST.
        
        Args:
            log_level: Logging level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            rotation: When to rotate log files
            retention: How long to keep old log files
            compression: Compression format for old log files
            enable_console: Whether to enable console logging
            enable_rich_tracebacks: Whether to enable rich exception formatting
            serialize: Whether to serialize logs as JSON
        """
        
        if self._configured:
            logger.warning("Logging already configured, skipping setup")
            return
        
        # Remove default handler
        logger.remove()
        
        # Console handler with rich formatting
        if enable_console:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            
            if enable_rich_tracebacks:
                console_format += "\n{exception}"
            
            logger.add(
                sys.stderr,
                format=console_format,
                level=log_level.upper(),
                colorize=True,
                backtrace=True,
                diagnose=True,
                catch=True
            )
        
        # File handler with rotation
        if log_file:
            self._log_file_path = Path(log_file)
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
            
            if serialize:
                # JSON format for structured logging
                logger.add(
                    str(self._log_file_path),
                    format=file_format,
                    level=log_level.upper(),
                    rotation=rotation,
                    retention=retention,
                    compression=compression,
                    serialize=True,
                    backtrace=True,
                    diagnose=True,
                    catch=True
                )
            else:
                # Plain text format
                logger.add(
                    str(self._log_file_path),
                    format=file_format + "\n{exception}",
                    level=log_level.upper(),
                    rotation=rotation,
                    retention=retention,
                    compression=compression,
                    backtrace=True,
                    diagnose=True,
                    catch=True
                )
        
        # Configure third-party loggers
        self._configure_third_party_loggers(log_level)
        
        self._configured = True
        logger.success(f"DTRUST logging configured - Level: {log_level.upper()}")
        
        if log_file:
            logger.info(f"Log file: {self._log_file_path}")
    
    def _configure_third_party_loggers(self, log_level: str):
        """Configure third-party library loggers."""
        import logging
        
        # Intercept standard logging and redirect to loguru
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                # Get corresponding Loguru level if it exists
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                
                # Find caller from where originated the logged message
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                
                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )
        
        # Set up interception for standard logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        
        # Configure specific third-party loggers
        third_party_loggers = [
            "uvicorn",
            "uvicorn.access",
            "uvicorn.error", 
            "fastapi",
            "sqlalchemy.engine",
            "httpx",
            "flwr",
            "asyncio"
        ]
        
        for logger_name in third_party_loggers:
            logging.getLogger(logger_name).handlers = [InterceptHandler()]
            logging.getLogger(logger_name).propagate = False
    
    def get_logger(self, name: str) -> "logger":
        """Get a logger instance with the given name."""
        return logger.bind(name=name)
    
    def log_exception(
        self, 
        exception: Exception, 
        message: str = "An exception occurred",
        context: Optional[Dict[str, Any]] = None,
        level: str = "ERROR"
    ):
        """
        Log an exception with full traceback and context.
        
        Args:
            exception: The exception to log
            message: Custom message to include
            context: Additional context information
            level: Log level (ERROR, CRITICAL, etc.)
        """
        
        # Get the full traceback
        tb_str = ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))
        
        # Create context string
        context_str = ""
        if context:
            context_str = f"\nContext: {context}"
        
        # Log with full details
        logger.opt(exception=True).log(
            level.upper(),
            f"{message}\n"
            f"Exception Type: {type(exception).__name__}\n"
            f"Exception Message: {str(exception)}"
            f"{context_str}\n"
            f"Full Traceback:\n{tb_str}"
        )
    
    def add_context(self, **kwargs) -> "logger":
        """Add context to logger."""
        return logger.bind(**kwargs)
    
    def catch_exceptions(self, reraise: bool = False):
        """
        Decorator to catch and log exceptions with full tracebacks.
        
        Args:
            reraise: Whether to reraise the exception after logging
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.log_exception(
                        e, 
                        f"Exception in {func.__name__}",
                        context={
                            "function": func.__name__,
                            "args": str(args)[:200],  # Limit length
                            "kwargs": str(kwargs)[:200]
                        }
                    )
                    if reraise:
                        raise
                    return None
            return wrapper
        return decorator
    
    def async_catch_exceptions(self, reraise: bool = False):
        """
        Async decorator to catch and log exceptions with full tracebacks.
        
        Args:
            reraise: Whether to reraise the exception after logging
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    self.log_exception(
                        e,
                        f"Exception in async {func.__name__}",
                        context={
                            "function": func.__name__,
                            "args": str(args)[:200],
                            "kwargs": str(kwargs)[:200]
                        }
                    )
                    if reraise:
                        raise
                    return None
            return wrapper
        return decorator


# Global logger instance
_dtrust_logger = DTRUSTLogger()

# Convenience functions
def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    **kwargs
) -> None:
    """Setup DTRUST logging."""
    _dtrust_logger.setup_logging(log_level, log_file, **kwargs)


def get_logger(name: str = None) -> "logger":
    """Get a logger instance."""
    if name:
        return _dtrust_logger.get_logger(name)
    return logger


def log_exception(
    exception: Exception,
    message: str = "An exception occurred", 
    context: Optional[Dict[str, Any]] = None,
    level: str = "ERROR"
):
    """Log an exception with full context."""
    _dtrust_logger.log_exception(exception, message, context, level)


def catch_exceptions(reraise: bool = False):
    """Decorator to catch and log exceptions."""
    return _dtrust_logger.catch_exceptions(reraise)


def async_catch_exceptions(reraise: bool = False):
    """Async decorator to catch and log exceptions."""
    return _dtrust_logger.async_catch_exceptions(reraise)


# Export the main logger
__all__ = [
    "setup_logging",
    "get_logger", 
    "log_exception",
    "catch_exceptions",
    "async_catch_exceptions",
    "logger"
] 