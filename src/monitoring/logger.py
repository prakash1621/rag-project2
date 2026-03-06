"""
Structured logging setup for production monitoring
"""

import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Add contextual information to log records"""
    
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record"""
        if not hasattr(record, 'extra_fields'):
            record.extra_fields = {}
        record.extra_fields.update(self.context)
        return True


def setup_structured_logging(
    log_level: str = 'INFO',
    log_file: str = None,
    structured: bool = False,
    context: Dict[str, Any] = None
) -> logging.Logger:
    """
    Setup structured logging for production.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Whether to use JSON structured logging
        context: Additional context to add to all logs
        
    Returns:
        Configured logger
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if structured:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
        
        logger.addHandler(file_handler)
    
    # Add context filter
    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    logger.info(f"Logging configured: level={log_level}, structured={structured}, file={log_file}")
    
    return logger


def get_logger(name: str, extra_context: Dict[str, Any] = None) -> logging.Logger:
    """
    Get a logger with optional extra context.
    
    Args:
        name: Logger name (usually __name__)
        extra_context: Additional context for this logger
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if extra_context:
        context_filter = ContextFilter(extra_context)
        logger.addFilter(context_filter)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Adapter to add extra fields to log records"""
    
    def process(self, msg, kwargs):
        """Add extra fields to log record"""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        if not hasattr(kwargs['extra'], 'extra_fields'):
            kwargs['extra']['extra_fields'] = {}
        
        kwargs['extra']['extra_fields'].update(self.extra)
        
        return msg, kwargs
