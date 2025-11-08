"""Logger configuration for datafast using loguru.

This module provides centralized logging configuration with support for both
console and file output.
"""

from loguru import logger
import sys


def configure_logger(
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
    colorize: bool = True,
    serialize: bool = False,
) -> None:
    """Configure the global logger for datafast.
    
    Args:
        level: Minimum log level (DEBUG, INFO, SUCCESS, WARNING, ERROR)
        log_file: Optional path to log file. If provided, logs will be written to both
                 console and file with automatic rotation.
        format_string: Custom format string (uses default if None)
        colorize: Enable colored output for console
        serialize: Output logs as JSON (useful for production monitoring)
    
    Examples:
        >>> # Default: INFO level, console only
        >>> configure_logger()
        
        >>> # With file logging
        >>> configure_logger(level="INFO", log_file="datafast.log")
        
        >>> # Debug mode with file
        >>> configure_logger(level="DEBUG", log_file="debug.log")
        
        >>> # Production: JSON format
        >>> configure_logger(level="WARNING", serialize=True, log_file="prod.log")
    """
    # Remove default handler
    logger.remove()
    
    if format_string is None:
        if serialize:
            # JSON format for production - add to both console and file
            logger.add(sys.stderr, serialize=True, level=level)
            if log_file:
                logger.add(
                    log_file,
                    serialize=True,
                    level=level,
                    rotation="10 MB",
                    retention="1 week",
                    compression="zip",
                )
        else:
            # Human-readable format
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
                "<level>{message}</level>"
            )
            
            # Console handler (colorized)
            logger.add(
                sys.stderr,
                format=format_string,
                level=level,
                colorize=colorize,
            )
            
            # File handler (no colors, with rotation)
            if log_file:
                logger.add(
                    log_file,
                    format=format_string,
                    level=level,
                    colorize=False,  # No ANSI colors in file
                    rotation="10 MB",  # Rotate when file reaches 10MB
                    retention="1 week",  # Keep logs for 1 week
                    compression="zip",  # Compress rotated logs
                )


# Initialize with default configuration
configure_logger()
