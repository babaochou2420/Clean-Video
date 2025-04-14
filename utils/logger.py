import logging
import os
import functools
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)


def setup_logger(name):
  """Setup a logger with file and console handlers"""
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)

  # Create handlers
  log_file = os.path.join(
      "logs", f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
  file_handler = logging.FileHandler(log_file)
  console_handler = logging.StreamHandler()

  # Create formatters and add it to handlers
  formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  console_handler.setFormatter(formatter)

  # Add handlers to the logger
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  return logger


def log_function(logger):
  """Decorator to log function entry and exit"""
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      logger.debug(f"Entering {func.__name__}")
      try:
        result = func(*args, **kwargs)
        logger.debug(f"Exiting {func.__name__}")
        return result
      except Exception as e:
        logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
        raise
    return wrapper
  return decorator
