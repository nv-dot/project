""" 
Will be used a handle logging, which is the process of 
recording events, errors, or informational messages that occur while a program runs.
This is especially useful for debugging, monitoring, and auditing the behavior of an application.
"""
import logging
import os

from src.exception import *


# Create a directory for logs if it doesn't exist
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define the log file name
log_file = os.path.join(log_dir, 'ml_project.log')

# Set up the logger
logger = logging.getLogger('MLLogger')
logger.setLevel(logging.DEBUG)

# Create a file handler that writes log messages to a file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a console handler for output to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define a log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Function to log messages
def log_message(level, message):
    if level == "info":
        logger.info(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
