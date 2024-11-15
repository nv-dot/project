"""
It allows you to define custom exceptions for specific errors in your application. 
By raising and catching specific exceptions, you can create more granular error-handling flows.
"""

import sys


# def error_message_detail(error,error_detail:sys): # The error is the error that we will get and error_detail is under sys
#     _,_,exc_tb = error_detail.exc_info() # Here exc_tb will have all the detail regarding the error
#     file_name = exc_tb.tb_frame.f_code.co_filename # By this step we will get the name of the file 
#     error_message = "Error occured in the python script name [{0}], line number [{1}] and error message was [{2}]".format(
#         file_name,exc_tb.tb_lineno,str(error)
#     )


# class errorexception(Exception):
#     def __init__(self,error_message,error_detail:sys):
#         super.__init__(error_message)
#         self.error_message = error_message_detail(error_message,error_detail=error_detail)
#     def __str__(self):
#         return self.error_message

import sys
import logging
from src.logger import logging

# Custom Exception Base Class
class MLBaseError(Exception):
    """Base class for all exceptions in the ML project."""
    pass

# Specific custom exception classes for different error types
class DataLoadingError(MLBaseError):
    """Raised when there is an issue with loading the dataset."""
    pass

class ModelTrainingError(MLBaseError):
    """Raised when there is an issue during model training."""
    pass

class PredictionError(MLBaseError):
    """Raised when there is an issue during prediction."""
    pass

# Function to handle exceptions and log details
def handle_exception(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Extract traceback info
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    error_message = f"Error occurred in file '{file_name}', line {line_number}: {error}"

    # Log the error message
    logging.error(error_message)
    
    # Print error message to console as well
    print(error_message)
