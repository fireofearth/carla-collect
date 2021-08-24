"""
TODO: use custom exceptions
"""

class CollectException(Exception):
    pass

class GenerateException(CollectException):
    pass

class InSimulationException(CollectException):
    pass
