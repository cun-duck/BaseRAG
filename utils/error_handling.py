from functools import wraps
from utils.logger import logger

def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File error: {str(e)}")
            return [], {'error': 'File tidak ditemukan'}
        except Exception as e:
            logger.error(f"System error: {str(e)}")
            return [], {'error': 'Kesalahan sistem'}
        except Exception as e:
            error_msg = f"🚨 Error: {str(e)}"
            logger.error(error_msg)
            response = error_msg[:500]
            tokens = {
                "input_tokens": 0,
                 "output_tokens": 0,
                 "total_tokens": 0
    }
    return wrapper