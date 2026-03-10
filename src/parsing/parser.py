import sys
from typing import Dict
from src.parsing.config import ConfigModel



def parse_args() -> Dict[str: str, str: ConfigModel | None]:
	if (len(sys.argv) == 1):
		return {'success': 'false', 'config': None}
	try:
		
	except ValidationError as e:
		print("Error while validating:", e)