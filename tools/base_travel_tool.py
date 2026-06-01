import json
import os
from typing import Dict, List, Optional, Union

TOOL_REGISTRY = {}


def register_tool(name, allow_overwrite=False):
    def decorator(cls):
        if name in TOOL_REGISTRY and not allow_overwrite:
            raise ValueError(f"Tool `{name}` already exists")
        cls.name = name
        TOOL_REGISTRY[name] = cls
        return cls
    return decorator


class BaseTool:
    name: str = ''
    description: str = ''
    parameters: Union[List[dict], dict] = []

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        if not self.name:
            raise ValueError(
                f'You must set {self.__class__.__name__}.name before instantiation.'
            )

    def call(self, params: Union[str, dict], **kwargs):
        raise NotImplementedError

    def _verify_json_format_args(self, params: Union[str, dict], strict_json: bool = False) -> dict:
        if isinstance(params, str):
            params_json = json.loads(params)
        else:
            params_json = params

        if isinstance(self.parameters, list):
            for param in self.parameters:
                if param.get('required') and param['name'] not in params_json:
                    raise ValueError(f"Parameters {param['name']} is required!")
        return params_json

PANDAS_AVAILABLE = None


def load_tool_schemas(schema_file: Optional[str] = None, language: str = 'en') -> Dict[str, dict]:
    """Load tool definitions from the English JSON schema file."""
    if language != 'en':
        raise ValueError(f"Unsupported language: {language}. This release is English-only.")

    if schema_file is None:
        schema_file = f'tool_schema_{language}.json'
    
    if not os.path.exists(schema_file):
        schema_file = os.path.join(os.path.dirname(__file__), schema_file)
    
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Tool schema file not found: {schema_file}")
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        schemas_list = json.load(f)
    
    schemas = {}
    for schema in schemas_list:
        if 'function' in schema:
            tool_name = schema['function']['name']
            schemas[tool_name] = schema['function']
    
    return schemas


_TOOL_SCHEMAS_CACHE: Dict[str, Dict[str, dict]] = {}

def get_cached_tool_schemas(language: str = 'en') -> Dict[str, dict]:
    """Get cached English tool schemas."""
    global _TOOL_SCHEMAS_CACHE
    if language not in _TOOL_SCHEMAS_CACHE:
        _TOOL_SCHEMAS_CACHE[language] = load_tool_schemas(language=language)
    return _TOOL_SCHEMAS_CACHE[language]


class BaseTravelTool(BaseTool):
    """Base class for travel tools with schema loading and shared helpers."""
    
    def __init__(self, cfg: Optional[Dict] = None):
        """
        Initialize travel tool
        
        Args:
            cfg: Tool configuration dictionary, may contain:
                - database_path: Path to database file
                - load_schema: Whether to load schema from JSON (default True)
                - language: Language code ('en', default 'en')
        """
        if cfg is None:
            cfg = {}
        
        self.language = cfg.get('language', 'en')
        if self.language != 'en':
            raise ValueError(f"Unsupported language: {self.language}. This release is English-only.")
        
        if not self.__class__.description and cfg.get('load_schema', True):
            self._load_schema_from_json()
        
        super().__init__(cfg)
        
        self.database_path = None
        self.data = None
    
    def _load_schema_from_json(self):
        """Load tool schema from the English JSON file."""
        tool_name = None
        for name, cls in TOOL_REGISTRY.items():
            if cls == self.__class__:
                tool_name = name
                break
        
        if not tool_name:
            if hasattr(self.__class__, 'name') and self.__class__.name:
                tool_name = self.__class__.name
            else:
                return
        
        schemas = get_cached_tool_schemas(language=self.language)
        
        if tool_name in schemas:
            schema = schemas[tool_name]
            self.__class__.name = schema.get('name', tool_name)
            self.__class__.description = schema.get('description', '')
            self.__class__.parameters = schema.get('parameters', {})
    
    def load_csv_database(self, path: str):
        """
        Load CSV format database
        
        Args:
            path: File path
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: File does not exist
            ImportError: pandas not installed or import failed
        """
        global PANDAS_AVAILABLE
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Database file not found: {path}")
        
        if PANDAS_AVAILABLE is None:
            try:
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'
                os.environ['OPENBLAS_NUM_THREADS'] = '1'
                
                import pandas as pd
                PANDAS_AVAILABLE = pd
            except Exception as e:
                PANDAS_AVAILABLE = False
                raise ImportError(
                    f"pandas import failed: {e}\n"
                    "Please run: pip install pandas\n"
                    "Or use JSON format database"
                )
        
        if PANDAS_AVAILABLE is False:
            raise ImportError(
                "pandas not installed or import failed, cannot load CSV database.\n"
                "Please run: pip install pandas\n"
                "Or use JSON format database"
            )
        
        pd = PANDAS_AVAILABLE
        return pd.read_csv(path, dtype=str)
    
    def format_result_as_json(self, result: Union[dict, list]) -> str:
        """
        Format result as JSON string
        
        Args:
            result: Result data
            
        Returns:
            JSON formatted string
        """
        return json.dumps(result, ensure_ascii=False, indent=2)
    
__all__ = [
    'BaseTool',
    'BaseTravelTool',
    'register_tool',
    'TOOL_REGISTRY',
    'load_tool_schemas',
    'get_cached_tool_schemas',
]
