"""
Modelos Pydantic para validação de dados
========================================

Modelos para validação dos dados de entrada da API SmartBI.

Autor: SmartBI Team
Versão: 2.0.0
Data: 2025
"""

from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, Dict, Any
from enum import Enum


class DatabaseType(str, Enum):
    """Tipos de base de dados suportados"""
    mysql = "mysql"
    postgresql = "postgresql"
    sqlite = "sqlite"
    mssql = "mssql"
    oracle = "oracle"


class DatabaseConnectionRequest(BaseModel):
    """Modelo para requisição de conexão com base de dados"""
    database_url: str
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v or not v.strip():
            raise ValueError('URL da base de dados é obrigatória')
        
        v = v.strip()
        
        # Validação básica de formato
        if '://' not in v:
            raise ValueError('URL deve conter o protocolo da base de dados (ex: postgresql://, mysql://)')
        
        # Exemplos de URLs válidas
        examples = [
            'postgresql://user:password@host:port/database',
            'mysql://user:password@host:port/database',
            'sqlite:///path/to/database.db'
        ]
        
        # Verifica se contém elementos básicos (exceto SQLite)
        if not v.startswith('sqlite://'):
            if '@' not in v:
                raise ValueError(f'URL deve conter credenciais de autenticação. Exemplos: {examples}')
        
        return v


class FileUploadInfo(BaseModel):
    """Informações sobre upload de arquivo"""
    filename: str
    size: int
    type: str
    processed_at: Optional[str] = None


class DatabaseConnectionInfo(BaseModel):
    """Informações sobre conexão com base de dados"""
    status: str
    database_type: str
    host: str
    port: Optional[int] = None
    database: Optional[str] = None
    connected_at: str


class AnalysisResponse(BaseModel):
    """Resposta da análise"""
    success: bool
    message: str
    source_type: str  # "file" ou "database"
    source_info: Dict[str, Any]
    data_summary: Dict[str, Any]
    gemini_response: str
    processing_time: float
    model_used: Optional[str] = None
    analyzed_at: Optional[str] = None


class SpecificInsightRequest(BaseModel):
    """Modelo para requisição de insights específicos"""
    database_url: str
    insight_request: str  # Descrição específica do insight desejado
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v or not v.strip():
            raise ValueError('URL da base de dados é obrigatória')
        
        v = v.strip()
        
        if '://' not in v:
            raise ValueError('URL deve conter o protocolo da base de dados (ex: postgresql://, mysql://)')
        
        if not v.startswith('sqlite://'):
            if '@' not in v:
                raise ValueError('URL deve conter credenciais de autenticação')
        
        return v
    
    @validator('insight_request')
    def validate_insight_request(cls, v):
        if not v or not v.strip():
            raise ValueError('Descrição do insight é obrigatória')
        
        v = v.strip()
        
        if len(v) < 10:
            raise ValueError('Descrição do insight deve ter pelo menos 10 caracteres')
        
        if len(v) > 500:
            raise ValueError('Descrição do insight deve ter no máximo 500 caracteres')
        
        return v


class SpecificInsightResponse(BaseModel):
    """Resposta para insights específicos"""
    success: bool
    message: str
    insight_request: str
    database_info: Dict[str, Any]
    strategic_insights: str
    processing_time: float
    model_used: Optional[str] = None
    analyzed_at: Optional[str] = None
