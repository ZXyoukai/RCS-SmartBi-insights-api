"""
Database Connector - Conexão com Bases de Dados
===============================================

Módulo responsável por conectar e extrair dados de diferentes tipos de bases de dados
para análise com o Google Gemini.

Autor: SmartBI Team
Versão: 2.0.0
Data: 2025
"""

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
from urllib.parse import urlparse
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Classe responsável por conectar e extrair dados de bases de dados
    """
    
    def __init__(self):
        """
        Inicializa o conector de base de dados
        """
        self.engine = None
        self.connection = None
        self.db_type = None
        
        # Tipos de base de dados suportados
        self.supported_databases = {
            'mysql': 'MySQL',
            'postgresql': 'PostgreSQL', 
            'sqlite': 'SQLite',
            'mssql': 'SQL Server',
            'oracle': 'Oracle'
        }
    
    def validate_database_url(self, db_url: str) -> Tuple[bool, str]:
        """
        Valida se a URL da base de dados está no formato correto
        
        Args:
            db_url: URL da base de dados (ex: postgresql://user:password@host:port/database)
            
        Returns:
            Tuple com (é_válida, mensagem)
        """
        try:
            parsed = urlparse(db_url)
            
            if not parsed.scheme:
                return False, "URL deve conter o protocolo da base de dados (ex: postgresql://, mysql://)"
            
            # Normaliza o scheme para compatibilidade
            scheme = parsed.scheme.lower()
            if scheme == 'postgres':
                scheme = 'postgresql'  # Normaliza postgres para postgresql
            
            if scheme not in self.supported_databases:
                supported = ', '.join(self.supported_databases.keys())
                return False, f"Tipo de base de dados '{parsed.scheme}' não suportado. Suportados: {supported}"
            
            if not parsed.hostname:
                return False, "URL deve conter o hostname/servidor (ex: localhost, ep-example.neon.tech)"
            
            # Validações específicas para diferentes tipos de base
            if scheme == 'sqlite':
                # SQLite pode usar arquivo local
                if not parsed.path or parsed.path == '/':
                    return False, "Para SQLite, especifique o caminho do arquivo (ex: sqlite:///path/to/database.db)"
            else:
                # Outras bases precisam de hostname
                if not parsed.username:
                    return False, "URL deve conter nome de usuário para autenticação"
                if not parsed.password:
                    return False, "URL deve conter senha para autenticação"
                if not parsed.path or parsed.path == '/':
                    return False, "URL deve especificar o nome da base de dados"
            
            return True, f"URL válida para {self.supported_databases[scheme]}"
            
        except Exception as e:
            return False, f"Erro ao validar URL: {str(e)}"
    
    def connect_to_database(self, db_url: str) -> Dict[str, Any]:
        """
        Conecta à base de dados usando a URL fornecida
        
        Args:
            db_url: URL da base de dados
            
        Returns:
            Dict com informações da conexão
        """
        try:
            # Valida a URL
            is_valid, message = self.validate_database_url(db_url)
            if not is_valid:
                raise ValueError(message)
            
            # Extrai o tipo de base de dados
            parsed = urlparse(db_url)
            scheme = parsed.scheme.lower()
            
            # Normaliza o scheme
            if scheme == 'postgres':
                scheme = 'postgresql'
                
            self.db_type = scheme
            
            logger.info(f"🔌 Conectando à base de dados {self.supported_databases[self.db_type]}...")
            logger.info(f"🌐 Host: {parsed.hostname}:{parsed.port or 'default'}")
            logger.info(f"📊 Database: {parsed.path.lstrip('/') if parsed.path else 'N/A'}")
            
            # Configurações específicas por tipo de base
            connect_args = {}
            if self.db_type == 'mysql':
                connect_args = {
                    'connect_timeout': 30,
                    'charset': 'utf8mb4',
                    'use_unicode': True,
                    'ssl_disabled': False,  # Permite SSL
                    'autocommit': True
                }
                # Se a URL contém SSL, configura adequadamente
                if 'ssl-mode=REQUIRED' in db_url.lower() or 'sslmode=require' in db_url.lower():
                    connect_args['ssl_disabled'] = False
                    connect_args['ssl_verify_cert'] = False
                    connect_args['ssl_verify_identity'] = False
            elif self.db_type == 'postgresql':
                connect_args = {
                    'connect_timeout': 30,
                    'sslmode': 'prefer'
                }
            elif self.db_type == 'sqlite':
                connect_args = {'check_same_thread': False}
            
            # Processa URL para MySQL se necessário
            processed_url = db_url
            if self.db_type == 'mysql':
                # Converte parâmetros SSL para formato correto do PyMySQL
                if 'ssl-mode=REQUIRED' in processed_url:
                    processed_url = processed_url.replace('ssl-mode=REQUIRED', 'ssl={"ssl_disabled": false}')
                
                # Adiciona charset se não existir
                if 'charset=' not in processed_url.lower():
                    separator = '&' if '?' in processed_url else '?'
                    processed_url += f'{separator}charset=utf8mb4'
            
            # Cria a engine de conexão
            self.engine = create_engine(
                processed_url,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_timeout=20,
                pool_size=5,
                max_overflow=10,
                connect_args=connect_args,
                echo=False,  # Set to True for SQL debugging
                execution_options={
                    "isolation_level": "READ_UNCOMMITTED" if self.db_type == 'mysql' else "READ_COMMITTED"
                }
            )
            
            # Testa a conexão com query específica por tipo de base
            test_query = {
                'mysql': "SELECT 1 as test",
                'postgresql': "SELECT 1 as test", 
                'sqlite': "SELECT 1 as test",
                'mssql': "SELECT 1 as test",
                'oracle': "SELECT 1 FROM DUAL"
            }.get(self.db_type, "SELECT 1 as test")
            
            with self.engine.connect() as conn:
                result = conn.execute(text(test_query))
                test_result = result.fetchone()
                logger.info(f"✅ Teste de conexão realizado: {test_result}")
            
            logger.info(f"✅ Conexão estabelecida com {self.supported_databases[self.db_type]}")
            
            return {
                "status": "connected",
                "database_type": self.supported_databases[self.db_type],
                "host": parsed.hostname,
                "port": parsed.port,
                "database": parsed.path.lstrip('/') if parsed.path else None,
                "connected_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar à base de dados: {e}")
            raise Exception(f"Erro de conexão: {str(e)}")
    
    def get_database_schema(self) -> Dict[str, Any]:
        """
        Obtém o esquema da base de dados (tabelas, colunas, etc.)
        
        Returns:
            Dict com informações do esquema
        """
        try:
            if not self.engine:
                raise Exception("Nenhuma conexão ativa. Conecte primeiro à base de dados.")
            
            logger.info("📋 Extraindo esquema da base de dados...")
            
            inspector = inspect(self.engine)
            
            # Obtém lista de tabelas
            tables = inspector.get_table_names()
            
            schema_info = {
                "total_tables": len(tables),
                "tables": {},
                "extracted_at": datetime.now().isoformat()
            }
            
            # Para cada tabela, obtém informações das colunas
            processed_tables = 0
            failed_tables = []
            
            for table_name in tables:  # Remove limitação - processa todas as tabelas
                try:
                    columns = inspector.get_columns(table_name)
                    
                    # Conta registros na tabela com timeout
                    with self.engine.connect() as conn:
                        # Usa LIMIT 0 para tabelas muito grandes, apenas para verificar se é acessível
                        try:
                            result = conn.execute(text(f"SELECT COUNT(*) FROM \"{table_name}\"")).fetchone()
                            row_count = result[0] if result else 0
                        except Exception:
                            # Se COUNT(*) falhar, tenta estimar com LIMIT
                            try:
                                conn.execute(text(f"SELECT * FROM \"{table_name}\" LIMIT 1"))
                                row_count = "Unknown (table accessible)"
                            except Exception:
                                row_count = "Error accessing table"
                    
                    schema_info["tables"][table_name] = {
                        "columns": [
                            {
                                "name": col["name"],
                                "type": str(col["type"]),
                                "nullable": col.get("nullable", True),
                                "primary_key": col.get("primary_key", False)
                            }
                            for col in columns
                        ],
                        "column_count": len(columns),
                        "row_count": row_count
                    }
                    
                    processed_tables += 1
                    logger.info(f"✅ Tabela processada: {table_name} ({len(columns)} colunas)")
                    
                except Exception as e:
                    failed_tables.append(f"{table_name}: {str(e)}")
                    logger.warning(f"⚠️ Erro ao processar tabela {table_name}: {e}")
                    continue
            
            # Adiciona informações sobre erros
            schema_info["processed_tables"] = processed_tables
            schema_info["failed_tables"] = failed_tables
            schema_info["success_rate"] = f"{processed_tables}/{len(tables)}"
            
            logger.info(f"✅ Esquema extraído: {len(schema_info['tables'])} tabelas processadas")
            return schema_info
            
        except Exception as e:
            logger.error(f"❌ Erro ao extrair esquema: {e}")
            raise Exception(f"Erro ao obter esquema: {str(e)}")
    
    def extract_sample_data(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Extrai dados de amostra da base de dados para análise
        
        Args:
            limit: Número máximo de registros por tabela
            
        Returns:
            Dict com dados de amostra
        """
        try:
            if not self.engine:
                raise Exception("Nenhuma conexão ativa. Conecte primeiro à base de dados.")
            
            logger.info(f"📊 Extraindo dados de amostra (limite: {limit} registros por tabela)...")
            
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            sample_data = {
                "tables_processed": 0,
                "total_records": 0,
                "data": {},
                "summary": {},
                "extracted_at": datetime.now().isoformat()
            }
            
            # Extrai dados de todas as tabelas (ou até um limite razoável)
            failed_extractions = []
            max_tables_to_process = min(len(tables), 50)  # Processa até 50 tabelas
            
            for i, table_name in enumerate(tables[:max_tables_to_process]):
                try:
                    logger.info(f"📊 Processando tabela {i+1}/{max_tables_to_process}: {table_name}")
                    
                    # Query para extrair dados de amostra com aspas para nomes problemáticos
                    query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
                    
                    with self.engine.connect() as conn:
                        df = pd.read_sql(query, conn)
                    
                    if len(df) > 0:
                        sample_data["tables_processed"] += 1
                        sample_data["total_records"] += len(df)
                        
                        # Converte dados para formato JSON serializável
                        # Limita a amostra para evitar payloads muito grandes
                        sample_size = min(len(df), 30)  # Máximo 30 registros por tabela
                        sample_data["data"][table_name] = df.head(sample_size).to_dict('records')
                        
                        # Estatísticas da tabela
                        sample_data["summary"][table_name] = {
                            "total_rows": len(df),
                            "columns": list(df.columns),
                            "column_count": len(df.columns),
                            "data_types": df.dtypes.astype(str).to_dict(),
                            "null_counts": df.isnull().sum().to_dict(),
                            "sample_values": {
                                col: df[col].dropna().head(3).tolist() 
                                for col in df.columns if not df[col].dropna().empty
                            }
                        }
                        
                        logger.info(f"✅ Tabela '{table_name}': {len(df)} registros extraídos")
                    else:
                        logger.info(f"⚠️ Tabela '{table_name}' está vazia")
                    
                except Exception as e:
                    failed_extractions.append(f"{table_name}: {str(e)}")
                    logger.warning(f"❌ Erro ao extrair dados da tabela '{table_name}': {e}")
                    continue
            
            # Adiciona informações sobre falhas
            sample_data["failed_extractions"] = failed_extractions
            sample_data["total_tables_available"] = len(tables)
            sample_data["success_rate"] = f"{sample_data['tables_processed']}/{len(tables)}"
            
            logger.info(f"✅ Extração concluída: {sample_data['tables_processed']} tabelas, {sample_data['total_records']} registros")
            return sample_data
            
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados: {e}")
            raise Exception(f"Erro na extração de dados: {str(e)}")
    
    def execute_custom_query(self, query: str, limit: int = 1000) -> Dict[str, Any]:
        """
        Executa uma consulta SQL personalizada
        
        Args:
            query: Consulta SQL
            limit: Limite de registros
            
        Returns:
            Dict com resultados da consulta
        """
        try:
            if not self.engine:
                raise Exception("Nenhuma conexão ativa. Conecte primeiro à base de dados.")
            
            # Adiciona LIMIT se não estiver presente e for uma query SELECT
            if query.strip().upper().startswith('SELECT') and 'LIMIT' not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit}"
            
            logger.info(f"🔍 Executando consulta personalizada...")
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            result = {
                "query": query,
                "rows_returned": len(df),
                "columns": list(df.columns),
                "data": df.to_dict('records'),
                "data_types": df.dtypes.astype(str).to_dict(),
                "executed_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Consulta executada: {len(df)} registros retornados")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao executar consulta: {e}")
            raise Exception(f"Erro na consulta: {str(e)}")
    
    def prepare_data_for_analysis(self, sample_data: Dict[str, Any]) -> str:
        """
        Prepara os dados para análise com Gemini
        
        Args:
            sample_data: Dados extraídos da base de dados
            
        Returns:
            String formatada para análise
        """
        try:
            content = []
            
            content.append("=== ANÁLISE DE BASE DE DADOS ===")
            content.append(f"Data de Extração: {sample_data.get('extracted_at')}")
            content.append(f"Tabelas Disponíveis: {sample_data.get('total_tables_available', 'N/A')}")
            content.append(f"Tabelas Processadas: {sample_data.get('tables_processed', 0)}")
            content.append(f"Taxa de Sucesso: {sample_data.get('success_rate', 'N/A')}")
            content.append(f"Total de Registros: {sample_data.get('total_records', 0)}")
            
            # Mostra falhas se houver
            if sample_data.get('failed_extractions'):
                content.append(f"Tabelas com Erro: {len(sample_data['failed_extractions'])}")
                content.append("Erros:")
                for error in sample_data['failed_extractions'][:5]:  # Mostra até 5 erros
                    content.append(f"   - {error}")
            
            content.append("")
            
            # Resumo das tabelas
            content.append("=== RESUMO DAS TABELAS ===")
            for table_name, summary in sample_data.get('summary', {}).items():
                content.append(f"\n📊 TABELA: {table_name}")
                content.append(f"   - Registros: {summary.get('total_rows', 0)}")
                content.append(f"   - Colunas: {summary.get('column_count', 0)}")
                content.append(f"   - Colunas: {', '.join(summary.get('columns', []))}")
                
                # Tipos de dados
                if summary.get('data_types'):
                    content.append("   - Tipos de Dados:")
                    for col, dtype in list(summary['data_types'].items())[:5]:  # Primeiras 5 colunas
                        content.append(f"     * {col}: {dtype}")
                
                # Valores de amostra
                if summary.get('sample_values'):
                    content.append("   - Valores de Amostra:")
                    for col, values in list(summary['sample_values'].items())[:3]:  # Primeiras 3 colunas
                        content.append(f"     * {col}: {values}")
            
            # Dados de amostra (limitado)
            content.append("\n=== DADOS DE AMOSTRA ===")
            for table_name, data in sample_data.get('data', {}).items():
                if data:
                    content.append(f"\n📋 AMOSTRA DA TABELA: {table_name}")
                    # Mostra apenas os primeiros 3 registros
                    for i, record in enumerate(data[:3]):
                        content.append(f"   Registro {i+1}: {record}")
                    
                    if len(data) > 3:
                        content.append(f"   ... e mais {len(data) - 3} registros")
            
            return "\n".join(content)
            
        except Exception as e:
            logger.error(f"❌ Erro ao preparar dados: {e}")
            return f"Erro ao preparar dados para análise: {str(e)}"
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre a conexão atual
        
        Returns:
            Dict com informações da conexão
        """
        return {
            "connected": self.engine is not None,
            "database_type": self.db_type,
            "supported_databases": self.supported_databases
        }
    
    def close_connection(self):
        """
        Fecha a conexão com a base de dados
        """
        try:
            if self.engine:
                self.engine.dispose()
                self.engine = None
                self.db_type = None
                logger.info("🔌 Conexão com base de dados fechada")
        except Exception as e:
            logger.warning(f"Aviso ao fechar conexão: {e}")
