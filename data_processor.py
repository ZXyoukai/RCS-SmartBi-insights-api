"""
DataProcessor - Módulo de processamento de dados
==============================================

Este módulo é responsável por processar arquivos CSV e SQL,
extrair informações relevantes e preparar os dados para análise.

Autor: SmartBI Team
Versão: 2.0.0
"""

import pandas as pd
import sqlparse
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Classe responsável pelo processamento de arquivos CSV e SQL
    """
    
    def __init__(self):
        """
        Inicializa o processador de dados
        """
        self.supported_extensions = ['.csv', '.sql']
        self.max_preview_rows = 100  # Máximo de linhas para preview
        self.max_content_length = 20000  # Máximo de caracteres para análise
    
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Processa um arquivo e retorna informações estruturadas
        
        Args:
            file_path: Caminho para o arquivo
        
        Returns:
            Dict: Dados processados e informações do arquivo
        """
        
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Formato não suportado: {file_extension}")
        
        logger.info(f"📊 Processando arquivo {file_extension}: {file_path.name}")
        
        try:
            if file_extension == '.csv':
                return self._process_csv(file_path)
            elif file_extension == '.sql':
                return self._process_sql(file_path)
        except Exception as e:
            logger.error(f"Erro ao processar arquivo: {e}")
            raise
    
    
    def _process_csv(self, file_path: Path) -> Dict[str, Any]:
        """
        Processa arquivo CSV
        
        Args:
            file_path: Caminho para o arquivo CSV
        
        Returns:
            Dict: Dados e informações do CSV
        """
        
        try:
            # Tenta diferentes separadores e encodings
            separators = [',', ';', '\t']
            encodings = ['utf-8', 'latin1', 'cp1252']
            
            df = None
            used_separator = ','
            used_encoding = 'utf-8'
            
            # Tenta diferentes combinações
            for encoding in encodings:
                for separator in separators:
                    try:
                        df_temp = pd.read_csv(file_path, sep=separator, encoding=encoding)
                        
                        # Verifica se conseguiu ler corretamente (mais de 1 coluna)
                        if df_temp.shape[1] > 1:
                            df = df_temp
                            used_separator = separator
                            used_encoding = encoding
                            break
                    except:
                        continue
                
                if df is not None:
                    break
            
            if df is None:
                raise ValueError("Não foi possível ler o arquivo CSV com os separadores e encodings testados")
            
            logger.info(f"✅ CSV lido com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")
            
            # Informações básicas
            info = {
                'type': 'csv',
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'column_names': list(df.columns),
                'separator': used_separator,
                'encoding': used_encoding,
                'file_size': int(file_path.stat().st_size),
                'processed_at': datetime.now().isoformat()
            }
            
            # Informações detalhadas das colunas
            column_info = {}
            for col in df.columns:
                col_data = df[col]
                column_info[col] = {
                    'dtype': str(col_data.dtype),
                    'non_null_count': int(col_data.count()),
                    'null_count': int(col_data.isnull().sum()),
                    'unique_values': int(col_data.nunique())
                }
                
                # Adiciona informações específicas por tipo
                if col_data.dtype in ['int64', 'float64']:
                    column_info[col].update({
                        'min': float(col_data.min()) if pd.notna(col_data.min()) else None,
                        'max': float(col_data.max()) if pd.notna(col_data.max()) else None,
                        'mean': float(col_data.mean()) if pd.notna(col_data.mean()) else None
                    })
                elif col_data.dtype == 'object':
                    # Pega algumas amostras de valores únicos
                    unique_samples = col_data.unique()[:5].tolist()
                    column_info[col]['sample_values'] = unique_samples
            
            # Converte todos os tipos numpy para tipos Python nativos
            info['column_details'] = self._convert_numpy_types(column_info)
            
            # Prepara conteúdo para análise (amostra dos dados)
            preview_df = df.head(self.max_preview_rows)
            
            # Converte para formato de texto estruturado
            content_parts = []
            content_parts.append("ESTRUTURA DO DATASET CSV:")
            content_parts.append(f"Linhas: {info['rows']}, Colunas: {info['columns']}")
            content_parts.append(f"Colunas: {', '.join(info['column_names'])}")
            content_parts.append("")
            
            # Informações das colunas
            content_parts.append("INFORMAÇÕES DAS COLUNAS:")
            for col, details in column_info.items():
                content_parts.append(f"- {col}: {details['dtype']}, {details['non_null_count']} valores válidos")
                if 'sample_values' in details:
                    content_parts.append(f"  Exemplos: {', '.join(map(str, details['sample_values']))}")
            content_parts.append("")
            
            # Amostra dos dados
            content_parts.append("AMOSTRA DOS DADOS:")
            content_parts.append(preview_df.to_string(index=False, max_rows=20))
            
            content = '\n'.join(content_parts)
            
            # Limita tamanho do conteúdo
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "\n\n[DADOS TRUNCADOS]"
            
            return {
                'info': info,
                'content': content
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar CSV: {e}")
            raise ValueError(f"Erro ao processar arquivo CSV: {str(e)}")
    
    
    def _process_sql(self, file_path: Path) -> Dict[str, Any]:
        """
        Processa arquivo SQL
        
        Args:
            file_path: Caminho para o arquivo SQL
        
        Returns:
            Dict: Dados e informações do SQL
        """
        
        try:
            # Lê o arquivo SQL
            encodings = ['utf-8', 'latin1', 'cp1252']
            content = None
            used_encoding = 'utf-8'
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        used_encoding = encoding
                        break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError("Não foi possível ler o arquivo SQL com os encodings testados")
            
            logger.info(f"✅ SQL lido com sucesso: {len(content)} caracteres")
            
            # Parse do SQL
            parsed_statements = sqlparse.parse(content)
            
            # Analisa as instruções
            tables_info = {}
            insert_counts = {}
            statement_types = {}
            
            for stmt in parsed_statements:
                stmt_type = stmt.get_type()
                if stmt_type not in statement_types:
                    statement_types[stmt_type] = 0
                statement_types[stmt_type] += 1
                
                # Extrai informações de tabelas
                if stmt_type == 'CREATE':
                    table_name = self._extract_table_name_from_create(str(stmt))
                    if table_name:
                        tables_info[table_name] = {
                            'type': 'table',
                            'columns': self._extract_columns_from_create(str(stmt))
                        }
                
                elif stmt_type == 'INSERT':
                    table_name = self._extract_table_name_from_insert(str(stmt))
                    if table_name:
                        if table_name not in insert_counts:
                            insert_counts[table_name] = 0
                        # Conta quantos VALUES existem (aproximação)
                        values_count = str(stmt).upper().count('VALUES')
                        insert_counts[table_name] += values_count
            
            # Informações básicas
            info = {
                'type': 'sql',
                'file_size': file_path.stat().st_size,
                'encoding': used_encoding,
                'statements': len(parsed_statements),
                'statement_types': statement_types,
                'tables_found': list(tables_info.keys()),
                'tables_count': len(tables_info),
                'insert_counts': insert_counts,
                'total_inserts': sum(insert_counts.values()) if insert_counts else 0,
                'processed_at': datetime.now().isoformat()
            }
            
            # Se há tabelas, adiciona detalhes
            if tables_info:
                info['table_details'] = tables_info
            
            # Prepara conteúdo para análise
            content_parts = []
            content_parts.append("ESTRUTURA DO BANCO DE DADOS SQL:")
            content_parts.append(f"Statements: {info['statements']}")
            content_parts.append(f"Tipos: {', '.join([f'{k}({v})' for k, v in statement_types.items()])}")
            content_parts.append("")
            
            if tables_info:
                content_parts.append("TABELAS ENCONTRADAS:")
                for table_name, table_info in tables_info.items():
                    content_parts.append(f"- {table_name}")
                    if table_info['columns']:
                        content_parts.append(f"  Colunas: {', '.join(table_info['columns'])}")
                    if table_name in insert_counts:
                        content_parts.append(f"  Registros inseridos: {insert_counts[table_name]}")
                content_parts.append("")
            
            # Adiciona parte do conteúdo original (limitado)
            content_parts.append("CONTEÚDO SQL (AMOSTRA):")
            sql_preview = content[:5000] if len(content) > 5000 else content
            content_parts.append(sql_preview)
            
            processed_content = '\n'.join(content_parts)
            
            # Limita tamanho do conteúdo
            if len(processed_content) > self.max_content_length:
                processed_content = processed_content[:self.max_content_length] + "\n\n[CONTEÚDO TRUNCADO]"
            
            return {
                'info': info,
                'content': processed_content
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar SQL: {e}")
            raise ValueError(f"Erro ao processar arquivo SQL: {str(e)}")
    
    
    def _extract_table_name_from_create(self, sql_statement: str) -> Optional[str]:
        """
        Extrai nome da tabela de uma instrução CREATE TABLE
        """
        try:
            # Simplificado - procura padrão CREATE TABLE
            sql_upper = sql_statement.upper()
            if 'CREATE TABLE' in sql_upper:
                # Encontra a posição após CREATE TABLE
                start = sql_upper.find('CREATE TABLE') + len('CREATE TABLE')
                remaining = sql_statement[start:].strip()
                
                # Pega a primeira palavra (nome da tabela)
                words = remaining.split()
                if words:
                    table_name = words[0].strip('(')
                    # Remove caracteres especiais comuns
                    table_name = table_name.replace('`', '').replace('"', '').replace('[', '').replace(']', '')
                    return table_name
        except:
            pass
        return None
    
    
    def _extract_table_name_from_insert(self, sql_statement: str) -> Optional[str]:
        """
        Extrai nome da tabela de uma instrução INSERT
        """
        try:
            sql_upper = sql_statement.upper()
            if 'INSERT INTO' in sql_upper:
                start = sql_upper.find('INSERT INTO') + len('INSERT INTO')
                remaining = sql_statement[start:].strip()
                
                # Pega a primeira palavra (nome da tabela)
                words = remaining.split()
                if words:
                    table_name = words[0].strip('(')
                    table_name = table_name.replace('`', '').replace('"', '').replace('[', '').replace(']', '')
                    return table_name
        except:
            pass
        return None
    
    
    def _extract_columns_from_create(self, sql_statement: str) -> List[str]:
        """
        Extrai nomes das colunas de uma instrução CREATE TABLE
        """
        try:
            # Procura o conteúdo entre parênteses
            start = sql_statement.find('(')
            end = sql_statement.rfind(')')
            
            if start != -1 and end != -1:
                columns_part = sql_statement[start+1:end]
                
                # Split por vírgulas e extrai nomes das colunas
                columns = []
                for line in columns_part.split(','):
                    line = line.strip()
                    if line and not line.upper().startswith(('PRIMARY', 'FOREIGN', 'KEY', 'CONSTRAINT', 'INDEX')):
                        # Pega a primeira palavra (nome da coluna)
                        words = line.split()
                        if words:
                            col_name = words[0].strip('`"[]')
                            columns.append(col_name)
                
                return columns
        except:
            pass
        return []
    
    
    def get_supported_formats(self) -> List[str]:
        """
        Retorna lista de formatos suportados
        
        Returns:
            List[str]: Extensões suportadas
        """
        return self.supported_extensions

    def _convert_numpy_types(self, obj):
        """
        Converte tipos numpy/pandas para tipos Python nativos para serialização JSON
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
