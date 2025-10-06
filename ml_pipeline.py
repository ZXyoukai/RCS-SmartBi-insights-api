import pandas as pd
import sqlite3
import os
import re
import math
import sqlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, mean_squared_error, 
                           mean_absolute_error, r2_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self):
        self.models = {}
        self.problem_type = None  # 'classification' ou 'regression'
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def detect_problem_type(self, y):
        """Detecta automaticamente se é problema de classificação ou regressão"""
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        
        # Se há apenas 1 classe, não é possível fazer classificação
        if n_unique == 1:
            print(f"⚠️ Detectada apenas 1 classe única: {unique_values[0]}")
            print(f"🔄 Convertendo para problema de regressão...")
            return 'regression'
        
        # Se tem poucas classes e são inteiros, provavelmente é classificação
        elif n_unique <= 20 and all(isinstance(val, (int, np.integer)) for val in unique_values):
            print(f"✅ Detectado problema de classificação: {n_unique} classes")
            return 'classification'
        
        # Se tem muitos valores únicos ou são contínuos, é regressão
        else:
            print(f"✅ Detectado problema de regressão: {n_unique} valores únicos")
            return 'regression'
    
    def setup_models(self, problem_type):
        """Configura modelos apropriados baseado no tipo de problema"""
        if problem_type == 'classification':
            self.models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
            }
            print("🤖 Modelos de classificação configurados")
        else:  # regression
            self.models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'SVM': SVR(),
                'LinearRegression': LinearRegression()
            }
            print("🤖 Modelos de regressão configurados")
        
        self.problem_type = problem_type
    
    def sanitize_for_json(self, obj):
        """Remove valores NaN e infinitos que não são compatíveis com JSON"""
        if isinstance(obj, dict):
            return {k: self.sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (np.ndarray,)):
            return self.sanitize_for_json(obj.tolist())
        elif isinstance(obj, (float, np.float64, np.float32)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (int, np.int64, np.int32)):
            return int(obj)
        else:
            return obj
        
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """Carrega dados de um arquivo CSV"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dados CSV carregados: {df.shape[0]} linhas, {df.shape[1]} colunas.")
            return df
        except Exception as e:
            raise Exception(f"Erro ao carregar CSV: {e}")
    
    def clean_sql_script(self, sql_script: str) -> str:
        """Limpa e processa script SQL para compatibilidade universal"""
        try:
            print("🧹 Iniciando limpeza com sqlparse...")
            # Usa sqlparse para processar o SQL de forma robusta
            parsed = sqlparse.parse(sql_script)
            cleaned_statements = []
            
            for i, statement in enumerate(parsed):
                # Remove comentários e normaliza
                cleaned = sqlparse.format(
                    str(statement), 
                    strip_comments=True, 
                    strip_whitespace=True
                )
                if cleaned.strip():
                    # Aplica conversões específicas após o sqlparse
                    cleaned = self.convert_to_sqlite_syntax(cleaned)
                    cleaned_statements.append(cleaned)
                    print(f"   Statement {i+1} processado: {cleaned[:100]}...")
            
            result = ';\n'.join(cleaned_statements) + ';'
            print(f"✅ sqlparse concluído: {len(cleaned_statements)} statements")
            return result
            
        except Exception as e:
            print(f"⚠️ sqlparse falhou ({e}), usando limpeza manual...")
            # Fallback: limpeza manual se sqlparse falhar
            return self.manual_sql_cleanup(sql_script)
    
    def manual_sql_cleanup(self, sql_script: str) -> str:
        """Limpeza manual de SQL como fallback"""
        print("🔧 Iniciando limpeza manual...")
        
        lines = sql_script.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            original_line = line
            line_stripped = line.strip()
            
            # Pula linhas de comentário e comandos específicos do phpMyAdmin
            if (line_stripped.startswith('--') or 
                line_stripped.startswith('/*') or
                line_stripped.startswith('SET ') or
                line_stripped.startswith('START TRANSACTION') or
                line_stripped.startswith('COMMIT') or
                line_stripped.startswith('/*!') or
                line_stripped.startswith('#') or
                not line_stripped):
                continue
            
            # Remove comentários no final da linha
            line_cleaned = re.sub(r'--.*$', '', line)  # Comentários --
            line_cleaned = re.sub(r'#.*$', '', line_cleaned)   # Comentários #
            
            # Converte sintaxes específicas para SQLite
            line_cleaned = self.convert_to_sqlite_syntax(line_cleaned)
            
            if line_cleaned.strip():
                cleaned_lines.append(line_cleaned)
                if i < 10:  # Debug das primeiras linhas
                    print(f"   Linha {i+1}: '{original_line.strip()[:50]}...' → '{line_cleaned.strip()[:50]}...'")
        
        result = '\n'.join(cleaned_lines)
        print(f"✅ Limpeza manual concluída: {len(cleaned_lines)} linhas processadas")
        return result
    
    def convert_to_sqlite_syntax(self, line: str) -> str:
        """Converte sintaxes específicas de outros SGBDs para SQLite"""
        # Remove comandos phpMyAdmin específicos
        line = re.sub(r'SET\s+SQL_MODE\s*=.*?;', '', line, flags=re.IGNORECASE)
        line = re.sub(r'START\s+TRANSACTION\s*;', '', line, flags=re.IGNORECASE)
        line = re.sub(r'SET\s+time_zone\s*=.*?;', '', line, flags=re.IGNORECASE)
        line = re.sub(r'SET\s+@OLD_.*?;', '', line, flags=re.IGNORECASE)
        line = re.sub(r'SET\s+NAMES\s+.*?;', '', line, flags=re.IGNORECASE)
        line = re.sub(r'COMMIT\s*;', '', line, flags=re.IGNORECASE)
        
        # Remove comandos phpMyAdmin com /*!
        line = re.sub(r'/\*!\d+.*?\*/', '', line, flags=re.IGNORECASE)
        
        # MySQL/MariaDB: Remove AUTO_INCREMENT, ENGINE, CHARSET, etc.
        line = re.sub(r'\s+AUTO_INCREMENT\s*=?\s*\d*', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+AUTO_INCREMENT', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+ENGINE\s*=\s*\w+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+DEFAULT\s+CHARSET\s*=\s*\w+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+COLLATE\s*=?\s*\w+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+CHARACTER\s+SET\s+\w+', '', line, flags=re.IGNORECASE)
        
        # Remove KEY e INDEX definitions que podem causar problemas
        line = re.sub(r'\s+KEY\s+`[^`]+`\s*\([^)]+\)', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+UNIQUE\s+KEY\s+`[^`]+`\s*\([^)]+\)', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+INDEX\s+`[^`]+`\s*\([^)]+\)', '', line, flags=re.IGNORECASE)
        
        # PostgreSQL: Converte tipos específicos
        line = re.sub(r'\bSERIAL\b', 'INTEGER', line, flags=re.IGNORECASE)
        line = re.sub(r'\bBIGSERIAL\b', 'INTEGER', line, flags=re.IGNORECASE)
        line = re.sub(r'\bTEXT\[\]', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bINTEGER\[\]', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bBOOLEAN\b', 'INTEGER', line, flags=re.IGNORECASE)
        
        # MySQL: Converte tipos específicos
        line = re.sub(r'\bTINYINT\b', 'INTEGER', line, flags=re.IGNORECASE)
        line = re.sub(r'\bSMALLINT\b', 'INTEGER', line, flags=re.IGNORECASE)
        line = re.sub(r'\bMEDIUMINT\b', 'INTEGER', line, flags=re.IGNORECASE)
        line = re.sub(r'\bBIGINT\([^)]+\)', 'INTEGER', line, flags=re.IGNORECASE)
        line = re.sub(r'\bBIGINT\b', 'INTEGER', line, flags=re.IGNORECASE)
        line = re.sub(r'\bINT\([^)]+\)', 'INTEGER', line, flags=re.IGNORECASE)
        line = re.sub(r'\bDOUBLE\([^)]+\)', 'REAL', line, flags=re.IGNORECASE)
        line = re.sub(r'\bDOUBLE\b', 'REAL', line, flags=re.IGNORECASE)
        line = re.sub(r'\bFLOAT\b', 'REAL', line, flags=re.IGNORECASE)
        line = re.sub(r'\bDECIMAL\([^)]+\)', 'REAL', line, flags=re.IGNORECASE)
        line = re.sub(r'\bVARCHAR\([^)]+\)', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bCHAR\([^)]+\)', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bTEXT\([^)]+\)', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bLONGTEXT\b', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bMEDIUMTEXT\b', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bTINYTEXT\b', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bDATETIME\b', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bTIMESTAMP\b', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bDATE\b', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bTIME\b', 'TEXT', line, flags=re.IGNORECASE)
        line = re.sub(r'\bENUM\([^)]+\)', 'TEXT', line, flags=re.IGNORECASE)
        
        # Remove constraints não suportados
        line = re.sub(r'\s+UNSIGNED', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+ZEROFILL', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+ON\s+UPDATE\s+CURRENT_TIMESTAMP', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+DEFAULT\s+CURRENT_TIMESTAMP', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+DEFAULT\s+current_timestamp\(\)', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s+NOT\s+NULL\s+DEFAULT\s+NULL', ' DEFAULT NULL', line, flags=re.IGNORECASE)
        
        return line
    
    def detect_sql_dialect(self, sql_script: str) -> str:
        """Detecta o dialeto SQL baseado em palavras-chave específicas"""
        sql_lower = sql_script.lower()
        
        if any(keyword in sql_lower for keyword in ['auto_increment', 'engine=', 'charset=']):
            return 'mysql'
        elif any(keyword in sql_lower for keyword in ['serial', 'bigserial', 'boolean', 'array']):
            return 'postgresql'
        elif any(keyword in sql_lower for keyword in ['identity', 'nvarchar', 'datetime2']):
            return 'sqlserver'
        elif any(keyword in sql_lower for keyword in ['autoincrement', 'pragma']):
            return 'sqlite'
        else:
            return 'generic'
    
    def load_data_from_sql(self, file_path: str) -> pd.DataFrame:
        """Carrega dados de um dump SQL universal (MySQL, PostgreSQL, MariaDB, etc.)"""
        dialect = 'desconhecido'
        try:
            # Lê o arquivo SQL
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            
            print(f"📄 Arquivo SQL lido: {len(sql_script)} caracteres")
            
            # Detecta o dialeto SQL
            dialect = self.detect_sql_dialect(sql_script)
            print(f"🔍 Dialeto SQL detectado: {dialect.upper()}")
            
            # Limpa e converte o script SQL
            cleaned_sql = self.clean_sql_script(sql_script)
            print(f"🧹 SQL limpo: {len(cleaned_sql)} caracteres")
            
            # Debug: mostra parte do SQL limpo
            print("📝 Primeiras linhas do SQL processado:")
            for i, line in enumerate(cleaned_sql.split('\n')[:10]):
                if line.strip():
                    print(f"   {i+1}: {line}")
            
            # Conecta ao banco SQLite temporário
            conn = sqlite3.connect(':memory:')
            
            # Processa por blocos (CREATE + INSERT separados)
            create_statements = []
            insert_statements = []
            ignored_statements = []
            
            statements = [stmt.strip() for stmt in cleaned_sql.split(';') if stmt.strip()]
            print(f"📊 Total de statements encontrados: {len(statements)}")
            
            for stmt in statements:
                stmt_upper = stmt.upper().strip()
                if stmt_upper.startswith('CREATE TABLE'):
                    create_statements.append(stmt)
                elif stmt_upper.startswith('INSERT'):
                    insert_statements.append(stmt)
                elif any(stmt_upper.startswith(ignore) for ignore in [
                    'ALTER TABLE', 'ADD CONSTRAINT', 'ADD FOREIGN KEY', 
                    'CREATE INDEX', 'DROP', 'SET', 'COMMIT', 'START TRANSACTION',
                    'LOCK', 'UNLOCK'
                ]):
                    ignored_statements.append(stmt[:100] + '...' if len(stmt) > 100 else stmt)
                else:
                    # Só processa outros statements seguros
                    if not any(keyword in stmt_upper for keyword in ['FOREIGN', 'CONSTRAINT', 'INDEX']):
                        create_statements.append(stmt)
            
            print(f"📋 CREATE statements: {len(create_statements)}")
            print(f"📋 INSERT statements: {len(insert_statements)}")
            print(f"📋 Ignored statements: {len(ignored_statements)}")
            if ignored_statements:
                print(f"   Exemplos ignorados: {ignored_statements[:3]}")
            
            # Executa CREATE statements primeiro
            for i, statement in enumerate(create_statements):
                try:
                    print(f"🔨 Executando CREATE {i+1}: {statement[:100]}...")
                    conn.execute(statement)
                    print(f"✅ CREATE {i+1} executado com sucesso")
                except sqlite3.Error as e:
                    print(f"❌ Erro no CREATE {i+1}: {e}")
                    print(f"   Statement: {statement}")
                    continue
            
            conn.commit()
            
            # Verifica se tabelas foram criadas
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables_after_create = cursor.fetchall()
            print(f"🏗️ Tabelas criadas: {[t[0] for t in tables_after_create]}")
            
            if not tables_after_create:
                print("❌ Nenhuma tabela foi criada. Tentando método alternativo...")
                # Método alternativo: executar tudo junto
                try:
                    conn.executescript(cleaned_sql)
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables_after_create = cursor.fetchall()
                    print(f"🔄 Tabelas após executescript: {[t[0] for t in tables_after_create]}")
                except Exception as alt_e:
                    print(f"❌ Método alternativo também falhou: {alt_e}")
            
            # Executa INSERT statements
            successful_inserts = 0
            for i, statement in enumerate(insert_statements):
                try:
                    conn.execute(statement)
                    successful_inserts += 1
                except sqlite3.Error as e:
                    print(f"⚠️ Erro no INSERT {i+1}: {e}")
                    continue
            
            print(f"📥 INSERTs executados com sucesso: {successful_inserts}/{len(insert_statements)}")
            conn.commit()
            
            # Obtém as tabelas finais
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if not tables:
                # Debug final: mostra o que realmente foi executado
                print("🔍 Debug final - SQL processado:")
                print(cleaned_sql[:500])
                raise Exception("Nenhuma tabela encontrada no dump SQL após processamento")
            
            # Lista todas as tabelas encontradas
            table_names = [table[0] for table in tables]
            print(f"🗂️ Tabelas finais encontradas: {table_names}")
            
            # Escolhe a tabela com mais dados
            best_table = None
            max_rows = 0
            
            for table_name in table_names:
                try:
                    count_result = conn.execute(f"SELECT COUNT(*) FROM `{table_name}`").fetchone()
                    row_count = count_result[0] if count_result else 0
                    print(f"📊 Tabela '{table_name}': {row_count} linhas")
                    
                    if row_count > max_rows:
                        max_rows = row_count
                        best_table = table_name
                except sqlite3.Error as e:
                    print(f"⚠️ Erro ao contar linhas da tabela '{table_name}': {e}")
                    continue
            
            if not best_table:
                best_table = table_names[0]
            
            print(f"🎯 Melhor tabela selecionada: '{best_table}' com {max_rows} linhas")
            
            # Carrega os dados da melhor tabela
            df = pd.read_sql_query(f"SELECT * FROM `{best_table}`", conn)
            conn.close()
            
            print(f"✅ Dados SQL carregados da tabela '{best_table}': {df.shape[0]} linhas, {df.shape[1]} colunas.")
            print(f"🔄 Dialeto original: {dialect.upper()} → Convertido para SQLite")
            
            return df
            
        except Exception as e:
            print(f"💥 Erro completo: {str(e)}")
            raise Exception(f"Erro ao carregar SQL ({dialect}): {e}")
    
    def select_best_target_column(self, df: pd.DataFrame):
        """Seleciona a melhor coluna para ser o target baseada na qualidade dos dados"""
        best_column = None
        best_score = -1
        
        print("🎯 Avaliando colunas para target:")
        
        for col in df.columns:
            # Calcula score da coluna
            non_null_count = df[col].notna().sum()
            non_null_ratio = non_null_count / len(df)
            
            if non_null_count == 0:
                score = 0
                print(f"   ❌ {col}: 0 valores válidos (score: {score:.2f})")
                continue
            
            # Tenta converter para numérico
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                unique_values = numeric_col.dropna().nunique()
                is_numeric = True
            except:
                unique_values = df[col].dropna().nunique()
                is_numeric = False
            
            # Score baseado em: % valores válidos + variabilidade + preferência numérica
            variability_score = min(unique_values / len(df), 0.5)  # Max 0.5
            numeric_bonus = 0.3 if is_numeric else 0
            score = non_null_ratio * 0.5 + variability_score + numeric_bonus
            
            print(f"   {'✅' if is_numeric else '📝'} {col}: {non_null_count}/{len(df)} válidos ({non_null_ratio:.1%}), {unique_values} únicos (score: {score:.2f})")
            
            if score > best_score:
                best_score = score
                best_column = col
        
        if best_column is None:
            raise Exception("Nenhuma coluna adequada encontrada para target")
        
        print(f"🎯 Melhor coluna para target: '{best_column}' (score: {best_score:.2f})")
        return best_column
    
    def preprocess_data(self, df: pd.DataFrame):
        """Pré-processamento dos dados"""
        if df.shape[1] < 2:
            raise Exception("Dataset deve ter pelo menos 2 colunas (features + target)")
        
        # Seleciona a melhor coluna para target
        target_column = self.select_best_target_column(df)
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Remove linhas com target nulo/None
        mask = pd.notna(y)
        y = y[mask]
        X = X[mask]
        
        if len(y) == 0:
            raise Exception("Todos os valores do target são nulos")
        
        print(f"📊 Dataset após limpeza: {len(y)} linhas válidas")
        
        # Seleciona apenas colunas numéricas para as features
        X_numeric = X.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            raise Exception("Nenhuma coluna numérica encontrada para features")
        
        print(f"🔢 Features numéricas: {list(X_numeric.columns)}")
        
        # Tratamento de valores nulos nas features
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X_numeric), columns=X_numeric.columns)
        
        # Normalização/Standardização
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_numeric.columns)
        
        # Converte target para numérico se possível
        try:
            y_numeric = pd.to_numeric(y, errors='coerce')
            if not y_numeric.isna().all():
                y = y_numeric
        except:
            pass
        
        print(f"🎯 Target: '{target_column}' com {len(y.unique())} valores únicos")
        
        return X_scaled, y
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Treina e avalia os modelos"""
        results = {}
        
        # Detectar tipo de problema e configurar modelos
        problem_type = self.detect_problem_type(y_train)
        self.setup_models(problem_type)
        
        for name, model in self.models.items():
            try:
                print(f"🤖 Treinando {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if problem_type == 'classification':
                    # Métricas de classificação
                    y_proba = None
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)
                        if y_proba.shape[1] > 2:  # multiclass
                            y_proba = y_proba
                        else:  # binary
                            y_proba = y_proba[:, 1]
                    
                    # Usa o número total de classes do problema para determinar o método
                    # (não apenas as que aparecem no conjunto de teste)
                    n_classes_total = len(np.unique(np.concatenate([y_train, y_test])))
                    
                    if n_classes_total == 2:
                        average_method = 'binary'
                    else:
                        average_method = 'weighted'
                    
                    # Calcular ROC-AUC apenas se há mais de uma classe
                    roc_auc = None
                    if y_proba is not None and n_classes_total > 1:
                        try:
                            roc_auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                            roc_auc = float(roc_auc_val) if not math.isnan(roc_auc_val) else None
                        except Exception:
                            roc_auc = None
                    
                    results[name] = {
                        'model': model,
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'problem_type': 'classification',
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, average=average_method, zero_division=0)),
                        'recall': float(recall_score(y_test, y_pred, average=average_method, zero_division=0)),
                        'f1': float(f1_score(y_test, y_pred, average=average_method, zero_division=0)),
                        'roc_auc': roc_auc,
                        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                        'classification_report': classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                    }
                    
                else:  # regression
                    # Métricas de regressão
                    mse = float(mean_squared_error(y_test, y_pred))
                    mae = float(mean_absolute_error(y_test, y_pred))
                    r2 = float(r2_score(y_test, y_pred))
                    rmse = float(math.sqrt(mse))
                    
                    results[name] = {
                        'model': model,
                        'y_pred': y_pred,
                        'problem_type': 'regression',
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse,
                        'r2_score': r2,
                        'y_test_mean': float(np.mean(y_test)),
                        'y_test_std': float(np.std(y_test))
                    }
                
            except Exception as e:
                print(f"Erro ao treinar modelo {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return self.sanitize_for_json(results)
    
    def generate_plots(self, y_test):
        """Gera gráficos e salva na pasta outputs"""
        plot_paths = []
        
        for name, res in self.results.items():
            if 'error' in res:
                continue
                
            problem_type = res.get('problem_type', 'classification')
            
            if problem_type == 'classification':
                # Matriz de confusão para classificação
                if 'confusion_matrix' in res:
                    plt.figure(figsize=(8, 6))
                    cm = np.array(res['confusion_matrix'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Matriz de Confusão - {name}')
                    plt.xlabel('Predito')
                    plt.ylabel('Real')
                    
                    plot_path = f"outputs/confusion_matrix_{name}_{self.timestamp}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_paths.append(plot_path)
                
                # Gráfico de métricas de classificação
                plt.figure(figsize=(10, 6))
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                values = [res.get(metric, 0) for metric in metrics]
                
                bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
                plt.title(f'Métricas de Classificação - {name}')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                
                # Adiciona valores nas barras
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.3f}', ha='center', va='bottom')
                
                metrics_plot_path = f"outputs/metrics_{name}_{self.timestamp}.png"
                plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(metrics_plot_path)
                
            else:  # regression
                # Gráfico de predições vs valores reais
                plt.figure(figsize=(10, 6))
                y_pred = res['y_pred']
                plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Valores Reais')
                plt.ylabel('Predições')
                plt.title(f'Predições vs Valores Reais - {name}')
                
                plot_path = f"outputs/predictions_{name}_{self.timestamp}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(plot_path)
                
                # Gráfico de métricas de regressão
                plt.figure(figsize=(10, 6))
                metrics = ['r2_score', 'mse', 'mae', 'rmse']
                values = [res.get(metric, 0) for metric in metrics]
                
                # Normalizar MSE, MAE, RMSE para visualização (usar log scale)
                display_values = []
                for i, (metric, value) in enumerate(zip(metrics, values)):
                    if metric == 'r2_score':
                        display_values.append(max(0, value))  # R² pode ser negativo
                    else:
                        display_values.append(value)
                
                bars = plt.bar(metrics, display_values, color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon'])
                plt.title(f'Métricas de Regressão - {name}')
                plt.ylabel('Valor')
                
                # Adiciona valores nas barras
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(display_values) * 0.01, 
                            f'{value:.3f}', ha='center', va='bottom')
                
                metrics_plot_path = f"outputs/metrics_{name}_{self.timestamp}.png"
                plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(metrics_plot_path)
        
        return plot_paths
    
    def run_pipeline(self, file_path: str, file_type: str = 'csv'):
        """Executa o pipeline completo"""
        try:
            # 1. Carrega dados
            if file_type.lower() == 'csv':
                df = self.load_data_from_csv(file_path)
            elif file_type.lower() == 'sql':
                df = self.load_data_from_sql(file_path)
            else:
                raise Exception("Tipo de arquivo não suportado. Use 'csv' ou 'sql'")
            
            # 2. Pré-processamento
            X, y = self.preprocess_data(df)
            
            # 3. Divisão treino/teste
            # Verifica se é possível usar stratify
            try:
                # Tenta usar stratify se possível
                unique_values = y.dropna().unique() if hasattr(y, 'dropna') else np.unique(y[~pd.isna(y)])
                min_class_count = min([np.sum(y == val) for val in unique_values]) if len(unique_values) > 1 else 1
                
                if len(unique_values) > 1 and min_class_count >= 2 and len(y) >= 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    # Sem stratify para casos problemáticos
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
            except Exception:
                # Fallback sem stratify
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # 4. Treina e avalia modelos
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            
            # 5. Gera gráficos
            plot_paths = self.generate_plots(y_test)
            
            # 6. Limpa dados temporários (remove referências aos modelos para JSON)
            clean_results = {}
            for name, res in results.items():
                if 'error' not in res:
                    clean_res = res.copy()
                    clean_res.pop('model', None)  # Remove o modelo para serialização JSON
                    clean_res.pop('y_pred', None)  # Remove arrays grandes
                    clean_res.pop('y_proba', None)
                    clean_results[name] = clean_res
                else:
                    clean_results[name] = res
            
            return {
                'success': True,
                'data_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'features': len(X.columns),
                    'target_classes': len(np.unique(y))
                },
                'results': clean_results,
                'plots': plot_paths,
                'timestamp': self.timestamp
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': self.timestamp
            }