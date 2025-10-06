"""
GeminiAnalyzer - Módulo de análise usando Google Gemini API
===========================================================

Este módulo é responsável por enviar dados para o Google Gemini e processar
as respostas para gerar insights estratégicos de negócios.

Autor: SmartBI Team
Versão: 2.0.0
"""

import google.generativeai as genai
import os
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

logger = logging.getLogger(__name__)


class GeminiAnalyzer:
    """
    Classe responsável pela análise de dados usando Google Gemini API
    """
    
    def __init__(self):
        """
        Inicializa o analisador Gemini
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.5-flash-lite"  # Modelo mais recente e eficiente
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info("✅ Gemini API configurada com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro ao configurar Gemini API: {e}")
                raise
        else:
            logger.warning("⚠️ GEMINI_API_KEY não encontrada nas variáveis de ambiente")
    
    
    def check_connection(self) -> bool:
        """
        Verifica se a conexão com Gemini está funcionando
        
        Returns:
            bool: True se conectado, False caso contrário
        """
        try:
            if not self.model:
                return False
            
            # Teste simples para verificar a conexão
            response = self.model.generate_content("Test connection")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar conexão Gemini: {e}")
            return False
    
    
    def _create_analysis_prompt(self, data_content: str, data_info: Dict, file_type: str) -> str:
        """
        Cria o prompt para análise estratégica de negócios
        
        Args:
            data_content: Conteúdo dos dados (SQL/CSV)
            data_info: Informações sobre os dados
            file_type: Tipo do arquivo (csv/sql)
        
        Returns:
            str: Prompt formatado para o Gemini
        """
        
        # Prompt estratégico direto
        main_prompt = """
Analise o SQL ou csv que você enviou (estrutura das tabelas, inserts, dados) e traga um resumo em linguagem de negócios.
Você é um consultor estratégico sênior de uma das Big 4 (McKinsey, BCG, Bain, Deloitte) analisando dados empresariais para o C-Level de uma organização.
OBJETIVO: Analise os dados fornecidos e gere insights estratégicos como um partner experiente de consultoria de negócios.
METODOLOGIA EXIGIDA:
RESUMO EXECUTIVO: Traduza a estrutura e volume de dados para linguagem de negócios, identificando maturidade organizacional, complexidade operacional e posicionamento competitivo
INSIGHTS ESTRATÉGICOS: Desenvolva 4-6 insights de alto impacto focando em:
Oportunidades de crescimento e expansão de mercado
Otimização de revenue streams e pricing strategy
Eficiência operacional e redução de custos
Vantagem competitiva e diferenciação
Gestão de risco e compliance
Potencial de transformação digital
RECOMENDAÇÕES EXECUTIVAS: Forneça 5-7 recomendações específicas e acionáveis com:
Impacto financeiro estimado (ROI, revenue upside, cost savings)
Timeline de implementação
Priorização baseada em esforço vs impacto
Considerações de investimento e recursos
ESTILO E TOM:
Linguagem executiva sofisticada (C-Level appropriate)
Quantificação de oportunidades com métricas de negócio
Foco em value creation e competitive advantage
Referências a frameworks estratégicos (Porter, Ansoff, Blue Ocean)
Benchmarking setorial e best practices
IMPORTANTE:
NÃO mencione aspectos técnicos de TI, programação ou detalhes de implementação
FOQUE exclusivamente em strategic business value
USE linguagem de consultoria estratégica empresarial
QUANTIFIQUE oportunidades sempre que possível
PRIORIZE insights que impactem P&L, market share ou operational excellence

DADOS PARA ANÁLISE:
"""
        
        return f"{main_prompt}\n\n{data_content}"
    
    

    
    
    async def analyze_data(self, data_content: str, data_info: Dict, file_type: str) -> Dict[str, Any]:
        """
        Realiza análise dos dados usando Gemini
        
        Args:
            data_content: Conteúdo dos dados (SQL/CSV)
            data_info: Informações sobre os dados
            file_type: Tipo do arquivo (csv/sql)
        
        Returns:
            Dict: Resposta do Gemini em formato JSON
        """
        
        if not self.model:
            raise Exception("Gemini API não configurada. Verifique a GEMINI_API_KEY")
        
        start_time = time.time()
        
        try:
            # 1. Criar prompt
            prompt = self._create_analysis_prompt(data_content, data_info, file_type)
            
            logger.info(f"🤖 Enviando dados para Gemini (prompt: {len(prompt)} chars)")
            
            # 2. Enviar para Gemini
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                raise Exception("Resposta vazia do Gemini")
            
            response_text = response.text.strip()
            logger.info(f"✅ Resposta recebida do Gemini ({len(response_text)} chars)")
            
            # 3. Preparar resultado final (apenas a resposta do Gemini)
            processing_time = time.time() - start_time
            
            result = {
                "gemini_response": response_text,
                "processing_time": round(processing_time, 2),
                "model_used": self.model_name,
                "analyzed_at": datetime.now().isoformat()
            }
            
            logger.info(f"🎯 Análise concluída em {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise Gemini: {e}")
            raise Exception(f"Erro ao processar análise: {str(e)}")
    


    async def analyze_specific_insights(self, database_schema: Dict, sample_data: Dict, insight_request: str) -> Dict[str, Any]:
        """
        Analisa dados da base de dados para gerar insights específicos solicitados
        
        Args:
            database_schema: Esquema da base de dados
            sample_data: Dados de amostra extraídos
            insight_request: Solicitação específica de insight
            
        Returns:
            Dict com a análise de insights específicos
        """
        try:
            start_time = time.time()
            
            # 1. Preparar contexto da base de dados
            database_context = self._prepare_database_context(database_schema, sample_data)
            
            # 2. Construir prompt especializado para insights específicos
            prompt = self._build_specific_insights_prompt(database_context, insight_request)
            
            logger.info(f"🎯 Iniciando análise de insights específicos...")
            logger.info(f"📋 Solicitação: {insight_request[:100]}...")
            
            # 3. Gerar resposta com Gemini
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                raise Exception("Resposta vazia do Gemini")
            
            response_text = response.text.strip()
            logger.info(f"✅ Insights específicos gerados ({len(response_text)} chars)")
            
            # 4. Preparar resultado final
            processing_time = time.time() - start_time
            
            result = {
                "strategic_insights": response_text,
                "insight_request": insight_request,
                "processing_time": round(processing_time, 2),
                "model_used": self.model_name,
                "analyzed_at": datetime.now().isoformat(),
                "database_context_size": len(database_context),
                "tables_analyzed": len(database_schema.get('tables', {}))
            }
            
            logger.info(f"🎯 Análise de insights específicos concluída em {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise de insights específicos: {e}")
            raise Exception(f"Erro ao processar insights específicos: {str(e)}")
    
    def _prepare_database_context(self, database_schema: Dict, sample_data: Dict) -> str:
        """
        Prepara contexto estruturado da base de dados para análise de insights
        
        Args:
            database_schema: Esquema da base de dados
            sample_data: Dados de amostra
            
        Returns:
            String com contexto formatado
        """
        context = []
        
        # Informações gerais
        context.append("=== CONTEXTO DA BASE DE DADOS ===")
        context.append(f"Total de Tabelas: {database_schema.get('total_tables', 0)}")
        context.append(f"Dados Extraídos: {sample_data.get('total_records', 0)} registros")
        context.append("")
        
        # Esquema das tabelas
        context.append("=== ESTRUTURA DAS TABELAS ===")
        for table_name, table_info in database_schema.get('tables', {}).items():
            context.append(f"\n📊 TABELA: {table_name}")
            context.append(f"   - Registros: {table_info.get('row_count', 0)}")
            context.append(f"   - Colunas: {table_info.get('column_count', 0)}")
            
            # Principais colunas
            columns = table_info.get('columns', [])[:10]  # Primeiras 10 colunas
            if columns:
                context.append("   - Estrutura:")
                for col in columns:
                    context.append(f"     * {col['name']}: {col['type']} {'(PK)' if col.get('primary_key') else ''}")
        
        # Dados de amostra das principais tabelas
        context.append("\n=== DADOS DE AMOSTRA ===")
        for table_name, data in sample_data.get('data', {}).items():
            if data and len(data) > 0:
                context.append(f"\n📋 AMOSTRA: {table_name}")
                # Mostra apenas os primeiros 5 registros
                for i, record in enumerate(data[:5]):
                    context.append(f"   Registro {i+1}: {record}")
                
                if len(data) > 5:
                    context.append(f"   ... e mais {len(data) - 5} registros disponíveis")
        
        return "\n".join(context)
    
    def _build_specific_insights_prompt(self, database_context: str, insight_request: str) -> str:
        """
        Constrói prompt especializado para insights específicos
        
        Args:
            database_context: Contexto da base de dados
            insight_request: Solicitação específica de insight
            
        Returns:
            Prompt formatado para Gemini
        """
        return f"""
Você é um consultor estratégico sênior de uma das Big 4 (McKinsey, BCG, Bain, Deloitte) especializado em análise de dados e business intelligence.

Sua missão é analisar os dados da base de dados fornecida e gerar insights estratégicos específicos baseados na solicitação do cliente C-Level.

CONTEXTO DA BASE DE DADOS:
{database_context}

SOLICITAÇÃO ESPECÍFICA DO CLIENTE:
"{insight_request}"

INSTRUÇÕES PARA ANÁLISE ESTRATÉGICA:

1. **ANÁLISE CONTEXTUAL**:
   - Analise os dados disponíveis na base de dados
   - Identifique padrões, tendências e anomalias relevantes à solicitação
   - Considere as relações entre diferentes tabelas e métricas

2. **INSIGHTS ESTRATÉGICOS**:
   - Gere insights específicos que respondam diretamente à solicitação
   - Foque em implicações de negócio e oportunidades estratégicas
   - Identifique riscos, oportunidades e recomendações acionáveis

3. **ESTRUTURA DA RESPOSTA**:
   - **EXECUTIVE SUMMARY**: Resumo executivo dos principais achados
   - **ANÁLISE DETALHADA**: Análise aprofundada dos dados relevantes
   - **INSIGHTS ESTRATÉGICOS**: Principais insights e descobertas
   - **RECOMENDAÇÕES**: Ações específicas recomendadas
   - **PRÓXIMOS PASSOS**: Sugestões para implementação

4. **ESTILO DE COMUNICAÇÃO**:
   - Linguagem executiva, clara e objetiva
   - Foque em valor de negócio e impacto estratégico
   - Use métricas e dados concretos sempre que possível
   - Apresente conclusões acionáveis para tomada de decisão

IMPORTANTE: Sua resposta deve ser específica à solicitação feita e baseada nos dados reais da base de dados. Evite generalizações e foque em insights práticos e estratégicos.

Gere sua análise estratégica em português brasileiro:
"""

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo Gemini em uso
        
        Returns:
            Dict: Informações do modelo
        """
        return {
            "model_name": self.model_name,
            "provider": "Google Gemini",
            "configured": self.model is not None,
            "api_key_present": bool(self.api_key)
        }
