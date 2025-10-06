"""
SmartBI Assistant - Análise de Dados com Google Gemini
=====================================================

Sistema de análise de dados que utiliza a API do Google Gemini para gerar 
insights estratégicos de negócios a partir de bases de dados ou arquivos CSV.

Autor: SmartBI Team
Versão: 2.0.0
Data: 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Importações dos módulos locais
from gemini_analyzer import GeminiAnalyzer
from data_processor import DataProcessor
from database_connector import DatabaseConnector
from models import DatabaseConnectionRequest, AnalysisResponse, SpecificInsightRequest, SpecificInsightResponse

# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="SmartBI Assistant",
    description="Sistema de análise de dados com Google Gemini para insights estratégicos de negócios",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diretório para arquivos temporários
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Inicializar serviços
try:
    gemini_analyzer = GeminiAnalyzer()
    data_processor = DataProcessor()
    database_connector = DatabaseConnector()
    logger.info("✅ Serviços inicializados com sucesso")
except Exception as e:
    logger.error(f"❌ Erro ao inicializar serviços: {e}")
    raise


@app.get("/")
async def root():
    """
    Endpoint raiz com informações da API
    """
    return {
        "name": "SmartBI Assistant",
        "version": "2.0.0",
        "description": "Análise de dados com Google Gemini",
        "status": "online",
        "endpoints": {
            "/": "GET - Informações da API",
            "/upload": "POST - Upload de arquivo para análise",
            "/analyze-database": "POST - Análise via conexão com base de dados",
            "/specific-insights": "POST - Insights estratégicos específicos baseados em solicitação",
            "/health": "GET - Status de saúde da aplicação",
            "/docs": "GET - Documentação interativa (Swagger)",
            "/redoc": "GET - Documentação alternativa (ReDoc)"
        },
        "supported_formats": ["CSV", "SQL", "Database URL"],
        "supported_databases": ["MySQL", "PostgreSQL", "SQLite", "SQL Server", "Oracle"],
        "connection_types": ["File Upload", "Database URL"],
        "ai_provider": "Google Gemini",
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Endpoint de verificação de saúde da aplicação
    """
    try:
        # Verifica se o Gemini está configurado
        gemini_status = gemini_analyzer.check_connection()
        
        return {
            "status": "healthy" if gemini_status else "warning",
            "services": {
                "api": "online",
                "gemini": "configured" if gemini_status else "not_configured",
                "temp_directory": "available" if TEMP_DIR.exists() else "unavailable"
            },
            "message": "Aplicação funcionando normalmente" if gemini_status else "Gemini API não configurada"
        }
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return {
            "status": "error",
            "services": {
                "api": "online",
                "gemini": "error",
                "temp_directory": "unknown"
            },
            "error": str(e)
        }


@app.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Endpoint para upload de arquivo e análise com Gemini
    
    Args:
        file: Arquivo CSV ou SQL via multipart/form-data
    
    Returns:
        JSON com análise completa dos dados usando Gemini
    """
    
    # Log do início da requisição
    logger.info(f"📤 Upload iniciado: {file.filename}")
    
    try:
        # 1. Validações iniciais
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nome do arquivo é obrigatório")
        
        if file.size and file.size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=413, detail="Arquivo muito grande (máximo 50MB)")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.csv', '.sql']:
            raise HTTPException(
                status_code=400, 
                detail=f"Formato de arquivo não suportado: {file_extension}. Use .csv ou .sql"
            )
        
        # 2. Salva arquivo temporariamente
        temp_file_path = TEMP_DIR / file.filename
        
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"📁 Arquivo salvo: {temp_file_path} ({len(content)} bytes)")
            
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo: {e}")
            raise HTTPException(status_code=500, detail="Erro ao salvar arquivo")
        
        # 3. Processa os dados
        try:
            processed_data = data_processor.process_file(temp_file_path)
            logger.info(f"📊 Dados processados: {processed_data['info']}")
            
        except Exception as e:
            logger.error(f"Erro ao processar dados: {e}")
            raise HTTPException(status_code=400, detail=f"Erro ao processar dados: {str(e)}")
        
        # 4. Análise com Gemini
        try:
            gemini_analysis = await gemini_analyzer.analyze_data(
                data_content=processed_data['content'],
                data_info=processed_data['info'],
                file_type=file_extension[1:]  # Remove o ponto da extensão
            )
            logger.info("🤖 Análise Gemini concluída")
            
        except Exception as e:
            logger.error(f"Erro na análise Gemini: {e}")
            raise HTTPException(status_code=500, detail=f"Erro na análise com IA: {str(e)}")
        
        # 5. Preparar resposta
        response = {
            "success": True,
            "message": "Análise concluída com sucesso",
            "file_info": {
                "filename": file.filename,
                "size": len(content),
                "type": file_extension[1:],
                "processed_at": processed_data['info'].get('processed_at')
            },
            "data_summary": processed_data['info'],
            "gemini_response": gemini_analysis['gemini_response'],
            "processing_time": gemini_analysis.get('processing_time', 0),
            "model_used": gemini_analysis.get('model_used', 'gemini-2.5-flash-lite'),
            "analyzed_at": gemini_analysis.get('analyzed_at')
        }
        
        logger.info(f"✅ Análise concluída para: {file.filename}")
        return JSONResponse(content=response)
        
    except HTTPException:
        # Re-levanta HTTPExceptions (erros de validação)
        raise
        
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")
        
    finally:
        # 6. Limpeza: remove arquivo temporário
        try:
            if temp_file_path.exists():
                temp_file_path.unlink()
                logger.info(f"🗑️ Arquivo temporário removido: {temp_file_path}")
        except Exception as e:
            logger.warning(f"Aviso: Erro ao remover arquivo temporário: {e}")


@app.post("/analyze-database")
async def analyze_database(request: DatabaseConnectionRequest = Body(...)):
    """
    Endpoint para análise de dados via conexão direta com base de dados
    
    Args:
        request: Dados da requisição contendo apenas a URL da base de dados
    
    Returns:
        JSON com análise completa dos dados usando Gemini
        
    Example:
        POST /analyze-database
        {
            "database_url": "postgresql://user:password@host:port/database"
        }
    """
    
    # Log do início da requisição
    logger.info(f"🗄️ Análise de base de dados iniciada: {request.database_url}")
    
    try:
        db_connector = DatabaseConnector()
        
        # 1. Conecta à base de dados
        try:
            connection_info = db_connector.connect_to_database(request.database_url)
            logger.info(f"🔗 Conectado à {connection_info['database_type']}")
            
        except Exception as e:
            logger.error(f"Erro na conexão: {e}")
            raise HTTPException(status_code=400, detail=f"Erro de conexão com a base de dados: {str(e)}")
        
        # 2. Extrai dados da base de dados (usando valores padrão)
        try:
            # Extrai dados de amostra das tabelas com limite padrão de 1000 registros
            logger.info(f"📊 Extraindo dados de amostra...")
            sample_data = db_connector.extract_sample_data(limit=1000)
            
            logger.info(f"📋 Dados extraídos: {sample_data['tables_processed']} tabelas, {sample_data['total_records']} registros")
            
        except Exception as e:
            logger.error(f"Erro na extração de dados: {e}")
            raise HTTPException(status_code=400, detail=f"Erro ao extrair dados: {str(e)}")
        
        # 3. Prepara dados para análise
        try:
            data_content = db_connector.prepare_data_for_analysis(sample_data)
            logger.info("📝 Dados preparados para análise com Gemini")
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao preparar dados: {str(e)}")
        
        # 4. Análise com Gemini
        try:
            gemini_analysis = await gemini_analyzer.analyze_data(
                data_content=data_content,
                data_info=sample_data,
                file_type="database"
            )
            logger.info("🤖 Análise Gemini concluída")
            
        except Exception as e:
            logger.error(f"Erro na análise Gemini: {e}")
            raise HTTPException(status_code=500, detail=f"Erro na análise com IA: {str(e)}")
        
        # 5. Preparar resposta
        response = {
            "success": True,
            "message": "Análise de base de dados concluída com sucesso",
            "source_type": "database",
            "source_info": {
                "database_type": connection_info["database_type"],
                "host": connection_info["host"],
                "database": connection_info.get("database"),
                "connected_at": connection_info["connected_at"],
                "sample_limit": 1000
            },
            "data_summary": {
                "tables_processed": sample_data["tables_processed"],
                "total_records": sample_data["total_records"],
                "extraction_method": "table_sampling",
                "extracted_at": sample_data["extracted_at"]
            },
            "gemini_response": gemini_analysis['gemini_response'],
            "processing_time": gemini_analysis.get('processing_time', 0),
            "model_used": gemini_analysis.get('model_used', 'gemini-2.5-flash-lite'),
            "analyzed_at": gemini_analysis.get('analyzed_at')
        }
        
        logger.info(f"✅ Análise de base de dados concluída")
        return JSONResponse(content=response)
        
    except HTTPException:
        # Re-levanta HTTPExceptions (erros de validação)
        raise
        
    except Exception as e:
        logger.error(f"❌ Erro inesperado na análise de base de dados: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")
        
    finally:
        # 6. Fecha conexão com a base de dados
        try:
            db_connector.close_connection()
        except Exception as e:
            logger.warning(f"Aviso ao fechar conexão: {e}")


@app.post("/specific-insights")
async def get_specific_insights(request: SpecificInsightRequest = Body(...)):
    """
    Endpoint para gerar insights estratégicos específicos baseados em solicitação
    
    Args:
        request: Dados da requisição contendo URL da base de dados e solicitação de insight
    
    Returns:
        JSON com insights estratégicos específicos usando Gemini
        
    Examples:
        - "De acordo com os pedidos realizados neste mês, me dê insights sobre performance de vendas"
        - "Analise o comportamento dos clientes com base nos dados de compras dos últimos 6 meses"
        - "Quais produtos têm melhor margem de lucro e oportunidades de crescimento?"
    """
    
    # Log do início da requisição
    logger.info(f"🎯 Solicitação de insights específicos iniciada")
    logger.info(f"💭 Solicitação: {request.insight_request[:100]}...")
    
    try:
        db_connector = DatabaseConnector()
        
        # 1. Conecta à base de dados
        try:
            connection_info = db_connector.connect_to_database(request.database_url)
            logger.info(f"🔗 Conectado à {connection_info['database_type']}")
            
        except Exception as e:
            logger.error(f"Erro na conexão: {e}")
            raise HTTPException(status_code=400, detail=f"Erro de conexão com a base de dados: {str(e)}")
        
        # 2. Extrai esquema da base de dados
        try:
            logger.info(f"📋 Extraindo esquema da base de dados...")
            database_schema = db_connector.get_database_schema()
            logger.info(f"✅ Esquema extraído: {database_schema['total_tables']} tabelas")
            
        except Exception as e:
            logger.error(f"Erro ao extrair esquema: {e}")
            raise HTTPException(status_code=400, detail=f"Erro ao extrair esquema: {str(e)}")
        
        # 3. Extrai dados de amostra relevantes
        try:
            logger.info(f"📊 Extraindo dados de amostra para análise...")
            sample_data = db_connector.extract_sample_data(limit=1000)
            logger.info(f"✅ Dados extraídos: {sample_data['tables_processed']} tabelas, {sample_data['total_records']} registros")
            
        except Exception as e:
            logger.error(f"Erro na extração de dados: {e}")
            raise HTTPException(status_code=400, detail=f"Erro ao extrair dados: {str(e)}")
        
        # 4. Análise de insights específicos com Gemini
        try:
            logger.info(f"🤖 Gerando insights estratégicos específicos...")
            insights_analysis = await gemini_analyzer.analyze_specific_insights(
                database_schema=database_schema,
                sample_data=sample_data,
                insight_request=request.insight_request
            )
            logger.info("✅ Insights estratégicos gerados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro na análise de insights: {e}")
            raise HTTPException(status_code=500, detail=f"Erro na geração de insights: {str(e)}")
        
        # 5. Preparar resposta
        response = {
            "success": True,
            "message": "Insights estratégicos específicos gerados com sucesso",
            "insight_request": request.insight_request,
            "database_info": {
                "database_type": connection_info["database_type"],
                "host": connection_info["host"],
                "database": connection_info.get("database"),
                "total_tables": database_schema["total_tables"],
                "tables_analyzed": insights_analysis["tables_analyzed"],
                "records_analyzed": sample_data["total_records"],
                "connected_at": connection_info["connected_at"]
            },
            "strategic_insights": insights_analysis["strategic_insights"],
            "processing_time": insights_analysis["processing_time"],
            "model_used": insights_analysis["model_used"],
            "analyzed_at": insights_analysis["analyzed_at"]
        }
        
        logger.info(f"🎯 Insights específicos concluídos em {insights_analysis['processing_time']:.2f}s")
        return JSONResponse(content=response)
        
    except HTTPException:
        # Re-levanta HTTPExceptions (erros de validação)
        raise
        
    except Exception as e:
        logger.error(f"❌ Erro inesperado na geração de insights: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")
        
    finally:
        # 6. Fecha conexão com a base de dados
        try:
            db_connector.close_connection()
        except Exception as e:
            logger.warning(f"Aviso ao fechar conexão: {e}")


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handler personalizado para 404"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint não encontrado",
            "message": "Verifique a URL e tente novamente",
            "available_endpoints": ["/", "/upload", "/health", "/docs"]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handler personalizado para erros 500"""
    logger.error(f"Erro interno do servidor: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro interno do servidor",
            "message": "Algo deu errado. Tente novamente ou entre em contato com o suporte."
        }
    )


def main():
    """
    Função principal para inicializar o servidor
    """
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    environment = os.getenv("ENVIRONMENT", "development")
    
    logger.info(f"🚀 Iniciando SmartBI Assistant v2.0.0")
    logger.info(f"🌐 Servidor: http://{host}:{port}")
    logger.info(f"📚 Documentação: http://{host}:{port}/docs")
    logger.info(f"🔧 Ambiente: {environment}")
    
    # Configurações para desenvolvimento vs produção
    if environment == "development":
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning"
        )


if __name__ == "__main__":
    main()
