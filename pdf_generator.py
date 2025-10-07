"""
PDFGenerator - Módulo de geração de PDFs para insights SmartBI
=============================================================

Este módulo é responsável por converter análises de texto/markdown em 
documentos PDF formatados e profissionais.

Autor: SmartBI Team
Versão: 1.0.0
"""

import io
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus import Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

logger = logging.getLogger(__name__)


class PDFGenerator:
    """
    Classe responsável pela geração de PDFs formatados para insights de negócio
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Inicializa o gerador de PDF
        
        Args:
            output_dir: Diretório onde os PDFs serão salvos
        """
        self.output_dir = output_dir
        self.ensure_output_directory()
        
        # Configurar estilos
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def ensure_output_directory(self):
        """
        Garante que o diretório de output existe
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"📁 Diretório criado: {self.output_dir}")
    
    def setup_custom_styles(self):
        """
        Configura estilos personalizados para o PDF
        """
        # Estilo para título principal
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Estilo para subtítulos
        self.styles.add(ParagraphStyle(
            name='SubTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=20,
            textColor=colors.darkred
        ))
        
        # Estilo para seções
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=15,
            textColor=colors.darkgreen
        ))
        
        # Estilo para texto normal com justificação
        self.styles.add(ParagraphStyle(
            name='JustifiedText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leftIndent=0,
            rightIndent=0
        ))
        
        # Estilo para listas
        self.styles.add(ParagraphStyle(
            name='ListItem',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=20,
            bulletIndent=10
        ))
        
        # Estilo para rodapé
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.grey
        ))
    
    def parse_markdown_to_elements(self, markdown_text: str) -> list:
        """
        Converte texto markdown em elementos do ReportLab
        
        Args:
            markdown_text: Texto em formato markdown
            
        Returns:
            Lista de elementos do ReportLab
        """
        elements = []
        lines = markdown_text.split('\n')
        
        current_paragraph = []
        in_list = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # Linha vazia - adiciona parágrafo atual se houver
                if current_paragraph:
                    text = ' '.join(current_paragraph)
                    elements.append(Paragraph(text, self.styles['JustifiedText']))
                    current_paragraph = []
                    in_list = False
                elements.append(Spacer(1, 6))
                continue
            
            # Títulos H1
            if line.startswith('# '):
                if current_paragraph:
                    text = ' '.join(current_paragraph)
                    elements.append(Paragraph(text, self.styles['JustifiedText']))
                    current_paragraph = []
                
                title = line[2:].strip()
                elements.append(Paragraph(title, self.styles['MainTitle']))
                in_list = False
                
            # Títulos H2
            elif line.startswith('## '):
                if current_paragraph:
                    text = ' '.join(current_paragraph)
                    elements.append(Paragraph(text, self.styles['JustifiedText']))
                    current_paragraph = []
                
                subtitle = line[3:].strip()
                elements.append(Paragraph(subtitle, self.styles['SubTitle']))
                in_list = False
                
            # Títulos H3
            elif line.startswith('### '):
                if current_paragraph:
                    text = ' '.join(current_paragraph)
                    elements.append(Paragraph(text, self.styles['JustifiedText']))
                    current_paragraph = []
                
                section = line[4:].strip()
                elements.append(Paragraph(section, self.styles['SectionTitle']))
                in_list = False
                
            # Itens de lista
            elif line.startswith('- ') or line.startswith('* '):
                if current_paragraph and not in_list:
                    text = ' '.join(current_paragraph)
                    elements.append(Paragraph(text, self.styles['JustifiedText']))
                    current_paragraph = []
                
                item_text = line[2:].strip()
                # Processar formatação em negrito e itálico
                item_text = self.process_text_formatting(item_text)
                elements.append(Paragraph(f"• {item_text}", self.styles['ListItem']))
                in_list = True
                
            # Texto normal
            else:
                if in_list:
                    # Se estava em lista e agora não é mais, adiciona espaço
                    elements.append(Spacer(1, 6))
                    in_list = False
                
                # Processar formatação em negrito e itálico
                formatted_line = self.process_text_formatting(line)
                current_paragraph.append(formatted_line)
        
        # Adicionar último parágrafo se houver
        if current_paragraph:
            text = ' '.join(current_paragraph)
            elements.append(Paragraph(text, self.styles['JustifiedText']))
        
        return elements
    
    def process_text_formatting(self, text: str) -> str:
        """
        Processa formatação markdown (negrito, itálico) para HTML do ReportLab
        
        Args:
            text: Texto com formatação markdown
            
        Returns:
            Texto com formatação HTML do ReportLab
        """
        # Negrito (**texto**)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Itálico (*texto*)
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
        
        return text
    
    def generate_pdf_from_analysis(self, analysis_data: Dict[str, Any], 
                                 filename: Optional[str] = None) -> str:
        """
        Gera PDF a partir dos dados de análise
        
        Args:
            analysis_data: Dados da análise (deve conter gemini_response ou strategic_insights)
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho do arquivo PDF gerado
        """
        try:
            # Determinar nome do arquivo
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"smartbi_analysis_{timestamp}.pdf"
            
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Criar documento PDF
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Preparar elementos do documento
            elements = []
            
            # Cabeçalho do documento
            elements.append(Paragraph("SmartBI Strategic Analysis", self.styles['MainTitle']))
            elements.append(Spacer(1, 20))
            
            # Informações do relatório
            info_data = [
                ["Data de Análise:", analysis_data.get('analyzed_at', datetime.now().isoformat())],
                ["Modelo Utilizado:", analysis_data.get('model_used', 'Gemini AI')],
                ["Tempo de Processamento:", f"{analysis_data.get('processing_time', 0)} segundos"]
            ]
            
            info_table = Table(info_data, colWidths=[2*inch, 3*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(info_table)
            elements.append(Spacer(1, 30))
            
            # Conteúdo da análise
            analysis_content = (analysis_data.get('gemini_response') or 
                              analysis_data.get('strategic_insights') or 
                              "Conteúdo da análise não disponível")
            
            # Converter markdown para elementos PDF
            content_elements = self.parse_markdown_to_elements(analysis_content)
            elements.extend(content_elements)
            
            # Rodapé
            elements.append(Spacer(1, 30))
            elements.append(Paragraph(
                f"Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')} | SmartBI Analytics Platform",
                self.styles['Footer']
            ))
            
            # Gerar PDF
            doc.build(elements)
            
            logger.info(f"✅ PDF gerado com sucesso: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar PDF: {e}")
            raise Exception(f"Erro na geração do PDF: {str(e)}")
    
    def generate_pdf_from_text(self, content: str, title: str = "SmartBI Analysis",
                             filename: Optional[str] = None) -> str:
        """
        Gera PDF a partir de texto simples
        
        Args:
            content: Conteúdo em texto/markdown
            title: Título do documento
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho do arquivo PDF gerado
        """
        analysis_data = {
            'gemini_response': content,
            'analyzed_at': datetime.now().isoformat(),
            'model_used': 'SmartBI',
            'processing_time': 0
        }
        
        return self.generate_pdf_from_analysis(analysis_data, filename)
    
    def get_output_directory(self) -> str:
        """
        Retorna o diretório onde os PDFs são salvos
        
        Returns:
            Caminho do diretório de output
        """
        return os.path.abspath(self.output_dir)