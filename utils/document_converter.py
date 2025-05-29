# utils/document_converter.py
"""
Document converter utility for converting unsupported formats to PDF or other supported formats.
Supports multiple conversion methods including LibreOffice, Office automation, and online APIs.
"""

import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import os
import platform

from utils.config import ParserConfig

logger = logging.getLogger(__name__)


class DocumentConverter:
    """
    Document converter that can convert various formats to PDF or other supported formats.
    Uses multiple conversion methods with fallback support.
    """
    
    def __init__(self, config: ParserConfig):
        """Initialize document converter with configuration."""
        self.config = config
        self.converter_config = getattr(config, 'converter', None)
        
        # Available conversion methods
        self.conversion_methods = []
        
        # Check for LibreOffice
        if self._check_libreoffice():
            self.conversion_methods.append('libreoffice')
            logger.info("LibreOffice converter available")
        
        # Check for Office automation (Windows only)
        if platform.system() == 'Windows':
            if self._check_office_automation():
                self.conversion_methods.append('office_automation')
                logger.info("Office automation converter available")
        
        # Check for online converters (if enabled)
        if self.converter_config and getattr(self.converter_config, 'enable_online', False):
            self.conversion_methods.append('online')
            logger.info("Online converter enabled")
        
        if not self.conversion_methods:
            logger.warning("No document conversion methods available")
        else:
            logger.info(f"Available conversion methods: {', '.join(self.conversion_methods)}")
    
    def convert_to_pdf(self, file_path: Path, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Convert document to PDF format.
        
        Args:
            file_path: Path to the source document
            output_dir: Directory for output file (uses temp if None)
            
        Returns:
            Path to converted PDF file or None if conversion failed
        """
        if not file_path.exists():
            logger.error(f"Source file not found: {file_path}")
            return None
        
        # Set output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        
        output_file = output_dir / f"{file_path.stem}.pdf"
        
        # Try conversion methods in order
        for method in self.conversion_methods:
            try:
                logger.info(f"Attempting {method} conversion for {file_path.name}")
                
                if method == 'libreoffice':
                    result = self._convert_with_libreoffice(file_path, output_dir)
                elif method == 'office_automation':
                    result = self._convert_with_office_automation(file_path, output_file)
                elif method == 'online':
                    result = self._convert_with_online_service(file_path, output_file)
                else:
                    continue
                
                if result and result.exists():
                    logger.info(f"Successfully converted {file_path.name} to PDF using {method}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Conversion with {method} failed: {str(e)}")
                continue
        
        logger.error(f"All conversion methods failed for {file_path.name}")
        return None
    
    def convert_to_pptx(self, ppt_file: Path, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Convert PPT file to PPTX format.
        
        Args:
            ppt_file: Path to the source PPT file
            output_dir: Directory for output file (uses temp if None)
            
        Returns:
            Path to converted PPTX file or None if conversion failed
        """
        if not ppt_file.exists():
            logger.error(f"Source PPT file not found: {ppt_file}")
            return None
        
        # Set output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        
        output_file = output_dir / f"{ppt_file.stem}.pptx"
        
        # Try conversion methods
        for method in self.conversion_methods:
            try:
                logger.info(f"Attempting {method} conversion for {ppt_file.name}")
                
                if method == 'libreoffice':
                    result = self._convert_ppt_with_libreoffice(ppt_file, output_dir)
                elif method == 'office_automation':
                    result = self._convert_ppt_with_office_automation(ppt_file, output_file)
                else:
                    continue
                
                if result and result.exists():
                    logger.info(f"Successfully converted {ppt_file.name} to PPTX using {method}")
                    return result
                    
            except Exception as e:
                logger.warning(f"PPT conversion with {method} failed: {str(e)}")
                continue
        
        logger.error(f"All PPT conversion methods failed for {ppt_file.name}")
        return None
    
    def _check_libreoffice(self) -> bool:
        """Check if LibreOffice is available."""
        try:
            # Try different LibreOffice commands
            commands = ['libreoffice', 'soffice', '/usr/bin/libreoffice', '/usr/bin/soffice']
            
            for cmd in commands:
                try:
                    result = subprocess.run([cmd, '--version'], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=10)
                    if result.returncode == 0:
                        logger.debug(f"LibreOffice found: {cmd}")
                        self.libreoffice_cmd = cmd
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
                    
        except Exception as e:
            logger.debug(f"LibreOffice check failed: {str(e)}")
        
        return False
    
    def _check_office_automation(self) -> bool:
        """Check if Office automation is available (Windows only)."""
        if platform.system() != 'Windows':
            return False
        
        try:
            import win32com.client
            return True
        except ImportError:
            logger.debug("win32com not available for Office automation")
            return False
    
    def _convert_with_libreoffice(self, file_path: Path, output_dir: Path) -> Optional[Path]:
        """Convert document using LibreOffice."""
        try:
            # Create command
            cmd = [
                self.libreoffice_cmd,
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', str(output_dir),
                str(file_path)
            ]
            
            # Run conversion
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=60)
            
            if result.returncode == 0:
                # Find the generated PDF
                expected_pdf = output_dir / f"{file_path.stem}.pdf"
                if expected_pdf.exists():
                    return expected_pdf
                
                # Sometimes LibreOffice creates different filename
                for pdf_file in output_dir.glob("*.pdf"):
                    if pdf_file.stem == file_path.stem:
                        return pdf_file
            
            logger.error(f"LibreOffice conversion failed: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            logger.error("LibreOffice conversion timed out")
        except Exception as e:
            logger.error(f"LibreOffice conversion error: {str(e)}")
        
        return None
    
    def _convert_ppt_with_libreoffice(self, ppt_file: Path, output_dir: Path) -> Optional[Path]:
        """Convert PPT to PPTX using LibreOffice."""
        try:
            # Create command for PPT to PPTX conversion
            cmd = [
                self.libreoffice_cmd,
                '--headless',
                '--convert-to', 'pptx',
                '--outdir', str(output_dir),
                str(ppt_file)
            ]
            
            # Run conversion
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=60)
            
            if result.returncode == 0:
                expected_pptx = output_dir / f"{ppt_file.stem}.pptx"
                if expected_pptx.exists():
                    return expected_pptx
                
                # Find any generated PPTX file
                for pptx_file in output_dir.glob("*.pptx"):
                    if pptx_file.stem == ppt_file.stem:
                        return pptx_file
            
            logger.error(f"LibreOffice PPT conversion failed: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            logger.error("LibreOffice PPT conversion timed out")
        except Exception as e:
            logger.error(f"LibreOffice PPT conversion error: {str(e)}")
        
        return None
    
    def _convert_with_office_automation(self, file_path: Path, output_file: Path) -> Optional[Path]:
        """Convert document using Office automation (Windows only)."""
        try:
            import win32com.client
            
            extension = file_path.suffix.lower()
            
            if extension in ['.doc', '.docx']:
                return self._convert_word_with_automation(file_path, output_file)
            elif extension in ['.ppt', '.pptx']:
                return self._convert_powerpoint_with_automation(file_path, output_file)
            elif extension in ['.xls', '.xlsx']:
                return self._convert_excel_with_automation(file_path, output_file)
            
        except ImportError:
            logger.error("win32com not available for Office automation")
        except Exception as e:
            logger.error(f"Office automation conversion error: {str(e)}")
        
        return None
    
    def _convert_word_with_automation(self, file_path: Path, output_file: Path) -> Optional[Path]:
        """Convert Word document using automation."""
        try:
            import win32com.client
            
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            
            try:
                doc = word.Documents.Open(str(file_path))
                doc.SaveAs(str(output_file), FileFormat=17)  # PDF format
                doc.Close()
                
                if output_file.exists():
                    return output_file
                    
            finally:
                word.Quit()
                
        except Exception as e:
            logger.error(f"Word automation conversion failed: {str(e)}")
        
        return None
    
    def _convert_powerpoint_with_automation(self, file_path: Path, output_file: Path) -> Optional[Path]:
        """Convert PowerPoint presentation using automation."""
        try:
            import win32com.client
            
            powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            powerpoint.Visible = False
            
            try:
                presentation = powerpoint.Presentations.Open(str(file_path))
                presentation.SaveAs(str(output_file), FileFormat=32)  # PDF format
                presentation.Close()
                
                if output_file.exists():
                    return output_file
                    
            finally:
                powerpoint.Quit()
                
        except Exception as e:
            logger.error(f"PowerPoint automation conversion failed: {str(e)}")
        
        return None
    
    def _convert_ppt_with_office_automation(self, ppt_file: Path, output_file: Path) -> Optional[Path]:
        """Convert PPT to PPTX using Office automation."""
        try:
            import win32com.client
            
            powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            powerpoint.Visible = False
            
            try:
                presentation = powerpoint.Presentations.Open(str(ppt_file))
                presentation.SaveAs(str(output_file), FileFormat=24)  # PPTX format
                presentation.Close()
                
                if output_file.exists():
                    return output_file
                    
            finally:
                powerpoint.Quit()
                
        except Exception as e:
            logger.error(f"PPT to PPTX automation conversion failed: {str(e)}")
        
        return None
    
    def _convert_excel_with_automation(self, file_path: Path, output_file: Path) -> Optional[Path]:
        """Convert Excel spreadsheet using automation."""
        try:
            import win32com.client
            
            excel = win32com.client.Dispatch("Excel.Application")
            excel.Visible = False
            
            try:
                workbook = excel.Workbooks.Open(str(file_path))
                workbook.SaveAs(str(output_file), FileFormat=57)  # PDF format
                workbook.Close()
                
                if output_file.exists():
                    return output_file
                    
            finally:
                excel.Quit()
                
        except Exception as e:
            logger.error(f"Excel automation conversion failed: {str(e)}")
        
        return None
    
    def _convert_with_online_service(self, file_path: Path, output_file: Path) -> Optional[Path]:
        """Convert document using online service (placeholder)."""
        # This is a placeholder for online conversion services
        # You could integrate with services like:
        # - CloudConvert API
        # - Convertio API
        # - ILovePDF API
        # - Custom conversion service
        
        logger.warning("Online conversion service not implemented")
        return None
    
    def get_supported_conversions(self) -> Dict[str, List[str]]:
        """Get list of supported conversion formats."""
        conversions = {
            'to_pdf': [],
            'to_pptx': [],
            'to_docx': []
        }
        
        if 'libreoffice' in self.conversion_methods:
            conversions['to_pdf'].extend([
                'doc', 'docx', 'odt', 'rtf', 'txt',
                'ppt', 'pptx', 'odp',
                'xls', 'xlsx', 'ods', 'csv'
            ])
            conversions['to_pptx'].extend(['ppt'])
            conversions['to_docx'].extend(['doc', 'odt', 'rtf'])
        
        if 'office_automation' in self.conversion_methods:
            conversions['to_pdf'].extend(['doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx'])
            conversions['to_pptx'].extend(['ppt'])
            conversions['to_docx'].extend(['doc'])
        
        # Remove duplicates
        for key in conversions:
            conversions[key] = list(set(conversions[key]))
        
        return conversions
    
    def can_convert(self, file_path: Path, target_format: str) -> bool:
        """Check if a file can be converted to target format."""
        extension = file_path.suffix.lower().lstrip('.')
        supported = self.get_supported_conversions()
        
        target_key = f'to_{target_format}'
        if target_key in supported:
            return extension in supported[target_key]
        
        return False
    
    def get_conversion_info(self) -> Dict[str, Any]:
        """Get information about available conversion methods."""
        return {
            'available_methods': self.conversion_methods,
            'supported_conversions': self.get_supported_conversions(),
            'libreoffice_available': 'libreoffice' in self.conversion_methods,
            'office_automation_available': 'office_automation' in self.conversion_methods,
            'online_conversion_available': 'online' in self.conversion_methods,
            'platform': platform.system()
        }