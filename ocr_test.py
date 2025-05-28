#!/usr/bin/env python3
"""
PDF Image Inspector - Check what images actually contain in the PDF
"""

import sys
from pathlib import Path

# Add parser to path
sys.path.append(str(Path(__file__).parent))

def inspect_pdf_images(pdf_path):
    """Inspect images in PDF to see if they contain text worth extracting."""
    try:
        import fitz  # PyMuPDF
        
        print(f"🔍 Inspecting images in: {pdf_path}")
        print("=" * 50)
        
        doc = fitz.open(pdf_path)
        
        total_images = 0
        text_bearing_images = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            
            print(f"\nPage {page_num + 1}: {len(image_list)} images")
            
            for img_index, img in enumerate(image_list):
                total_images += 1
                xref = img[0]
                
                try:
                    # Get image
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Check if it's a reasonable size for text
                    width, height = pix.width, pix.height
                    size_mb = len(pix.tobytes()) / (1024 * 1024)
                    
                    print(f"  Image {img_index}: {width}x{height} pixels, {size_mb:.2f}MB")
                    
                    # Heuristic: images with text are usually larger and not tiny decorative elements
                    if width > 100 and height > 50:  # Reasonable size for text
                        text_bearing_images += 1
                        print(f"    ✅ Likely contains text (size: {width}x{height})")
                        
                        # Try to extract with EasyOCR if available
                        try:
                            import easyocr
                            import numpy as np
                            from PIL import Image
                            
                            # Convert pixmap to PIL Image
                            img_data = pix.tobytes("ppm")
                            pil_img = Image.open(io.BytesIO(img_data))
                            img_array = np.array(pil_img)
                            
                            # Quick OCR test
                            reader = easyocr.Reader(['en'], verbose=False)
                            ocr_results = reader.readtext(img_array)
                            
                            if ocr_results:
                                # Get text with confidence > 0.5
                                text_found = [result[1] for result in ocr_results if result[2] > 0.5]
                                if text_found:
                                    print(f"    📝 OCR detected: {' '.join(text_found[:3])}...")  # First 3 items
                                else:
                                    print(f"    ⚠️  OCR found text but low confidence")
                            else:
                                print(f"    ❌ No text detected by OCR")
                                
                        except ImportError:
                            print(f"    🔧 EasyOCR not available for testing")
                        except Exception as e:
                            print(f"    ⚠️  OCR test failed: {str(e)[:50]}...")
                    else:
                        print(f"    ⏭️  Too small for text (likely decorative)")
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    print(f"  ❌ Error processing image {img_index}: {e}")
        
        print(f"\n📊 Summary:")
        print(f"  Total images: {total_images}")
        print(f"  Likely text-bearing: {text_bearing_images}")
        print(f"  Decorative/small: {total_images - text_bearing_images}")
        
        if text_bearing_images == 0:
            print(f"\n💡 Analysis: This PDF's text is likely already extracted by pdfplumber.")
            print(f"   The images appear to be decorative elements or logos without meaningful text.")
            print(f"   OCR may not add much value for this particular document.")
        else:
            print(f"\n💡 Analysis: {text_bearing_images} images might contain extractable text.")
            print(f"   OCR should be able to extract additional content from these images.")
        
        doc.close()
        
    except ImportError:
        print("❌ PyMuPDF not available. Install with: pip install PyMuPDF")
    except Exception as e:
        print(f"❌ Error inspecting PDF: {e}")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pdf_image_inspector.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    inspect_pdf_images(pdf_path)


if __name__ == "__main__":
    import io
    main()