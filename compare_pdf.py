import pymupdf
from pypdf import PdfReader


PDF = r"D:\kb_docs\Fundamentals_of_Database_Systems,_6th_Edition.pdf"


doc = pymupdf.open(PDF)
page_no = next(i for i, page in enumerate(doc) if "EMP_DEPT" in page.get_text())


print(f"===== pypdf (page {page_no + 1}) =====")
print(PdfReader(PDF).pages[page_no].extract_text()[:1000])
print(f"\n===== PyMuPDF (page {page_no + 1}) =====")
print(doc[page_no].get_text()[:1000])