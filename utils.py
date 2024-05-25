from llmsherpa.readers import LayoutPDFReader

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

def pdf_to_text(pdf_path):
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    pdf = pdf_reader.read_pdf(pdf_path)
    docs = []

    for chunk in pdf.chunks():
        if len(docs) < chunk.page_idx + 1:
            docs.append('')
        docs[chunk.page_idx] += chunk.to_context_text() 

    return docs