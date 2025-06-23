import fitz  # PyMuPDF

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_index, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                "page": page_index + 1,
                "image_bytes": base_image["image"],
                "ext": base_image["ext"]
            })
    return images

