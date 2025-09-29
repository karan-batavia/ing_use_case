import pdfplumber

pdf_path = r"kbo_project.pdf"

with pdfplumber.open(pdf_path) as pdf:
    full_text = ""
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        #print(f"--- Page {i} ---")
        print(text)
        full_text += text + "\n"

# Save all text to a .txt file
output_file = pdf_path.replace(".pdf", "_text.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\n✅ Text extracted and saved to: {output_file}")
