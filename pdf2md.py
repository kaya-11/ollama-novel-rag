from logging.config import fileConfig

import pdfplumber
from markdownify import markdownify as md

def pdf_to_markdown(pdf_path, output_md_path):
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""

        # Extract text from each page
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"

    # Optionally, you can process the text with markdownify
    # This is useful if the PDF content is HTML-based, but for plain text PDFs, you can skip it.
    #markdown_text = md(full_text)
    markdown_text = full_text

    #markdown_text = full_text  # If the content is plain text without needing HTML conversion.

    # Save the result to a Markdown file
    with open(output_md_path, 'w') as md_file:
        md_file.write(markdown_text)

book_title=('prozess-inhalt')

pdf_to_markdown(f'data/pdfs/{book_title}.pdf', f'data/books/{book_title}.md')
