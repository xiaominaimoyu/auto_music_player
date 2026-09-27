import importlib.util
import os
import tempfile
import unittest

from core.document_import import import_document


@unittest.skipUnless(importlib.util.find_spec("docx"), "python-docx 未安装")
class DocumentImportTests(unittest.TestCase):
    def test_docx_requires_review_and_preserves_text(self):
        from docx import Document

        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "jianpu.docx")
            document = Document()
            document.add_paragraph("1 2 3_ 4")
            document.save(path)
            result = import_document(path)
            self.assertEqual(result.kind, "document")
            self.assertTrue(result.source_metadata["manual_confirmation_required"])
            self.assertEqual(result.raw_text.strip(), "1 2 3_ 4")
            self.assertEqual(len(result.notes), 4)

    @unittest.skipUnless(importlib.util.find_spec("pypdf"), "pypdf 未安装")
    def test_real_pdf_text_is_extracted_and_requires_review(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "jianpu.pdf")
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=300)
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            font_ref = writer._add_object(font)
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): font_ref}
                    )
                }
            )
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 72 200 Td (1 2 3 4) Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
            with open(path, "wb") as output:
                writer.write(output)

            result = import_document(path)
            self.assertEqual(result.kind, "document")
            self.assertEqual(result.source_metadata["tool"], "pypdf")
            self.assertTrue(result.source_metadata["manual_confirmation_required"])
            self.assertIn("1 2 3 4", result.raw_text)
            self.assertEqual(len(result.notes), 4)


if __name__ == "__main__":
    unittest.main()
