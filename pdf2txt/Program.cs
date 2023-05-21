using System.Text;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor;

namespace pdf2txt;

public static class Program
{
    const string InputsFolder = "Input";

    public static void Main()
    {
        // Get all pdf files from the folder
        var pdfFiles = Directory.GetFiles(InputsFolder, "*.pdf");

        // Process each file
        foreach (var pdfFilePath in pdfFiles)
        {
            string outFileName = pdfFilePath.Replace(".pdf", ".txt");
            if (!File.Exists(outFileName))
                File.Delete(outFileName);
            
            using (PdfDocument document = PdfDocument.Open(pdfFilePath))
            {
                StringBuilder extractedText = new StringBuilder();

                foreach (Page page in document.GetPages())
                {
                    var words = page.GetWords(NearestNeighbourWordExtractor.Instance);
                    var blocks = RecursiveXYCut.Instance.GetBlocks(words);

                    foreach (var block in blocks)
                    {
                        extractedText.AppendLine(block.Text);
                        extractedText.AppendLine();
                    }
                }

                File.WriteAllText(outFileName, extractedText.ToString(), Encoding.UTF8);
            }
        }
    }
}