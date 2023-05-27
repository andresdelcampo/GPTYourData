using System.Text.Json;
using OpenAI.Embeddings;

namespace GPTYourDataProgram;

class Program
{
    private const string EmbeddingsFolder = "Embeddings";
    private const string ConsolidatedFolder = "Consolidated";

    static async Task Main(string[] args)
    {
        Console.WriteLine("This program converts the embeddings of V1.0 to V1.1. You need to manually move the embeddings from Consolidated folder to Embeddings replacing the existing ones. Trying to upgrade embeddings upgraded already will fail.");
        
        // Create directory to store consolidated embeddings
        Directory.CreateDirectory(ConsolidatedFolder);

        // Get all json files from the folder
        var jsonFiles = Directory.GetFiles(EmbeddingsFolder, "*.json");

        // Group files by source filename
        var groupedFiles = jsonFiles.GroupBy(f => 
        {
            var splitName = Path.GetFileName(f).Split('_');
            return string.Join('_', splitName.Skip(1).Take(splitName.Length - 2));
        });

        // Process each group
        foreach (var group in groupedFiles)
        {
            await ConsolidateFiles(group.Key, group.ToList());
        }
    }

    private static async Task ConsolidateFiles(string sourceFileName, List<string> filePaths)
    {
        List<EmbeddingData> embeddingsList = new List<EmbeddingData>();

        foreach (var filePath in filePaths)
        {
            string json = await File.ReadAllTextAsync(filePath);
            if (!string.IsNullOrEmpty(json))
            {
                var embeddingObject = JsonSerializer.Deserialize<EmbeddingData>(json);

                // Removes the sourceFileName from the embeddingObject
                embeddingObject!.sourceFileName = null;
                embeddingsList.Add(embeddingObject);
            }
        }

        await WriteConsolidatedFile(sourceFileName, embeddingsList);
    }

    private static async Task WriteConsolidatedFile(string filename, List<EmbeddingData> embeddingsList)
    {
        // Create an object that includes the list of embeddings and the source file name
        var outputObject = new
        {
            sourceFileName = filename,
            embeddings = embeddingsList
        };

        // Write the embeddings JSON file with the filename in Consolidated folder
        string jsonFileName = $"embed_{filename}.json";
        string jsonFilePath = Path.Combine(ConsolidatedFolder, jsonFileName);

        string json = JsonSerializer.Serialize(outputObject);
        await File.WriteAllTextAsync(jsonFilePath, json);
        Console.WriteLine($"Consolidated file written: {jsonFilePath}");
    }
    
#pragma warning disable 8618
    public class EmbeddingData
    {
        public string text { get; set; }
        public EmbeddingsResponse embeddings { get; set; }
        public string? sourceFileName { get; set; }
    }
#pragma warning restore 8618
}