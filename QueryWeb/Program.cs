using System.Diagnostics;
using System.Text;
using System.Text.Json;
using OpenAI;
using OpenAI.Completions;
using OpenAI.Embeddings;
using OpenAI.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Http;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;
using System.Linq;

OpenAIClient? Api;
const string OpenaiApiKeyFileName = "OpenAI-API.key";
const string EmbeddingsFolder = "Embeddings";
const double HighSimilarityThreshold = 0.8;     // For determining what documents to include in the context

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/api/gptquery", async (HttpContext httpContext) =>
{
    var formCollection = await httpContext.Request.ReadFormAsync();
    
    string query = formCollection["query"]!;
    string apiKey = "";
    if (File.Exists(OpenaiApiKeyFileName))
        apiKey = File.ReadAllText(OpenaiApiKeyFileName);

    if (apiKey.Length < 50)
    {
        return Results.BadRequest("The OpenAI API key is missing or invalid. Please add it to the OpenAI-API.key file");
    }

    Api = new OpenAIClient(apiKey, Model.Ada);

    EmbeddingsResponse? queryEmbed = await Api.EmbeddingsEndpoint.CreateEmbeddingAsync(query);
    Debug.Assert(queryEmbed != null);

    double[] queryVector = queryEmbed.Data.SelectMany(datum => datum.Embedding).ToArray();
    queryVector = NormalizeVector(queryVector); // Normalize the query vector

    // Get all json files from the folder
    var jsonFiles = Directory.GetFiles(EmbeddingsFolder, "*.json");

    // Reading files
    var documentVectors = new List<double[]>();
    var documents = new List<string>();
    foreach (var filePath in jsonFiles)
    {
        string embeddingsJson = await File.ReadAllTextAsync(filePath);
        var fileData = JsonSerializer.Deserialize<ConsolidatedEmbeddingsFileData>(embeddingsJson);
        Debug.Assert(fileData != null);

        foreach (var embeddingObject in fileData.embeddings)
        {
            double[] documentVector = embeddingObject.embeddings.Data.SelectMany(datum => datum.Embedding).ToArray();
            documentVector = NormalizeVector(documentVector); // Normalize the document vector
            documentVectors.Add(documentVector);
            documents.Add(embeddingObject.text);
        }
    }

    // Finding in files
    var similarities = new List<double>();
    for (int i = 0; i < documentVectors.Count; i++)
    {
        similarities.Add(CosineSimilarity(queryVector, documentVectors[i]));
    }

    var sortedDocuments = documents.Zip(similarities, (d, s) => new { Document = d, Similarity = s })
        .OrderByDescending(x => x.Similarity)
        .ToList();

    // Build the context from all highly similar documents
    const int maxChars = 8192;
    var contextBuilder = new StringBuilder();
    var tokenCount = 0;

    foreach (var doc in sortedDocuments.Where(x => x.Similarity >= HighSimilarityThreshold))
    {
        var docTokens = System.Text.Encoding.UTF8.GetByteCount(doc.Document);
        if (tokenCount + docTokens > maxChars)
            break; // Stop if the next document would exceed the maximum token count

        contextBuilder.AppendLine(doc.Document).AppendLine();
        tokenCount += docTokens;
    }
    var context = contextBuilder.ToString();

    // If no good matches were found:
    if (tokenCount == 0 || string.IsNullOrWhiteSpace(context))
    {
        return Results.NotFound("No good matches found.");
    }

    string completeQuery = @$"The following information is provided for context: \n\n{context} \n\n Given this information, can you please answer the following question: \n\n ""{query}""?";
    CompletionResult? result = await Api.CompletionsEndpoint.CreateCompletionAsync(completeQuery, model: Model.Davinci, temperature: 0.7, max_tokens: 256);

    // And if a result was found:
    return Results.Ok(result.ToString().TrimStart());
});

app.Run();


static double[] NormalizeVector(double[] vector)
{
    double length = Math.Sqrt(vector.Sum(x => x * x));
    return vector.Select(x => x / length).ToArray();
}
    
static double CosineSimilarity(double[] vector1, double[] vector2)
{
    double dotProduct = 0.0f;
    double magnitude1 = 0.0f;
    double magnitude2 = 0.0f;
    for (int i = 0; i < vector1.Length; i++)
    {
        dotProduct += vector1[i] * vector2[i];
        magnitude1 += vector1[i] * vector1[i];
        magnitude2 += vector2[i] * vector2[i];
    }

    magnitude1 = (float)Math.Sqrt(magnitude1);
    magnitude2 = (float)Math.Sqrt(magnitude2);
    if (magnitude1 != 0.0f && magnitude2 != 0.0f)
    {
        return dotProduct / (magnitude1 * magnitude2);
    }
    else
    {
        return 0.0f;
    }
}

#pragma warning disable 8618
public class EmbeddingData
{
    public EmbeddingsResponse embeddings { get; set; }
    public string text { get; set; }
}

public class ConsolidatedEmbeddingsFileData
{
    public List<EmbeddingData> embeddings { get; set; }
    public string sourceFileName { get; set; }
}
#pragma warning restore 8618