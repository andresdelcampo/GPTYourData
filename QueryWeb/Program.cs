using System.Diagnostics;
using System.Net;
using System.Text;
using System.Text.Json;
using OpenAI;
using OpenAI.Completions;
using OpenAI.Embeddings;
using OpenAI.Models;

OpenAIClient? Api;
const string OpenaiApiKeyFileName = ".openai";
const string LogFileName = "GptYourData.log";
const string InputsFolder = "Input";
const string EmbeddingsFolder = "Embeddings";
const double HighSimilarityThreshold = 0.75;     // For determining what documents to include in the context

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/api/gptquery", async (HttpContext httpContext) =>
{
    var formCollection = await httpContext.Request.ReadFormAsync();
    
    string query = formCollection["query"]!;

    // Process a file upload
    var file = formCollection.Files.GetFile("file");
    if (file != null)
    {
        // Check the file size
        if (file.Length > 1024 * 1024)
            return Results.Problem("The uploaded file is too large. Please upload a file smaller than 1MB.", statusCode: 400);

        // Check the file type by looking at the extension
        if (Path.GetExtension(file.FileName).ToLower() != ".txt")
            return Results.Problem("Invalid file type. Please upload a .txt file.", statusCode: 400);

        Directory.CreateDirectory(InputsFolder);
        var filePath = Path.Combine(InputsFolder, file.FileName);
        await using (var stream = File.Create(filePath))
        {
            await file.CopyToAsync(stream);
        }

        // Prepare the embeddings of the new file 
        var process = new Process();
        process.StartInfo.FileName = "Embed.exe";
        process.Start();
        
        return Results.Ok($"File {file.FileName} uploaded.");
    }
    
    // Process a question
    if (string.IsNullOrWhiteSpace(query))
        return Results.BadRequest("Empty question asked.");

    Api = new OpenAIClient(OpenAIAuthentication.LoadFromDirectory("."));

    // Create the embedding for the query
    EmbeddingsResponse? queryEmbed;
    try
    {
        queryEmbed = await Api.EmbeddingsEndpoint.CreateEmbeddingAsync(query);
        Debug.Assert(queryEmbed != null);
    }
    catch (HttpRequestException e)
    {
        if (e.StatusCode == HttpStatusCode.Unauthorized)
        {
            string error = $"The OpenAI API key is invalid. Please add it to the {OpenaiApiKeyFileName} file";
            Console.WriteLine(error);
            return Results.BadRequest(error);
        }
        throw;
    }

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
        LogToFile($"Question: {query}, Answer: No good matches found.");
        return Results.NotFound("No good matches found.");
    }

    // Query the model with a retry providing the question and context to obtain the final answer
    string completeQuery = @$"The following information is provided for context: \n\n{context} \n\n Given this information, can you please answer the following question: \n\n ""{query}""?";
    const int MaxRetries = 3; // set the maximum number of retries
    int retries = 0; // initialize the retry count
    CompletionResult? result = null;

    while (retries < MaxRetries)
    {
        try
        {
            result = await Api.CompletionsEndpoint.CreateCompletionAsync(completeQuery, model: Model.Davinci, temperature: 0.1, maxTokens: 1024);
            break; // if the operation is successful, break out of the loop
        }
        catch (Exception ex)
        {
            retries++; 
            if (retries >= MaxRetries)
            {
                LogToFile($"Question: {query}, Exception: {ex}");
                return Results.Problem("An error occurred contacting the OpenAI service. You may try again in a moment.");
            }

            await Task.Delay(1000 * retries); // wait for a period of time before retrying (exponential backoff)
        }
    }

    // And if a result was found:
    if (result != null)
    {
        LogToFile($"Question: {query}, Answer: {result.ToString().TrimStart()}");
        string resultForWeb = System.Net.WebUtility.HtmlEncode(result.ToString().TrimStart()).Replace("\r\n", "<br />");
        return Results.Ok(resultForWeb);
    }

    LogToFile($"Question: {query}, Answer: Not answered.");
    return Results.NotFound("Unable to answer your query. The model did not return an answer.");
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

    return 0.0f;
}

static void LogToFile(string content)
{
    using StreamWriter writer = new StreamWriter(LogFileName, true);
    writer.WriteLine(content);
    writer.WriteLine();
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