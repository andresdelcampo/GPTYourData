using System.Diagnostics;
using System.Net;
using System.Text;
using System.Text.Json;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Embeddings;
using OpenAI.Models;

namespace QueryWeb;

public class QuestionAnswerer
{
    const string OpenaiApiKeyFileName = ".openai";
    const string EmbeddingsFolder = "Embeddings";
    const double HighSimilarityThreshold = 0.75;     // For determining what documents to include in the context

    private readonly string _logFileName;

    OpenAIClient? _api;

    public QuestionAnswerer(string logFileName)
    {
        _logFileName = logFileName;
    }

    private IResult InitApi()
    {
        try
        {
            _api = new OpenAIClient(OpenAIAuthentication.LoadFromDirectory("."));
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            LogToFile($"Error initializing OpenAI API. Make sure the key is added to the {OpenaiApiKeyFileName} file");
            return Results.Problem("No good matches found.");
        }

        return Results.Ok();
    }
    
    public async Task<IResult> ProcessQuestion(string query)
    {
        // Initialize OpenAI API
        IResult apiResult = InitApi();
        if (apiResult != Results.Ok())
            return apiResult;

        // Create the embedding for the query
        EmbeddingsResponse? queryEmbed;
        try
        {
            queryEmbed = await CallApiWithRetry(query, () => CreateEmbeddingForQuery(query));
        }
        catch (Exception ex)
        {
            return HandleApiException(query, ex);
        }
        
        double[] queryVector = NormalizeEmbeddingsToVector(queryEmbed!); // Normalize the query vector

        List<ConsolidatedEmbeddingsFileData> fileDataList;
        try
        {
            fileDataList = await ReadEmbeddingFiles();
        }
        catch (Exception)
        {
            LogToFile($"Question: {query}, An error occurred while reading the file data.");
            return Results.Problem("An error occurred while reading the file data.");
        }

        var (documentVectors, documents) = SeparateEmbeddingsAndFragmentText(fileDataList);
        var sortedDocuments = SortFragmentsByRelevance(queryVector, documentVectors, documents);
        string context = BuildStringWithMostRelevantDocuments(sortedDocuments);

        // If no good matches were found:
        if (string.IsNullOrWhiteSpace(context))
        {
            LogToFile($"Question: {query}, Answer: No good matches found.");
            return Results.Problem("No good matches found.");
        }

        // Query the model with a retry providing the question and context to obtain the final answer
        string completeQuery = @$"The following information is provided for context: \n\n{context} \n\n Given this information, can you please answer the following question: \n\n ""{query}""?";

        ChatResponse? result;
        try
        {
            var messages = new List<Message>
            {
                new(Role.System, "You are a helpful assistant called 'Ask F&T' which stands for Ask Frameworks and Tools."),
                new(Role.User, completeQuery),
            };
            var chatRequest = new ChatRequest(messages, Model.GPT3_5_Turbo_16K, temperature: 0.1, maxTokens: 512);
            result = await CallApiWithRetry(query, () => _api!.ChatEndpoint.GetCompletionAsync(chatRequest));
        }
        catch (Exception ex)
        {
            return HandleApiException(query, ex);
        }

        return ProcessQueryResult(query, result);
    }
    
    private async Task<EmbeddingsResponse> CreateEmbeddingForQuery(string query)
    {
        EmbeddingsResponse queryEmbed = await _api!.EmbeddingsEndpoint.CreateEmbeddingAsync(query);
        return queryEmbed;
    }

    private IResult HandleApiException(string query, Exception ex)
    {
        if (ex is UnauthorizedAccessException or HttpRequestException { StatusCode: HttpStatusCode.Unauthorized })
        {
            string error = $"The OpenAI API key is invalid. Please add it to the {OpenaiApiKeyFileName} file";
            Console.WriteLine(error);
            return Results.Problem(error);
        }
        
        LogToFile($"Question: {query}, An error occurred contacting the OpenAI service. You may try again in a moment.");
        return Results.Problem("An error occurred contacting the OpenAI service. You may try again in a moment.");
    }
    
    private async Task<List<ConsolidatedEmbeddingsFileData>> ReadEmbeddingFiles()
    {
        var jsonFiles = Directory.GetFiles(EmbeddingsFolder, "*.json");
        var fileDataList = new List<ConsolidatedEmbeddingsFileData>();

        foreach (var filePath in jsonFiles)
        {
            string embeddingsJson = await File.ReadAllTextAsync(filePath);
            var fileData = JsonSerializer.Deserialize<ConsolidatedEmbeddingsFileData>(embeddingsJson);
            Debug.Assert(fileData != null);
            fileDataList.Add(fileData);
        }

        return fileDataList;
    }
    
    private (List<double[]>, List<string>) SeparateEmbeddingsAndFragmentText(List<ConsolidatedEmbeddingsFileData> fileDataList)
    {
        var documentVectors = new List<double[]>();
        var documents = new List<string>();

        foreach (var fileData in fileDataList)
        {
            foreach (var embeddingObject in fileData.embeddings)
            {
                double[] documentVector = embeddingObject.embeddings.Data.SelectMany(datum => datum.Embedding).ToArray();
                documentVector = NormalizeVector(documentVector); // Normalize the document vector
                documentVectors.Add(documentVector);
                documents.Add(embeddingObject.text);
            }
        }

        return (documentVectors, documents);
    }
    
    private List<(string Document, double Similarity)> SortFragmentsByRelevance(double[] queryVector, List<double[]> documentVectors, List<string> documents)
    {
        var similarities = new List<double>();
        for (int i = 0; i < documentVectors.Count; i++)
        {
            similarities.Add(CosineSimilarity(queryVector, documentVectors[i]));
        }

        var sortedDocuments = documents.Zip(similarities, (d, s) => (Document: d, Similarity: s))
            .OrderByDescending(x => x.Similarity)
            .ToList();

        return sortedDocuments;
    }
    
    private string BuildStringWithMostRelevantDocuments(List<(string Document, double Similarity)> sortedDocuments)
    {
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

        return contextBuilder.ToString();
    }
    
    private IResult ProcessQueryResult(string query, ChatResponse? result)
    {
        // And if a result was found:
        if (result != null)
        {
            LogToFile($"Question: {query}, Answer: {result.FirstChoice.Message.Content.TrimStart()}");
            string resultForWeb = System.Net.WebUtility.HtmlEncode(result.FirstChoice.Message.Content.TrimStart()).Replace("\r\n", "<br />").Replace("\n", "<br />");
            return Results.Ok(resultForWeb);
        }

        LogToFile($"Question: {query}, Answer: Not answered.");
        return Results.Problem("Unable to answer your query. The model did not return an answer.");
    }

    async Task<T?> CallApiWithRetry<T>(string query, Func<Task<T>> action)
    {
        const int maxRetries = 3;

        int retries = 0;
        while (true)
        {
            try
            {
                return await action();  // if the operation is successful, return the result
            }
            catch (Exception ex)
            {
                retries++; 
                if (retries >= maxRetries)
                {
                    // Log error and return default value
                    LogToFile($"Question: {query}, Exception: {ex}");
                    throw;
                }

                await Task.Delay(1000 * retries); // wait for a period of time before retrying (exponential backoff)
            }
        }
    }
    
    private double[] NormalizeEmbeddingsToVector(EmbeddingsResponse embed)
    {
        double[] vector = embed.Data.SelectMany(datum => datum.Embedding).ToArray();
        return NormalizeVector(vector);
    }

    private double[] NormalizeVector(double[] vector)
    {
        double length = Math.Sqrt(vector.Sum(x => x * x));
        return vector.Select(x => x / length).ToArray();
    }

    private static double CosineSimilarity(double[] vector1, double[] vector2)
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
    
    private void LogToFile(string content)
    {
        using StreamWriter writer = new StreamWriter(_logFileName, true);
        writer.WriteLine(content);
        writer.WriteLine();
    }
}

#region EmbeddingsFormat

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

#endregion
