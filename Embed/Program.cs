﻿using System.Net;
using System.Text;
using System.Text.Json;
using OpenAI;
using OpenAI.Embeddings;

namespace GPTYourDataEmbed;

class Program
{
    private static OpenAIClient? Api;
    private const string OpenaiApiKeyFileName = ".openai";
    private const string InputsFolder = "Input";
    private const string InputsDoneFolder = "InputDone";
    private const string EmbeddingsFolder = "Embeddings";

    static async Task Main(string[] args)
    {
        Api = new OpenAIClient(OpenAIAuthentication.LoadFromDirectory("."));
        
        // Create directory to store embeddings and to move the input files once done
        Directory.CreateDirectory(EmbeddingsFolder);
        Directory.CreateDirectory(InputsDoneFolder);

        // Get all txt files from the folder
        var txtFiles = Directory.GetFiles(InputsFolder, "*.txt");
        
        // Process each file
        foreach (var filePath in txtFiles)
        {
            bool processed = await ProcessFile(filePath);
            if (!processed) break;  // API call failed, abort
        }
    }

    private static async Task<bool> ProcessFile(string filePath)
    {
        Console.Write($"Processing {filePath}... ");

        string input = await File.ReadAllTextAsync(filePath);
        string filename = Path.GetFileName(filePath);

        // Split the input into sections
        const int maxSectionLength = 2048;
        var sections = SplitIntoSections(input, maxSectionLength);

        List<object> embeddingsList = new List<object>();
        
        foreach (var section in sections)
        {
            EmbeddingsResponse? result;
            try
            {
                result = await Api!.EmbeddingsEndpoint.CreateEmbeddingAsync(section);
            }
            catch (HttpRequestException e)
            {
                if (e.StatusCode == HttpStatusCode.Unauthorized)
                {
                    Console.WriteLine($"The OpenAI API key is invalid. Please add it to the {OpenaiApiKeyFileName} file");
                    return false;
                }
                throw;
            }

            // Create an object that includes the embeddings, the original text
            var embeddingObject = new
            {
                embeddings = result,
                text = section
            };
            
            embeddingsList.Add(embeddingObject);
        }

        await WriteAllEmbeddingsToFile(filename, embeddingsList);
        
        Console.WriteLine("DONE");

        // Move processed file to InputDone folder
        string newFilePath = Path.Combine(InputsDoneFolder, filename);
        MoveFile(filePath, newFilePath);
        return true;
    }

    private static void MoveFile(string filePath, string newFilePath)
    {
        try
        {
            File.Move(filePath, newFilePath);
        }
        catch (Exception)
        {
            Console.WriteLine($"ERROR: Could not move {filePath} to {newFilePath}. It will need to be moved manually.");
        }
    }

    private static List<string> SplitIntoSections(string text, int maxSectionLength)
    {
        var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var sections = new List<string>();
        var currentSection = new StringBuilder();

        foreach (var line in lines)
        {
            if (currentSection.Length + line.Length > maxSectionLength && currentSection.Length > 0) 
            {
                // if adding this line would exceed the maxSectionLength, start a new section
                sections.Add(currentSection.ToString());
                currentSection.Clear();
            }

            // simple heuristic: a line followed by an empty line is a header
            if (line.Trim().Length == 0 && currentSection.Length > 0) // we've hit a header, start a new section
            {
                sections.Add(currentSection.ToString());
                currentSection.Clear();
            }
            else
            {
                currentSection.AppendLine(line);
            }
        }

        // Add the final section if it's not empty
        if (currentSection.Length > 0)
        {
            sections.Add(currentSection.ToString());
        }

        return sections;
    }

    private static async Task WriteAllEmbeddingsToFile(string filename, List<object> embeddingsList)
    {
        // Create an object that includes the list of embeddings and the source file name
        var outputObject = new
        {
            sourceFileName = filename,
            embeddings = embeddingsList
        };

        // Write the embeddings JSON file with the filename in Embeddings folder
        string jsonFileName = $"embed_{filename}.json";
        string jsonFilePath = Path.Combine(EmbeddingsFolder, jsonFileName);

        string json = JsonSerializer.Serialize(outputObject);
        await File.WriteAllTextAsync(jsonFilePath, json);
    }
}