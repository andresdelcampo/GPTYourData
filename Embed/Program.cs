﻿using System.Text;
using OpenAI;
using OpenAI.Models;
using System.Text.Json;

namespace GPTYourDataProgram;

class Program
{
    private static OpenAIClient? Api;
    private const string OpenaiApiKeyFileName = "OpenAI-API.key";
    private const string InputsFolder = "Input";
    private const string EmbeddingsFolder = "Embeddings";

    static async Task Main(string[] args)
    {
        string apiKey = "";
        if (File.Exists(OpenaiApiKeyFileName))
            apiKey = File.ReadAllText(OpenaiApiKeyFileName);

        if (apiKey.Length < 50)
        {
            Console.WriteLine("The OpenAI API key is missing or invalid. Please add it to the OpenAI-API.key file");
            return;
        }
        
        Api = new OpenAIClient(apiKey, Model.Ada);
        
        // Create directory to store embeddings
        Directory.CreateDirectory(EmbeddingsFolder);

        // Get all txt files from the folder
        var txtFiles = Directory.GetFiles(InputsFolder, "*.txt");
        
        // Process each file
        foreach (var filePath in txtFiles)
        {
            await ProcessFile(filePath);
        }
    }

    private static async Task ProcessFile(string filePath)
    {
        string input = await File.ReadAllTextAsync(filePath);
        string filename = Path.GetFileName(filePath);

        // Split the input into sections
        const int maxSectionLength = 2048;
        var sections = SplitIntoSections(input, maxSectionLength);

        int partCount = 0;
        foreach (var section in sections)
        {
            await ProcessPart(section, filename, partCount++);
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

    private static async Task ProcessPart(string part, string filename, int partCount)
    {
        // Get the embeddings
        var result = await Api.EmbeddingsEndpoint.CreateEmbeddingAsync(part);

        // Create an object that includes the embeddings, the original text, and the source file name
        var outputObject = new
        {
            embeddings = result,
            text = part,
            sourceFileName = filename
        };

        // Write the embeddings JSON file with unique name in Embeddings folder
        string jsonFileName = $"embed_{filename}_{partCount}.json";
        string jsonFilePath = Path.Combine(EmbeddingsFolder, jsonFileName);

        string json = JsonSerializer.Serialize(outputObject);
        await File.WriteAllTextAsync(jsonFilePath, json);
        Console.WriteLine(json);
    }
}