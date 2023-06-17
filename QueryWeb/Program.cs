using System.Diagnostics;
using QueryWeb;

const string LogFileName = "GptYourData.log";
const string InputsFolder = "Input";

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
        return await ProcessFileUpload(file);
    
    // Process a question
    if (string.IsNullOrWhiteSpace(query))
        return Results.Problem("Empty question asked.");

    return await ProcessQuestion(query);
});

app.Run();

async Task<IResult> ProcessFileUpload(IFormFile file)
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

async Task<IResult> ProcessQuestion(string query)
{
    Embeddings embeddings = new Embeddings(LogFileName);
    IResult result = await embeddings.ProcessQuestion(query);
    return result;
}

