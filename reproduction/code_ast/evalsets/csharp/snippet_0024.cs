using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using BulkReplace.Console.CommandArgs;
using CsvHelper;
using PowerArgs;

namespace BulkReplace.Console
{
    [TabCompletion]
    [ArgExceptionBehavior(ArgExceptionPolicy.StandardExceptionHandling)]
    [ArgDescription("Bulk replace a set of strings")]
    [ArgExample("bulkreplace ", "replace a set of strings in a file")]
    public class Controller
    {
        [HelpHook, ArgShortcut("-?"), ArgDescription("Shows this help")]
        public bool Help { get; set; }

        [ArgActionMethod, ArgDescription("replace list of strings in file"), ArgShortcut("f")]
        public async Task FromFile(FromFileArgs args)
        {
            IEnumerable<Replacement> replacements;

            // read list of replacement strings
            using (var reader = new StreamReader(args.ReplacementsFile))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                csv.Configuration.HeaderValidated = null;
                csv.Configuration.HasHeaderRecord = false;

                replacements = csv.GetRecords<Replacement>().ToList();
            }

            // read file
            string text = await System.IO.File.ReadAllTextAsync(args.TargetFile);

            foreach(var replacement in replacements)
            {
                text = text.Replace(replacement.Token, replacement.Target);
            }

            string outputFile = args.OutputFile;

            // write to replacement file
            await System.IO.File.WriteAllTextAsync(outputFile, text);
        }
    }

    public class Replacement
    {
        public string Token { get; set; }
        public string Target { get; set; }
    }
}