namespace AssemblyToProcess
{
    using System.Threading.Tasks;

    public class AsyncDisposable : IAsyncDisposable
    {
        public Task DisposeAsync()
        {
            return Task.FromResult(0);
        }

        public string SayMeHelloWorld()
        {
            return "Hello World!";
        }

        public Task DoSomethingAsync()
        {
            return Task.FromResult(0);
        }
    }
}