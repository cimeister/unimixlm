using System.Collections.Generic;
using Xunit;
using System.Net;
using RestSharp.Serialization.Json;

namespace RestSharp
{
    public class BedConfigurationTest
    {
        [Fact]
        public void CheckResponseStatus()
        {
            var client = new RestClient("https://localhost:44369/");
            var request = new RestRequest("api/BedConfiguration", Method.GET);

            var response = client.Execute(request);

            Assert.True(response.StatusCode == HttpStatusCode.OK);

        }

        [Fact]
        public void GetData()
        {
            var client = new RestClient("https://localhost:44369/");
            var request = new RestRequest("api/BedConfiguration/10", Method.GET);

            var response = client.Execute(request);
            var deserialize = new JsonDeserializer();
            var output = deserialize.Deserialize<Dictionary<string, string>>(response);

            var result = output["noOfBed"];
            Assert.True(result == "10");
        }

        [Fact]
        public void GetFalseData()
        {
            var client = new RestClient("https://localhost:44369/");
            var request = new RestRequest("api/BedConfiguration/10", Method.GET);

            var response = client.Execute(request);
            var deserialize = new JsonDeserializer();
            var output = deserialize.Deserialize<Dictionary<string, string>>(response);

            var result = output["noOfBed"];
            Assert.False(result == "20");
        }
    }
}