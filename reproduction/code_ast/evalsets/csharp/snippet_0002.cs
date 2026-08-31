using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using TestWebApp.Services;

namespace TestWebApp.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ValuesV2Controller : ControllerBase
    {
        //private readonly ITransientService _transient;
        private readonly IScopedService _scoped;
        private readonly ISingletonService _singleton;

        public ValuesV2Controller(ISingletonService singleton, IScopedService scoped)
        {
            //_transient = transient;
            _scoped = scoped;
            _singleton = singleton;
        }
        // GET api/values
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            //var j1 = _transient.DoJob("transient job");
            var j2 = _scoped.DoJob("scoped job");
            var j3 = _singleton.DoJob("singleton job");
            return new string[] { j2, j3 };
        }

        // GET api/values/5
        [HttpGet("{id}")]
        public ActionResult<string> Get(int id)
        {
            return "value";
        }

        // POST api/values
        [HttpPost]
        public void Post([FromBody] string value)
        {
        }

        // PUT api/values/5
        [HttpPut("{id}")]
        public void Put(int id, [FromBody] string value)
        {
        }

        // DELETE api/values/5
        [HttpDelete("{id}")]
        public void Delete(int id)
        {
        }
    }
}