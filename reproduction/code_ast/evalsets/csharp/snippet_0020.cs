using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using VueExample.Models.SRV6;
using VueExample.Providers.Srv6.Interfaces;

namespace VueExample.Controllers
{
    [Route ("api/[controller]")]
    public class StatisticParameterController : Controller
    {
        private readonly IStatParameterService _statParameterService;
        public StatisticParameterController(IStatParameterService statParameterService) 
        {
           _statParameterService = statParameterService;
        }

        [HttpGet]
        [ProducesResponseType (typeof(List<StatisticParameter>), StatusCodes.Status200OK)]
        [ProducesResponseType (StatusCodes.Status404NotFound)]
        [Route("stage/{stageId:int}")]
        public async Task<IActionResult> GetAllParametersByStageId([FromRoute] int stageId) 
        {
            var statParametersList = await _statParameterService.GetAllParametersByStageId(stageId);
            return statParametersList.Count == 0 ? (IActionResult)NotFound() : Ok(statParametersList);
        }
    }
}