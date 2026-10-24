using Magia.Application.AgendamentoContext.Commands;
using Magia.Application.AgendamentoContext.Queries;
using Magia.Domain.Core.Notification;
using MediatR;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;

namespace Magia.WebAPI.Controllers
{
    [ApiController]
    [Route("salas")]
    public class SalaController : ApiController
    {
        public SalaController(IDomainNotificationHandler notification,
            IMediator mediator) : base(notification, mediator)
        {
        }

        [HttpPost]
        public async Task<IActionResult> CadastrarSala(NovaSalaCommand command)
        {
            return CustomResponse(await Mediator.Send(command));
        }

        [HttpGet]
        public async Task<IActionResult> ObterSalas()
        {
            return CustomResponse(await Mediator.Send(new ObterSalasQuery()));
        }
    }
}