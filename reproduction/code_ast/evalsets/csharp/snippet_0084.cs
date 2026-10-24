using SecretSaanta.Models.Views;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Http;

namespace SecretSantaa.Controllers
{
    public class LinksController : ApiController
    {
        [HttpGet]
        public UserView getReceiver([FromUri] string username, [FromUri] string groupName)
        {
            return new UserView { username = "username" };
        } 

        [HttpPost]
        public void link([FromUri] string groupName)
        {

        }

    }
}