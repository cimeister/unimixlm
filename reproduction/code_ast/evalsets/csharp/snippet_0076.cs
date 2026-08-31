using System;
using System.Threading.Tasks;
using EsatateApp.Data.Entities;
using EstateApp.web.Interfaces;
using EstateApp.web.Models;

namespace EstateApp.web.Services
{
    public class AccountsService : IAccountsService
    {
        public async Task<ApplicationUser> CreateUserAsync(LoginModel Model)
        {
            throw new NotImplementedException();
        }
    }
}