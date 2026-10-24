using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using System;
using WalletService;

namespace FunctionalTests
{
    public class WalletTestsStartup : Startup
    {
        public WalletTestsStartup(IConfiguration env) : base(env)
        {
        }

        public override IServiceProvider ConfigureServices(IServiceCollection services)
        {
            // Added to avoid the Authorize data annotation in test environment. 
            // Property "SuppressCheckForUnhandledSecurityMetadata" in appsettings.json
            services.Configure<RouteOptions>(Configuration);
            return base.ConfigureServices(services);
        }
    }
}