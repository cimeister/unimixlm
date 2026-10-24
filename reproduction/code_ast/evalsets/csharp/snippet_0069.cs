namespace SFA.DAS.AdminService.Web.Validators.Roatp
{
    using SFA.DAS.AdminService.Common.Validation;
    using System.Threading.Tasks;

    public interface ISearchTermValidator
    {
        Task<ValidationResponse> ValidateSearchTerm(string searchTerm);
    }
}