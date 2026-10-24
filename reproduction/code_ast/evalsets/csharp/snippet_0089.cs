using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Security.Principal;
using System.Text;

namespace HREntity
{

  public  interface IUserName
    {
        public string UserName { get; set; }
    }

    [Table("User")]
   public class User : IIdentity
    {
        [Key]
        
        public int? Id { get; set; }

        
        public string Name { get; set; }

        public string address { get; set; }

        public string contact { get; set; }
      [Required]
        public string UserName { get; set; }
      [Required]
      
        public string Password { get; set; }

      
        public string Role { get; set; }

        [NotMapped]
        public string AuthenticationType { get; set; }

        [NotMapped]
        public bool IsAuthenticated { get; set; }
    }
}