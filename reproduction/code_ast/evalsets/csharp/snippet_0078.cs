using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SprintCreator.Common
{
    public class TFSConnectionSettings
    {
        public bool ConnectAutomatically { get; set; }
        public bool SaveUserNameAndPassword { get; set; }
        public string TfsServer { get; set; }
        public string TfsProject { get; set; }
        public string Domain { get; set; }
        public string UserName { get; set; }
        public string DefectWorkItemType { get; set; }
        public string Password { get; set; }
        public Configuration ConfigurationSetting { get; set; }
        AppSettingsSection AppSettingsSection { get; set; }

        public TFSConnectionSettings()
        {
            this.ConnectAutomatically = false;
            this.SaveUserNameAndPassword = false;
            this.Domain = string.Empty;
            this.UserName = string.Empty;
            this.Password = string.Empty;
            this.TfsServer = string.Empty;
            this.TfsProject = string.Empty;
            this.DefectWorkItemType = "Bug";
        }

        public TFSConnectionSettings(string tfsServer, string tfsProject, string domain, string userName, string password, bool saveUserPwd, bool autoConnect, string defectWorkItemType = "Bug")
        {
            this.ConnectAutomatically = autoConnect;
            this.SaveUserNameAndPassword = saveUserPwd;
            this.Domain = domain;
            this.UserName = userName;
            this.Password = password;
            this.TfsServer = tfsServer;
            this.TfsProject = tfsProject;
            this.DefectWorkItemType = defectWorkItemType;
        }

    }
}