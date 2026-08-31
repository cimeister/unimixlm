using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityModManagerNet;

namespace AutoPauser
{
    public class Settings : UnityModManager.ModSettings
    {
        public bool SaveAfterEveryChange = true;

        public bool AutoPauseOnAreaLoad = false;

        public bool AutoPauseOnDialogFinished = false;

        public bool AutoPauseOnBattleEnd = false;

        public bool AutoPauseOnCharacterScreenOpened = false;

        public bool AutoPauseOnLocalMapOpened = false;

        public bool AutoPauseOnInventoryScreenOpened = false;
        
        public bool AutoPauseOnLootWindowOpened = false;

        public override void Save(UnityModManager.ModEntry modEntry)
        {
            UnityModManager.ModSettings.Save<Settings>(this, modEntry);
        }
    }
}