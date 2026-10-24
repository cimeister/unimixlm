using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Web;
using Fhr.ModernHistory.Utils;
using static System.Net.Mime.MediaTypeNames;

namespace Fhr.ModernHistory.Services.Impl
{
      /// <summary>
      /// 图片服务实现
      /// 2017/07/07 fhr
      /// </summary>
      public class PictureServiceClass : IPictureService
      {
            //图片根目录
            private static readonly string PICTURE_ROOT = System.Configuration.ConfigurationManager.AppSettings["pictureroot"];
            //人员文件夹
            private static readonly string PERSON_DIR = "person";
            //事件文件夹
            private static readonly string EVENT_DIR = "event";

            public Stream GetEventImageFile(object eventId)
            {
                  var fileName = string.Format("{0}\\{1}\\{2}.jpg",PICTURE_ROOT, EVENT_DIR, eventId);
                  if (!File.Exists(fileName))
                  {
                        return null;
                  }
                  return new FileStream(fileName, FileMode.Open);
            }

            public Stream GetPersonImageFile(object personId)
            {
                  var fileName = string.Format("{0}\\{1}\\{2}.jpg", PICTURE_ROOT, PERSON_DIR, personId);
                  if (!File.Exists(fileName))
                  {
                        return null;
                  }
                  return new FileStream(fileName, FileMode.Open);
            }

            public void SaveEventImageFile(MultipartFileData eventImg, object eventId)
            {
                  var fileName = string.Format("{0}\\{1}\\{2}.jpg", PICTURE_ROOT, EVENT_DIR, eventId);
                  saveImg(eventImg, fileName);
            }

            public void SavePersonImagFile(MultipartFileData personImg, object personId)
            {
                  var fileName = string.Format("{0}\\{1}\\{2}.jpg", PICTURE_ROOT, PERSON_DIR, personId);
                  saveImg(personImg, fileName);
            }

            private static void saveImg(MultipartFileData personImg, string fileName)
            {
                  if (File.Exists(fileName))
                  {
                        File.Delete(fileName);
                  }
                  var desFile = new FileInfo(fileName);
                  if (!desFile.Directory.Exists)
                  {
                        desFile.Directory.Create();
                  }
                  File.Copy(personImg.LocalFileName, fileName);
                  File.Delete(personImg.LocalFileName);
            }
      }
}