using System;
using System.Collections.Generic;
using System.Text;

namespace StegBMP
{
    struct HEADER_STEGBMP
    {
        public string Identifier;
        public int Version;
        public int nFiles;
        public FILE_PROPERTY[] FileProperties;
    }

    struct FILE_PROPERTY
    {
        public long FileSize;
        public string FileName;
    }
}