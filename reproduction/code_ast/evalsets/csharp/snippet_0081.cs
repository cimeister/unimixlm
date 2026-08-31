//==================================================================
// C# OpenGL Framework (http://www.csharpopenglframework.com)
// Copyright (c) 2005-2006 devDept (http://www.devdept.com)
// All rights reserved.
//
// For more information on this program, please visit: 
// http://www.csharpopenglframework.com
//
// For licensing information, please visit: 
// http://www.csharpopenglframework.com/licensing.html
//==================================================================

using System;
using System.Collections.Generic;
using System.Text;
using OpenGL;
using System.Drawing;
using System.IO;

namespace openglFramework
{
    [Serializable]
    class Point : Entity
    {

        float x, y, z;

        // on-screen projection
        float xProj, yProj;

        public Point(float x, float y, float z, Color color) : base (color)
        {

            this.x = x;
            this.y = y;
            this.z = z;

        }

        public override void Draw()
        {

            // Color selection in base class
            base.Draw();

            gl.Vertex3f(x,y,z);

        }

        public override void DrawForShadow()
        {

            // no shadow for point entity

        }

        void ProjectVerticesOnScreen()
        {

            float winX, winY, winZ;

            Label.Project(x, y, z, out winX, out winY, out winZ);

            xProj = winX;
            yProj = winY;

        }

        public override void BoundingBox(ref float xMin, ref float yMin, ref float zMin, ref float xMax, ref float yMax, ref float zMax)
        {

            if (x < xMin)

                xMin = x;

            if (x > xMax)

                xMax = x;

            if (y < yMin)

                yMin = y;

            if (y > yMax)

                yMax = y;

            if (z < zMin)

                zMin = z;

            if (z > zMax)

                zMax = z; 

        }

    }
}