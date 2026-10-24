// CheckerBoard.cpp: implementation of the CCheckerBoard class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "RayTracer.h"
#include "RayTracing.h"
#include "CheckerBoard.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CCheckerBoard::CCheckerBoard()
{
	Colour1.Set(1.0f, 0.0f, 0.0f);
	Colour2.Set(0.0f, 1.0f, 0.0f);
}

CCheckerBoard::~CCheckerBoard()
{

}

void CCheckerBoard::InitialiseValues(s_IntersectionInfo *Intersection, CRay *IncomingRay)
{
	// call the base class version first
	CSceneObject::InitialiseValues(Intersection, IncomingRay);

	// here we are going to set CSceneObject::Colour to either Colour1 or Colour2
	// depending on the position we hit the plane

	double SignTest;

	double temp;
	temp = PI / (2.0 * CheckSize);

	// this works if the plane lies on the  xz axes 
	// try putting all three components in
	
	//TODO:
	//
	// This needs to be changed.   I need to transform the plane so as that it is parallel
	// to say the xz plane, then the equation below can just use x and z as y will be unchanging
	//
	// I can't figure out how to do that now so we will have to leave it till later.

	SignTest = sin( Intersection->Point.Get_x() * temp ) 
		* sin( Intersection->Point.Get_z() * temp );

	if (SignTest <= 0.0)
		Colour = Colour1;
	else
		Colour = Colour2;
}