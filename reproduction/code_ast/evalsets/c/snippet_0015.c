#ifndef __AI100Rasterize__
#define __AI100Rasterize__

/*
 *        Name:	AI100Rasterize.h
 *     Purpose:	Adobe Illustrator 10.0 Rasterize Suite.
 *
 * ADOBE SYSTEMS INCORPORATED
 * Copyright 1986-2007 Adobe Systems Incorporated.
 * All rights reserved.
 *
 * NOTICE:  Adobe permits you to use, modify, and distribute this file 
 * in accordance with the terms of the Adobe license agreement 
 * accompanying it. If you have received this file from a source other 
 * than Adobe, then your use, modification, or distribution of it 
 * requires the prior written permission of Adobe.
 *
 */


/*******************************************************************************
 **
 **	Imports
 **
 **/

#include "AIRasterize.h"

#include "AIHeaderBegin.h"

/*******************************************************************************
 **
 **	Constants
 **
 **/

#define kAI100RasterizeSuite			kAIRasterizeSuite
#define kAIRasterizeSuiteVersion4		AIAPI_VERSION(4)
#define kAI100RasterizeSuiteVersion		kAIRasterizeSuiteVersion4


/*******************************************************************************
 **
 **	Suite
 **
 **/

// Illustrator 10.0 Rasterize suite
typedef struct {

/**********************************************************************
	ComputeArtBounds()
	
	This returns the art bounds for the entire artSet.
	If honorCropBox is true, and there is a crop box, then it will 
	return only what's inside the Illustrator crop box (which is 
	created only if Cropmarks->Make was previously done.)
**********************************************************************/
	AIAPI AIErr (*ComputeArtBounds) ( AIArtSet artSet, 
									  AIRealRect *artBounds, 
									  AIBoolean honorCropBox );

/**********************************************************************
	Rasterize()
	
	This creates a raster object (type kRasterArt) from the art contained
	in artSet. You must pass in the artBounds which you can get from 
	calling ComputeArtBounds(). paintOrder and prepArt determine where
	the new kRasterArt object will be created. A pointer to the AIArtHandle
	is returned as raster.
**********************************************************************/
	AIAPI AIErr (*Rasterize) ( AIArtSet artSet, 
							   AIRasterizeSettings *settings, 
							   AIRealRect *artBounds, 
							   short paintOrder, 
							   AIArtHandle prepArt, 
							   AIArtHandle *raster,
							   AIRasterizeProgressProc progressProc );

/**********************************************************************
	CreateMask()
	
	This will create a clip group containing:
	   1. A unionized version of whatever was in artSet (this is the clipArt)
	   2. Whatever kRasterArt you pass as rasterArt
	After the clip group is created, it is placed in another group and 
	this group is returned as maskGroup. You specify paintOrder to indicate
	where the maskGroup should be placed. The default value for paintOrder 
	should be kPlaceAbove.
**********************************************************************/
	AIAPI AIErr (*CreateMask) ( AIArtSet artSet, 
								short paintOrder, 
								AIArtHandle rasterArt, 
								AIArtHandle *maskGroup );

/**********************************************************************
	RasterizeWithPadding()
	
	Almost identical to the Rasterize call above, except adds the
	specified padding to resulting raster. The padding parameter is in
	document points.
**********************************************************************/
	AIAPI AIErr (*RasterizeWithPadding) ( AIArtSet artSet, 
										  AIRasterizeSettings *settings, 
										  AIRealRect *artBounds, 
										  short paintOrder, 
										  AIArtHandle prepArt, 
										  AIArtHandle *raster,
										  AIRasterizeProgressProc progressProc,
										  AIReal padding );

/**********************************************************************
	RasterizeDocument()
	
	This creates a raster object (type kRasterArt) for the current
	document. paintOrder and prepArt determine where the new kRasterArt
	object will be created. A pointer to the AIArtHandle is returned
	as raster.
**********************************************************************/
	AIAPI AIErr (*RasterizeDocument) ( AIRasterizeSettings *settings, 
							   short paintOrder, 
							   AIArtHandle prepArt, 
							   AIArtHandle *raster,
							   AIRasterizeProgressProc progressProc,
							   long options );

} AI100RasterizeSuite;


#include "AIHeaderEnd.h"


#endif