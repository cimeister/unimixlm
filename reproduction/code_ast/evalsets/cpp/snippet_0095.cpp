// Copyright Epic Games, Inc. All Rights Reserved.

#include "test27cppGameMode.h"
#include "test27cppCharacter.h"
#include "UObject/ConstructorHelpers.h"

Atest27cppGameMode::Atest27cppGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPersonCPP/Blueprints/ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}