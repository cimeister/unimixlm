// Name: SeaOfThieves, Version: 2.0.23

#include "../pch.h"

/*!!DEFINE!!*/

/*!!HELPER_DEF!!*/

/*!!HELPER_INC!!*/

#ifdef _MSC_VER
	#pragma pack(push, 0x01)
#endif

namespace CG
{
//---------------------------------------------------------------------------
// Functions
//---------------------------------------------------------------------------

// Function BP_Shipwright.BP_Shipwright_C.UserConstructionScript
// (Event, Public, BlueprintCallable, BlueprintEvent)
void ABP_Shipwright_C::UserConstructionScript()
{
	static auto fn = UObject::FindObject<UFunction>("Function BP_Shipwright.BP_Shipwright_C.UserConstructionScript");

	ABP_Shipwright_C_UserConstructionScript_Params params;

	auto flags = fn->FunctionFlags;

	UObject::ProcessEvent(fn, &params);
	fn->FunctionFlags = flags;

}


// Function BP_Shipwright.BP_Shipwright_C.ReceiveBeginPlay
// (Event, Public, BlueprintEvent)
void ABP_Shipwright_C::ReceiveBeginPlay()
{
	static auto fn = UObject::FindObject<UFunction>("Function BP_Shipwright.BP_Shipwright_C.ReceiveBeginPlay");

	ABP_Shipwright_C_ReceiveBeginPlay_Params params;

	auto flags = fn->FunctionFlags;

	UObject::ProcessEvent(fn, &params);
	fn->FunctionFlags = flags;

}


// Function BP_Shipwright.BP_Shipwright_C.ReceiveEndPlay
// (Event, Public, BlueprintEvent)
// Parameters:
// TEnumAsByte<Engine_EEndPlayReason> EndPlayReason                  (Parm, ZeroConstructor, IsPlainOldData, NoDestructor)
void ABP_Shipwright_C::ReceiveEndPlay(TEnumAsByte<Engine_EEndPlayReason> EndPlayReason)
{
	static auto fn = UObject::FindObject<UFunction>("Function BP_Shipwright.BP_Shipwright_C.ReceiveEndPlay");

	ABP_Shipwright_C_ReceiveEndPlay_Params params;
	params.EndPlayReason = EndPlayReason;

	auto flags = fn->FunctionFlags;

	UObject::ProcessEvent(fn, &params);
	fn->FunctionFlags = flags;

}


// Function BP_Shipwright.BP_Shipwright_C.ExecuteUbergraph_BP_Shipwright
// (HasDefaults)
// Parameters:
// int                            EntryPoint                     (Parm, ZeroConstructor, IsPlainOldData, NoDestructor)
void ABP_Shipwright_C::ExecuteUbergraph_BP_Shipwright(int EntryPoint)
{
	static auto fn = UObject::FindObject<UFunction>("Function BP_Shipwright.BP_Shipwright_C.ExecuteUbergraph_BP_Shipwright");

	ABP_Shipwright_C_ExecuteUbergraph_BP_Shipwright_Params params;
	params.EntryPoint = EntryPoint;

	auto flags = fn->FunctionFlags;

	UObject::ProcessEvent(fn, &params);
	fn->FunctionFlags = flags;

}


void ABP_Shipwright_C::AfterRead()
{
	ANPC::AfterRead();

	READ_PTR_FULL(CargoRunParticipant, UCargoRunParticipantComponent);
	READ_PTR_FULL(NPCDialogComponent, UNPCDialogComponent);
	READ_PTR_FULL(AnimNotifyWwiseEmitter, UAnimNotifyWwiseEmitterComponent);
	READ_PTR_FULL(SolidHits, UCapsuleComponent);
	READ_PTR_FULL(WwiseShipwrightHummingPlay, UWwiseEvent);
	READ_PTR_FULL(WwiseShipwrightHummingStop, UWwiseEvent);
}

void ABP_Shipwright_C::BeforeDelete()
{
	ANPC::BeforeDelete();

	DELE_PTR_FULL(CargoRunParticipant);
	DELE_PTR_FULL(NPCDialogComponent);
	DELE_PTR_FULL(AnimNotifyWwiseEmitter);
	DELE_PTR_FULL(SolidHits);
	DELE_PTR_FULL(WwiseShipwrightHummingPlay);
	DELE_PTR_FULL(WwiseShipwrightHummingStop);
}

}

#ifdef _MSC_VER
	#pragma pack(pop)
#endif