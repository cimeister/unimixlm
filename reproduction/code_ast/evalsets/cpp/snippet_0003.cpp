/*----------------------------------------------------------------------------*/
/* Copyright (c) 2019 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

#include "commands/SetBallHolder.h"

SetBallHolder::SetBallHolder(BallHolder * ballholder, double indexer, double magazine) {
  AddRequirements({ballholder});
  m_indexerTarget = indexer;
  m_magazineTarget = magazine;
  m_ballholder = ballholder;
}

// Called when the command is initially scheduled.
void SetBallHolder::Initialize() {
  m_ballholder->SetIndexer(m_indexerTarget);
  m_ballholder->SetMagazine(m_magazineTarget);
}

// Called repeatedly when this Command is scheduled to run
void SetBallHolder::Execute() {}

// Called once the command ends or is interrupted.
void SetBallHolder::End(bool interrupted) {}

// Returns true when the command should end.
bool SetBallHolder::IsFinished() { return true; }