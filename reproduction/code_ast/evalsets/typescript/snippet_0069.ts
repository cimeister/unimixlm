import { OpCodeHandler } from "../OpCodeHandler";
import { Thread } from "../Thread";
import { EvaluationResult } from "../EvaluationResult";
import { OpCode } from "../../common/OpCode";

export class ReadInputHandler extends OpCodeHandler{
    public readonly code: OpCode = OpCode.ReadInput;

    handle(thread:Thread){
        this.logInteraction(thread);
        
        return EvaluationResult.SuspendForInput;
    }
}