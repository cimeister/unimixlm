"use strict";

import { conf } from "../app";
import { Procedure } from "../classes/Procedure";
import { AuthenticationMethod } from "../types/AuthenticationMethodEnum";
import { Sequence } from "../classes/Sequence";
import { ConfigAreaInterface } from "../types/ConfigInterface";
import { MissingParameterError } from "./../errors/MissingParameterError";
import { NotFoundError } from "./../errors/NotFoundError";

/*
  Arguments list:
    * area
    * x
    * y
*/
export class PlantProcedure extends Procedure {
  public readonly name: string;
  public readonly auth_method: AuthenticationMethod = AuthenticationMethod.IS_OWNER;

  public async callback(ret: string): Promise<any> {
    return new Promise(async (resolve, reject) => {
      const gcode_ary: Array<string> = ret.split(" ");
      if (gcode_ary[0] === "R02") {
        // Command finished successfuly
        this.sequences_to_run.shift();
        return resolve();
      } else if (gcode_ary[0] === "R03") {
        // Command finished with error
        return reject(gcode_ary[0]);
      }
    });
  }

  public async do(): Promise<any> {
    return new Promise(async (resolve, reject) => {
      // Get area object
      if (!this.call.args.area) {
        // throw new MissingParameterError();
      }
      const area: ConfigAreaInterface = conf.areas[this.call.args.area];
      if (!area) {
        // throw new NotFoundError();
      }

      // Run sequences
      // this.sequences_to_run.push((await Sequence.get("takePlanter")).fill({}));
      // this.sequences_to_run.push((await Sequence.get("takeSeed")).fill({}));
      this.sequences_to_run.push((await Sequence.get("plant")).fill({
        x: area.origin.x + this.call.args["x"],
        y: area.origin.y + this.call.args["y"]
      }));

      return resolve();
    });
  }
}