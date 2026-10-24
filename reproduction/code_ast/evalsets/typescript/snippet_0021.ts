import { Response, Router } from "express";
import AppData from "../../../data/app/AppData";
import { APPROVAL_STATUSES, GRADE_LEVELS, PLATFORMS, PRIVACY_STATUSES, SUBJECTS } from "../../../data/app/appdata-enums";
import AppDataInit from "../../../data/app/AppDataInit";
import { allAreFromArray } from "../../../util/array";
import { ERROR, errorCatcher, resError } from "../errors";
import requiresAuth from "../requiresAuth";
import { dataRes, succesRes } from "../resBuilder";

export const appRoute = Router();


/* Get all apps.
RESPONSE:
    AppDataInit[] */
appRoute.get("", errorCatcher(async (req,res)=>{
    const apps = await AppData.getAllJSON();
    res.status(200).json(dataRes(apps));
}));


/* Get an app by id.
RESPONSE:
    AppDataInit | undefined */
appRoute.get("/:id", errorCatcher<{id:string},unknown,unknown,never,{[key:string]:string}>(async (req,res)=>{
    const app = await AppData.getByIdJSON(req.params.id);
    if (app)
        res.status(200).json(dataRes(app));
    else resError(res,ERROR.appNotFound);
}));


/**
 * Verify that the AppDataInit received is valid.
 * @param data The data to test.
 * @param res Resolve object to throw to if format error.
 * @param required Whether all fields are required.
 * @returns If a type mismatch was found.
 */
function appDataVerify(data:Omit<AppDataInit,"id">|Required<Omit<AppDataInit,"id">>, res:Response, required:boolean):boolean {
    if (data === undefined || data === null) {
        // Data was undefined or null, unacceptable
        resError(res,ERROR.requestBodyInvalid);
        return true;
    }
    if (required) {
        // require all the data to be present
        const {name,url,embed,approval,privacy,platforms,grades,subjects} = data as Required<Omit<AppDataInit,"id">>;
        if (typeof(name)   !== "string" ||
            typeof(url)    !== "string" ||
            typeof(embed)  !== "string" ||
            !allAreFromArray(APPROVAL_STATUSES,[approval ]) ||
            !allAreFromArray(PRIVACY_STATUSES, [privacy  ]) ||
            !allAreFromArray(PLATFORMS,         platforms ) ||
            !allAreFromArray(GRADE_LEVELS,      grades    ) ||
            !allAreFromArray(SUBJECTS,          subjects  )
        ) {
            // something was wrong with the data, throw and return true
            resError(res,ERROR.requestBodyInvalid);
            return true;
        }
    } else {
        // require some the data to be present but of the right type
        const {name,url,embed,approval,privacy,platforms,grades,subjects} = data as Omit<AppDataInit,"id">;
        if (typeof(name   ?? "") !== "string" ||
            typeof(url    ?? "") !== "string" ||
            typeof(embed  ?? "") !== "string" ||
            !allAreFromArray(APPROVAL_STATUSES,[approval  ?? "UNK"]) ||
            !allAreFromArray(PRIVACY_STATUSES, [privacy   ?? "UNK"]) ||
            !allAreFromArray(PLATFORMS,         platforms ?? []    ) ||
            !allAreFromArray(GRADE_LEVELS,      grades    ?? []    ) ||
            !allAreFromArray(SUBJECTS,          subjects  ?? []    )
        ) {
            // something was wrong with the data, throw and return true
            resError(res,ERROR.requestBodyInvalid);
            return true;
        }
    }
    // it was ok, return false.
    return false;
}

/* Add a new app. (Requires authentication)
BODY:
    Required<Omit<AppDataInit,"id">>
RESPONSE:
    AppDataInit */
appRoute.post("", requiresAuth("edit"), errorCatcher(async (req,res)=>{
    if (appDataVerify(req.body, res, true)) return;
    const {name,url,embed,approval,privacy,platforms,grades,subjects} = req.body as Required<Omit<AppDataInit,"id">>;
    const createdApp = (await AppData.createApp({name,url,embed,approval,platforms,grades,privacy,subjects})).toJSON();
    res.status(200).json(dataRes(createdApp));
}));


/* Bulk add new apps. (Requires authentication)
BODY:
    Required<Omit<AppDataInit,"id">>[]
RESPONSE:
    AppDataInit[] */
appRoute.post("/bulk", requiresAuth("edit"), errorCatcher(async (req,res)=>{
    const items = req.body as Required<Omit<AppDataInit,"id">>[];
    if (!(req.body instanceof Array)) {
        resError(res,ERROR.requestBodyInvalid);
        return;
    }
    for (const item of items)
        if (appDataVerify(item,res,true)) return;
    const createdApps = (await AppData.bulkCreateApps(
        items.map(({name,url,embed,approval,platforms,grades,privacy,subjects})=>({name,url,embed,approval,platforms,grades,privacy,subjects}))
    )).map(v=>v.toJSON());
    res.status(200).json(dataRes(createdApps));
}));


/* Add a new app. (Requires authentication)
BODY:
    Partial<Omit<AppDataInit,"id">> */
appRoute.patch("/:id", requiresAuth("edit"), errorCatcher(async (req,res)=>{
    const { id } = req.params;
    const {name,url,embed,approval,privacy,platforms,grades,subjects} = req.body as Omit<AppDataInit,"id">;
    if (appDataVerify(req.body, res, false)) return;
    const patchedApp = (await AppData.patchApp(id,{name,url,embed,approval,privacy,grades,platforms,subjects})).toJSON();
    res.status(200).json(dataRes(patchedApp));
}));


/* Delete an app by id. (Requires authentication) 
RESPONSE:
    *success* */
appRoute.delete("/:id", requiresAuth("edit"), errorCatcher(async (req,res)=>{
    await AppData.deleteApp(req.params.id);
    res.status(200).json(succesRes());
}));