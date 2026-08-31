import { System } from "./system.entity";
export declare class Clause {
    CLAUSE_ID: number;
    CLAUSE_NUMBER: number;
    CLAUSE_DESCRIPTION: string;
    STATUS: string;
    CREATE_TIMESTAMP: Date;
    CREATE_USER: string;
    updated_at: Date;
    UPDATE_USER: string;
    CLAUSE_TITLE: string;
    SYSTEMID: number;
    sytem: System;
}