import { SelectQueryBuilder } from "typeorm";
export declare const paginate: (builder: SelectQueryBuilder<any>, page: number, per_page: number) => Promise<PaginationAwareObject>;
export interface PaginationAwareObject {
    from: any;
    to: any;
    per_page: any;
    total: number | any;
    current_page: number;
    prev_page?: number | null;
    next_page?: number | null;
    last_page: number | null;
    data: Array<object | any> | any;
}