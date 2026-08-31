import {IBaseInfo} from 'app/shared/model/base-info.model';

export interface IHistoryUpdate
{
  id?: number;
  uuid?: string;
  version?: number | null;
  content?: string | null;
  baseInfo?: IBaseInfo | null;
}

export const defaultValue: Readonly<IHistoryUpdate> = {};