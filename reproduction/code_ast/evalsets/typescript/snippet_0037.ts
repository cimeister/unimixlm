import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Role } from 'src/users/user.model';
import { UserService } from 'src/users/users.service';
import { IHouse } from './house.model';

@Injectable()
export class HouseService {
  constructor(
    @InjectModel('House') private readonly houseModel: Model<IHouse>,
    private readonly userService: UserService,
  ) {}

  async create(flatNo: number): Promise<IHouse> {
    const house = new this.houseModel({ flatNo: flatNo });
    const result = await house.save();
    return result;
  }

  async getAll(): Promise<IHouse[]> {
    const result = await this.houseModel.find().exec();
    return result;
  }

  async getHouses(ids: string[]): Promise<IHouse[]> {
    return await this.houseModel.find({ _id: { $in: ids } });
  }

  async getVacantHouses(role: Role): Promise<IHouse[]> {
    const occupieds: IHouse[] = (await this.userService.getUsersForRole(role)).map(
      u => u.house,
    ) as IHouse[];
    return await this.houseModel.find({
      _id: { $nin: occupieds.map(h => h._id) },
    });
  }

  async deleteHouses(ids: string[]): Promise<number> {
    const result = await this.houseModel.deleteMany({ _id: { $in: ids } });
    return result.deletedCount;
  }
}