import { Injectable, NotFoundException } from "@nestjs/common";
import { InjectRepository } from "@nestjs/typeorm";
import { Repository } from "typeorm";
import { HospitalEntity } from "./Hospital.entity";

@Injectable()
export class HospitalService {
  constructor(
    @InjectRepository(HospitalEntity)
    private hospitalRepository: Repository<HospitalEntity>
  ) {
  }

  async getResult(resultId) {

    const resultData = await this.hospitalRepository.findOne(resultId);

    if (resultData) {
      return {
        "resultCode": 200,
        "resultData": resultData,
        "resultMsg": "success"
      };
    } else {
      throw new NotFoundException("not found");
    }
  }
}