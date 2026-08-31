import { Injectable, Inject } from '@nestjs/common';
import { Attempts } from '../../entity/attempts.entity';

@Injectable()
export class AttemptsService {
  constructor(
    @Inject('ATTEMPT_REPOSITORY')
    private attemptRepository: typeof Attempts,
  ) {}

  async findAll(): Promise<Attempts[]> {
    return this.attemptRepository.findAll<Attempts>();
  }
}