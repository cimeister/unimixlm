import { Request, Response } from 'express';
import { getRepository } from 'typeorm';

import { Folder } from '../database/entities/Folder';
import { File } from '../database/entities/File';

import { HTTP_ADDRESS, HTTP_STATICFOLDER } from '../config/env';

class FolderFileController {
  public async store(request: Request, response: Response) {
    if (!request.file) {
      return response.status(400).json({
        error: 'cannot upload a file that doesnt was attached',
      });
    }

    const { folderId } = request.params;
    const { originalname, size, mimetype, filename } = request.file;

    const [name] = originalname.split('.');
    const accessUrl = `${HTTP_ADDRESS}${HTTP_STATICFOLDER}/${filename}`;

    const folderRepository = getRepository(Folder);
    const fileRepository = getRepository(File);

    const folder = await folderRepository.findOne({ where: { id: folderId } });

    if (!folder) {
      return response.status(400).json({
        error: 'cannot upload a file on a folder that doesnt exist',
      });
    }

    const file = fileRepository.create({
      folder,
      name,
      size,
      accessUrl,
      mimetype,
    });
    await fileRepository.save(file);

    return response.json(file);
  }
}

export default new FolderFileController();