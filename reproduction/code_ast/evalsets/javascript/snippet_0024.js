const httpStatus = require('http-status');
const MarketError = require('../utils/MarketError');
const catchAsync = require('../utils/catchAsync');
const pick = require('../utils/pick');
const { rentalService } = require('../services');
const { serialize } = require('../serialize/rental.serialize');

const createRental = catchAsync(async (req, res) => {
  const result = await rentalService.createRental(req.body);
  res.status(httpStatus.CREATED).send(serialize(result));
});

const getRental = catchAsync(async (req, res) => {
  const filter = pick(req.query, [
    'nome',
    'cnpj',
    'atividades',
    'endereco.cep',
    'endereco.logradouro',
    'endereco.complemento',
    'endereco.bairro',
    'endereco.number',
    'endereco.localidade',
    'endereco.uf',
  ]);
  const options = pick(req.query, ['limit', 'offset']);
  const result = await rentalService.rentalList(filter, options);
  res.status(200).send(result);
});

const getRentalId = catchAsync(async (req, res) => {
  const result = await rentalService.getRentalById(req.params.rentalId);
  if (!result) {
    throw new MarketError(httpStatus.NOT_FOUND, 'Locadora não encontrada');
  }
  res.status(200).send(serialize(result));
});

const updateRental = catchAsync(async (req, res) => {
  const result = await rentalService.updateRentalById(req.params.rentalId, req.body);
  res.status(200).send(serialize(result));
});

const deleteRental = catchAsync(async (req, res) => {
  await rentalService.deleteRentalById(req.params.rentalId);
  res.status(httpStatus.NO_CONTENT).send();
});
module.exports = {
  createRental,
  getRental,
  getRentalId,
  updateRental,
  deleteRental,
};