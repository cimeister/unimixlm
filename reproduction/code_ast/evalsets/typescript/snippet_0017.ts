// Representation of payment plans used in SICO
export interface IPaymentPlan {
  intCuota: number;
  dtmFechaPago: Date;
  numValor: number;
  numRestante: number;
}