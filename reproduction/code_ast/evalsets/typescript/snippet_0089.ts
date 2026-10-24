import Currency from './Currency';
import DomainException from "../../DomainException";

class Money {
    public amount: number;
    public currency: Currency;

    constructor(amount: number, currency: Currency) {
        this.amount = amount;
        this.currency = currency;
    }

    public add(money: Money): Money {
        if (!money.currency.equals(this.currency)) {
            throw new DomainException('Attempted to add a money from a different currency');
        }
        return new Money(this.amount + money.amount, this.currency);
    }

    public times(num: number): Money {
        return new Money(this.amount * num, this.currency);
    }

    public display(): string {
        return this.amount.toLocaleString('en_BR', {
            minimumFractionDigits: 1,
            minimumIntegerDigits: 1,
        });
    }

    public equals(money: Money): boolean {
        return money.amount === this.amount && money.currency.equals(this.currency);
    }
}

export default Money;