declare namespace ntt {
    
    type KeyArray = Array<KeyType>;

    type KeyType = number | string;
    
    export type EntityModel<T extends FlatObject, K extends keyof T> = Omit<T, K>;

    export type EntityKey<T extends FlatObject, K extends keyof T> = Pick<T, K>;

    export type PropertyType<T extends SimpleObject, K extends keyof T, P extends T[K]> = P;

    export type ComplexPropertyType<T extends SimpleObject, K extends keyof T, P extends FlatObject & T[K]> = P;

    export type ValuePropertyType<T extends SimpleObject, K extends keyof T, P extends ValueType & T[K]> = P;

    export interface IStore<T extends SimpleObject, K extends keyof T> {
        // Scope the target points to (domain, namespace)
        scope: 'global' | 'individual' | 'network' | 'organization' | 'region';

        // Target within the scope
        target: string;

        create: (model: T) => Promise<boolean>;

        destroy: (key: Pick<T, K>) => Promise<boolean>;

        discard: (key: Pick<T, K>) => Promise<boolean>;

        retrieve: (key: Pick<T, K>) => Promise<T>;

        update: (model: Partial<T>, key: Pick<T, K>) => Promise<T>;
    }

    /*
    export interface IDocumentStore<T extends DocumentObject, K extends keyof T> {
        erase: (key: K) => Promise<boolean>;

        erase: (key: K, props: Array<Exclude<keyof T, K>>) => Promise<T>;

        fetch: (key: K) => T;

        merge: (partial: Partial<Omit<T, K>>) => Promise<T>;

        save: (document: T) => Promise<boolean>;
    }

    export interface IEntityStore<T extends FlatObject, K extends keyof T> {
        create: (model: Omit<T, K>) => Promise<boolean>;

        create: (model: T) => Promise<boolean>;

        discard: (key: Pick<T, K>) => Promise<boolean>;

        retrieve: (key: Pick<T, K>) => Promise<T>;

        update: (model: Partial<Omit<T, K>> & Pick<T, K>) => Promise<T>;

        update: (model: Partial<Omit<T, K>>, key: Pick<T, K>) => Promise<T>;
    }

    export type pickFn<T extends SimpleObject, props extends keyof T> = (document: T) => Pick<T, props> & FlatObject;

    export type extractFn<T extends SimpleObject, prop extends keyof T> = (document: T) => T[prop] & FlatObject;

    export type excludeFn<T extends SimpleObject, props extends keyof T> = (document: T) => Omit<T, props> & FlatObject;

    export interface ITransform<T extends SimpleObject, key extends keyof T, prop extends keyof T> {
        split: (document: T) => [
            Exclude<T, prop>,
            Pick<T, key> & T[prop] & FlatObject
        ];
    }

    export interface ITransform2<T extends SimpleObject, key extends keyof T, prop1 extends keyof T, prop2 extends keyof T> {
        split: (document: T) => [
            Exclude<T, prop1 | prop2>,
            Pick<T, key> & T[prop1] & FlatObject,
            Pick<T, key> & T[prop2] & FlatObject
        ];
    }

    interface Y extends SimpleObject {
        a: boolean;
        b: FlatObject;
        c: number;
        d: FlatObject;
    }

    interface X extends ITransform2<Y, 'c', 'a', 'd'> {

    }
    */
}