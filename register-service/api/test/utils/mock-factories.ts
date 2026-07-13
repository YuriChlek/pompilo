type MethodNames<T> = {
    [K in keyof T]: T[K] extends (...args: any[]) => any ? K : never;
}[keyof T];

type MockedMethods<T, K extends keyof T> = {
    [P in K]: T[P] extends (...args: infer A) => infer R ? jest.Mock<R, A> : T[P];
};

export function createMock<T extends object, K extends MethodNames<T>>(
    methodNames: K[],
    overrides: Partial<T> = {},
): MockedMethods<T, K> & T {
    const mock: Partial<Record<string, jest.Mock>> = {};

    methodNames.forEach(name => {
        mock[name as string] = jest.fn();
    });

    return {
        ...(mock as MockedMethods<T, K>),
        ...(overrides as T),
    };
}

export const createMockRepository = <R extends object>(
    methodNames: Array<MethodNames<R>>,
    overrides: Partial<R> = {},
): jest.Mocked<R> => createMock<R, MethodNames<R>>(methodNames, overrides) as jest.Mocked<R>;
