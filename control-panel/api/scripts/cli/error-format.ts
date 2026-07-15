export function toError(error: unknown): Error {
    if (error instanceof Error) {
        return error;
    }

    return new Error(typeof error === 'string' ? error : String(error));
}

export function getErrorMessage(error: unknown): string {
    return toError(error).message;
}
