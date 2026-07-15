export const LOGGER_CONFIG = Symbol('LOGGER_CONFIG');

export function getLoggerChannelToken(channel: string): symbol {
    return Symbol.for(`LOGGER_CHANNEL:${channel}`);
}
