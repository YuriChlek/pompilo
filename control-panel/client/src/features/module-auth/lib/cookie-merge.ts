/**
 * Merges a list of 'Set-Cookie' header strings into an existing 'cookie' header string.
 * It strictly takes only the name-value pair from each 'Set-Cookie' (ignoring attributes like Path, HttpOnly, etc.).
 *
 * @param currentCookieHeader The current 'cookie' header value (e.g., "a=1; b=2")
 * @param setCookies An array of 'Set-Cookie' header values (e.g., ["c=3; Path=/", "a=4; HttpOnly"])
 * @returns A new 'cookie' header string with merged values (e.g., "a=4; b=2; c=3")
 */
export function mergeSetCookiesIntoCookieHeader(
    currentCookieHeader: string | null | undefined,
    setCookies: string[],
): string {
    const cookies = new Map<string, string>();

    // Parse existing cookies from 'cookie' header
    if (currentCookieHeader) {
        for (const item of currentCookieHeader.split(';')) {
            const trimmed = item.trim();
            if (!trimmed) continue;
            const [rawName, ...rawValueParts] = trimmed.split('=');
            if (rawName) {
                cookies.set(rawName.trim(), rawValueParts.join('=').trim());
            }
        }
    }

    // Merge new cookies from 'Set-Cookie' headers
    for (const setCookie of setCookies) {
        // Set-Cookie looks like "name=value; Path=/; HttpOnly"
        // We only want "name=value"
        const mainPart = setCookie.split(';')[0];
        if (!mainPart) continue;

        const [rawName, ...rawValueParts] = mainPart.trim().split('=');
        if (rawName) {
            cookies.set(rawName.trim(), rawValueParts.join('=').trim());
        }
    }

    // Serialize back to 'cookie' header format (a=1; b=2)
    return [...cookies.entries()]
        .map(([name, value]) => `${name}=${value}`)
        .join('; ');
}
