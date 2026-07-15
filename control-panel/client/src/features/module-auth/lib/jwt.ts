type JwtPayload = {
    id: string;
    name: string;
    email: string;
    role: string;
    exp?: number;
    iat?: number;
    [key: string]: unknown;
};

/**
 * Decodes a JWT token without verification.
 * 
 * NOTE: This is used in the Proxy (Middleware) ONLY as a UX optimization
 * (e.g., to trigger a refresh before a request reaches the backend).
 * IT IS NOT A SECURITY GUARANTEE. 
 * Final signature verification MUST always happen on the backend.
 * 
 * @param token JWT token string
 * @returns Decoded payload or null if invalid format
 */
export function decodeJwt(token: string): JwtPayload | null {

    try {
        const parts = token.split('.');
        if (parts.length !== 3) return null;

        const payload = parts[1];
        if (!payload) return null;

        const base64 = payload.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(
            atob(base64)
                .split('')
                .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
                .join('')
        );

        return JSON.parse(jsonPayload) as JwtPayload;
    } catch {
        return null;
    }
}

/**
 * Checks if a JWT token is expired.
 *
 * @param token JWT token string
 * @param bufferSeconds Optional buffer in seconds (default: 0)
 * @returns true if expired or invalid, false otherwise
 */
export function isJwtExpired(token: string, bufferSeconds = 0): boolean {
    const payload = decodeJwt(token);
    if (!payload || !payload.exp) {
        return true;
    }

    const currentTime = Math.floor(Date.now() / 1000);
    return payload.exp < currentTime + bufferSeconds;
}
