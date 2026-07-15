export const BREAKPOINTS = {
    tablet: 768,
    laptop: 1024,
    desktop: 1280,
    wide: 1536,
} as const;

export const MEDIA_QUERIES = {
    tabletUp: `@media (min-width: ${BREAKPOINTS.tablet}px)`,
    laptopUp: `@media (min-width: ${BREAKPOINTS.laptop}px)`,
    desktopUp: `@media (min-width: ${BREAKPOINTS.desktop}px)`,
    wideUp: `@media (min-width: ${BREAKPOINTS.wide}px)`,
} as const;

export const CONTAINER_WIDTHS = {
    mobile: '100%',
    tablet: 720,
    laptop: 960,
    desktop: 1200,
    wide: 1440,
} as const;

export const LAYOUT_GUTTERS = {
    mobile: 16,
    tablet: 20,
    laptop: 24,
    desktop: 32,
    wide: 40,
} as const;

export const SECTION_SPACING = {
    mobile: 32,
    tablet: 40,
    laptop: 48,
    desktop: 64,
    wide: 80,
} as const;
