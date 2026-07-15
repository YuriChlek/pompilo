declare module '*.svg' {
    import type { ComponentProps, ReactElement } from 'react';

    const SvgComponent: (props: ComponentProps<'svg'>) => ReactElement;

    export default SvgComponent;
}

declare module '*.svg?url' {
    const content: string;

    export default content;
}
