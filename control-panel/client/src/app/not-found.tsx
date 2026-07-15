import { SystemState } from '@/components/system-state/system-state';

export default function NotFoundPage() {
    return (
        <SystemState
            eyebrow="Page not found"
            title="This route is off the map"
            description="The page may have moved, the link may be outdated, or the address may be incorrect."
            tone="not-found"
        />
    );
}
