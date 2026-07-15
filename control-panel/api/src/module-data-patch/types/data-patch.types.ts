export type DataPatchContext = {
    client: import('pg').PoolClient;
    logger: {
        info(message: string): void;
        warn(message: string): void;
        error(message: string): void;
    };
    env: NodeJS.ProcessEnv;
};

export type DataPatch = {
    name: string;
    description?: string;
    apply(context: DataPatchContext): Promise<void>;
};
