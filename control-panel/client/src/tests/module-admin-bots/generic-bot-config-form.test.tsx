import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { GenericBotConfigForm } from '@/features/module-admin-bots/components/generic-bot-config-form';
import type { BotConfigSchema } from '@/features/module-admin-bots/interfaces/admin-bots.interfaces';

describe('GenericBotConfigForm', () => {
    it('renders fixture schemas without custom bot components', () => {
        render(
            <GenericBotConfigForm
                schema={fixtureSchema}
                onSubmit={vi.fn()}
            />,
        );

        expect(screen.getByLabelText('Strategy Name')).toBeInTheDocument();
        expect(screen.getByLabelText('Max Positions')).toBeInTheDocument();
        expect(screen.getByLabelText('Risk Decimal')).toBeInTheDocument();
        expect(screen.getByLabelText('Enabled')).toBeInTheDocument();
        expect(screen.getByLabelText('Mode')).toBeInTheDocument();
        expect(screen.getByLabelText('Symbol')).toBeInTheDocument();
        expect(screen.getByLabelText('Symbols')).toBeInTheDocument();
        expect(screen.getByLabelText('Timeframe')).toBeInTheDocument();
        expect(screen.getByLabelText('Timeframes')).toBeInTheDocument();
        expect(screen.getByLabelText('Secret Reference')).toBeInTheDocument();
        expect(screen.getByText('Risk Object')).toBeInTheDocument();
        expect(screen.getByLabelText('Nested Flag')).toBeInTheDocument();
        expect(screen.getByLabelText('Tags')).toBeInTheDocument();
        expect(screen.queryByLabelText(/raw json/i)).not.toBeInTheDocument();
    });

    it('submits normalized config after backend validation passes', async () => {
        const onValidate = vi.fn().mockResolvedValue([]);
        const onSubmit = vi.fn();
        const user = userEvent.setup();

        render(
            <GenericBotConfigForm
                schema={fixtureSchema}
                onValidate={onValidate}
                onSubmit={onSubmit}
            />,
        );

        fireEvent.change(screen.getByLabelText('Symbols'), {
            target: { value: 'ETHUSDT, BTCUSDT' },
        });
        await user.click(screen.getByRole('button', { name: /validate and save/i }));

        expect(onValidate).toHaveBeenCalledWith(
            expect.objectContaining({
                symbols: ['ETHUSDT', 'BTCUSDT'],
            }),
        );
        expect(onSubmit).toHaveBeenCalledWith(
            expect.objectContaining({
                symbols: ['ETHUSDT', 'BTCUSDT'],
            }),
        );
    });

    it('shows backend validation errors without submitting', async () => {
        const onValidate = vi.fn().mockResolvedValue(['symbols must not be empty']);
        const onSubmit = vi.fn();
        const user = userEvent.setup();

        render(
            <GenericBotConfigForm
                schema={fixtureSchema}
                onValidate={onValidate}
                onSubmit={onSubmit}
            />,
        );

        await user.click(screen.getByRole('button', { name: /validate and save/i }));

        expect(await screen.findByText('symbols must not be empty')).toBeInTheDocument();
        expect(onSubmit).not.toHaveBeenCalled();
    });
});

const fixtureSchema: BotConfigSchema = {
    schema_version: 1,
    sections: [
        {
            key: 'general',
            label: 'General',
            fields: [
                {
                    key: 'strategy_name',
                    type: 'string',
                    label: 'Strategy Name',
                    default: 'fixture',
                },
                {
                    key: 'max_positions',
                    type: 'integer',
                    label: 'Max Positions',
                    default: 3,
                    min: 1,
                },
                {
                    key: 'risk_decimal',
                    type: 'decimal',
                    label: 'Risk Decimal',
                    default: '0.25',
                },
                {
                    key: 'enabled',
                    type: 'boolean',
                    label: 'Enabled',
                    default: true,
                },
                {
                    key: 'mode',
                    type: 'enum',
                    label: 'Mode',
                    allowed: ['dry_run', 'signal_only'],
                    default: 'dry_run',
                },
                {
                    key: 'symbol',
                    type: 'symbol',
                    label: 'Symbol',
                    default: 'ETHUSDT',
                },
                {
                    key: 'symbols',
                    type: 'symbol_list',
                    label: 'Symbols',
                    default: ['ETHUSDT'],
                },
                {
                    key: 'timeframe',
                    type: 'timeframe',
                    label: 'Timeframe',
                    default: '1h',
                },
                {
                    key: 'timeframes',
                    type: 'timeframe_list',
                    label: 'Timeframes',
                    default: ['1h', '4h'],
                },
                {
                    key: 'secret_ref',
                    type: 'secret_ref',
                    label: 'Secret Reference',
                    default: 'telegram',
                },
                {
                    key: 'risk',
                    type: 'object',
                    label: 'Risk Object',
                    fields: [
                        {
                            key: 'nested_flag',
                            type: 'boolean',
                            label: 'Nested Flag',
                            default: false,
                        },
                    ],
                },
                {
                    key: 'tags',
                    type: 'array',
                    label: 'Tags',
                    default: ['alpha'],
                },
            ],
        },
    ],
};
