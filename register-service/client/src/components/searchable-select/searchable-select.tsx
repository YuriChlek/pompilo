'use client';

import clsx from 'clsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import styles from '@/components/searchable-select/styles.module.css';

export type SearchableSelectOption = {
    value: string;
    label: string;
};

type SearchableSelectProps = {
    options: SearchableSelectOption[];
    value: string | string[];
    onChange: (value: string | string[]) => void;
    placeholder: string;
    searchPlaceholder: string;
    emptyLabel: string;
    helperText?: string;
    multiple?: boolean;
};

export function SearchableSelect({
    options,
    value,
    onChange,
    placeholder,
    searchPlaceholder,
    emptyLabel,
    helperText,
    multiple = false,
}: SearchableSelectProps) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [open, setOpen] = useState(false);
    const [search, setSearch] = useState('');
    const selectedValues = useMemo(
        () => (Array.isArray(value) ? value : value ? [value] : []),
        [value],
    );

    useEffect(() => {
        function handleOutsideClick(event: MouseEvent) {
            if (!containerRef.current?.contains(event.target as Node)) {
                setOpen(false);
            }
        }

        document.addEventListener('mousedown', handleOutsideClick);

        return () => {
            document.removeEventListener('mousedown', handleOutsideClick);
        };
    }, []);

    const filteredOptions = useMemo(() => {
        const query = search.trim().toLowerCase();
        const safeOptions = Array.isArray(options) ? options : [];

        if (!query) {
            return safeOptions;
        }

        return safeOptions.filter(option => option.label.toLowerCase().includes(query));
    }, [options, search]);

    const selectedOptions = useMemo(() => {
        const safeOptions = Array.isArray(options) ? options : [];
        return safeOptions.filter(option => selectedValues.includes(option.value));
    }, [options, selectedValues]);

    function handleSelect(optionValue: string) {
        if (multiple) {
            const nextValues = selectedValues.includes(optionValue)
                ? selectedValues.filter(valueItem => valueItem !== optionValue)
                : [...selectedValues, optionValue];

            onChange(nextValues);

            return;
        }

        onChange(optionValue);
        setOpen(false);
    }

    return (
        <div ref={containerRef} className={styles.root}>
            <button
                type="button"
                className={clsx(styles.trigger, {
                    [styles.triggerOpen]: open,
                })}
                onClick={() => setOpen(currentOpen => !currentOpen)}
            >
                {selectedOptions.length > 0 ? (
                    <span className={styles.value}>
                        {multiple ? (
                            selectedOptions.map(option => (
                                <span key={option.value} className={styles.tag}>
                                    {option.label}
                                </span>
                            ))
                        ) : (
                            <span>{selectedOptions[0].label}</span>
                        )}
                    </span>
                ) : (
                    <span className={styles.placeholder}>{placeholder}</span>
                )}
                <span className={styles.chevron}>{open ? '▲' : '▼'}</span>
            </button>

            {open ? (
                <div className={styles.panel}>
                    <input
                        className={styles.search}
                        type="text"
                        value={search}
                        onChange={event => setSearch(event.target.value)}
                        placeholder={searchPlaceholder}
                    />

                    <div className={styles.list}>
                        {filteredOptions.length > 0 ? (
                            filteredOptions.map(option => {
                                const isSelected = selectedValues.includes(option.value);

                                return (
                                    <button
                                        key={option.value}
                                        type="button"
                                        className={clsx(styles.option, {
                                            [styles.optionSelected]: isSelected,
                                        })}
                                        onClick={() => handleSelect(option.value)}
                                    >
                                        <span>{option.label}</span>
                                        <span className={styles.optionMeta}>
                                            {isSelected ? 'Selected' : 'Choose'}
                                        </span>
                                    </button>
                                );
                            })
                        ) : (
                            <div className={styles.empty}>{emptyLabel}</div>
                        )}
                    </div>

                    {helperText ? <div className={styles.footer}>{helperText}</div> : null}
                </div>
            ) : null}
        </div>
    );
}
