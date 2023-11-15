export const enum FieldType {
    range = "range",
    seed = "seed",
    textarea = "textarea",
}

export interface FieldProps {
    default: number | string;
    max?: number;
    min?: number;
    title: string;
    field: FieldType;
    step?: number;
    disabled?: boolean;
    hide?: boolean;
}