export const enum FieldType {
    range = "range",
    seed = "seed",
    textarea = "textarea",
    checkbox = "checkbox",
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
    id: string;
}
export interface PipelineInfo {
    name: string;
    description: string;
    mode: string;
}