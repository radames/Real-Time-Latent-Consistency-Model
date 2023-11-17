export const enum FieldType {
    range = "range",
    seed = "seed",
    textarea = "textarea",
    checkbox = "checkbox",
}
export const enum PipelineMode {
    image = "image",
    video = "video",
    text = "text",
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
    input_mode: {
        default: PipelineMode;
    }
}