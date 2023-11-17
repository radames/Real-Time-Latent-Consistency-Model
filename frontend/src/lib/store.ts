
import {
    writable, type Writable, get
} from 'svelte/store';

export const pipelineValues = writable({} as Record<string, any>);
export const getPipelineValues = () => get(pipelineValues);