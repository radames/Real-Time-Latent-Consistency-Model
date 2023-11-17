
import { writable, type Writable } from 'svelte/store';

export const pipelineValues = writable({} as Record<string, any>);
