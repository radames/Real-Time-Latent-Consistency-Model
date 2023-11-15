<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = 8.0;
  export let params: FieldProps;
  onMount(() => {
    value = Number(params?.default) ?? 8.0;
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center gap-3">
  <label class="text-sm font-medium" for="guidance-scale">{params?.title}</label>
  <input
    class="col-span-2"
    bind:value
    type="range"
    id="guidance-scale"
    name="guidance-scale"
    min={params?.min}
    max={params?.max}
    step={params?.step ?? 1}
  />
  <input
    type="number"
    step={params?.step ?? 1}
    bind:value
    class="rounded-md border border-gray-700 px-1 py-1 text-center text-xs font-light"
  />
</div>
