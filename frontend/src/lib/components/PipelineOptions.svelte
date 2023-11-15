<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { FieldProps } from '$lib/types';
  import { FieldType } from '$lib/types';
  import InputRange from './InputRange.svelte';
  import SeedInput from './SeedInput.svelte';
  import TextArea from './TextArea.svelte';

  export let pipelineParams: FieldProps[];
  export let pipelineValues = {} as any;

  $: advanceOptions = pipelineParams?.filter((e) => e?.hide == true);
  $: featuredOptions = pipelineParams?.filter((e) => e?.hide !== true);
</script>

<div>
  {#if featuredOptions}
    {#each featuredOptions as params}
      {#if params.field === FieldType.range}
        <InputRange {params} bind:value={pipelineValues[params.title]}></InputRange>
      {:else if params.field === FieldType.seed}
        <SeedInput bind:value={pipelineValues[params.title]}></SeedInput>
      {:else if params.field === FieldType.textarea}
        <TextArea {params} bind:value={pipelineValues[params.title]}></TextArea>
      {/if}
    {/each}
  {/if}
</div>

<details open>
  <summary class="cursor-pointer font-medium">Advanced Options</summary>
  <div class="flex flex-col gap-3 py-3">
    {#if advanceOptions}
      {#each advanceOptions as params}
        {#if params.field === FieldType.range}
          <InputRange {params} bind:value={pipelineValues[params.title]}></InputRange>
        {:else if params.field === FieldType.seed}
          <SeedInput bind:value={pipelineValues[params.title]}></SeedInput>
        {:else if params.field === FieldType.textarea}
          <TextArea {params} bind:value={pipelineValues[params.title]}></TextArea>
        {/if}
      {/each}
    {/if}
  </div>
</details>
