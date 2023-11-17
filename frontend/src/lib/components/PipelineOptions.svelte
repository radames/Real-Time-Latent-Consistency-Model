<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { FieldProps } from '$lib/types';
  import { FieldType } from '$lib/types';
  import InputRange from './InputRange.svelte';
  import SeedInput from './SeedInput.svelte';
  import TextArea from './TextArea.svelte';
  import Checkbox from './Checkbox.svelte';

  export let pipelineParams: FieldProps[];
  export let pipelineValues = {} as any;

  $: advanceOptions = pipelineParams?.filter((e) => e?.hide == true);
  $: featuredOptions = pipelineParams?.filter((e) => e?.hide !== true);
</script>

<div>
  {#if featuredOptions}
    {#each featuredOptions as params}
      {#if params.field === FieldType.range}
        <InputRange {params} bind:value={pipelineValues[params.id]}></InputRange>
      {:else if params.field === FieldType.seed}
        <SeedInput bind:value={pipelineValues[params.id]}></SeedInput>
      {:else if params.field === FieldType.textarea}
        <TextArea {params} bind:value={pipelineValues[params.id]}></TextArea>
      {:else if params.field === FieldType.checkbox}
        <Checkbox {params} bind:value={pipelineValues[params.id]}></Checkbox>
      {/if}
    {/each}
  {/if}
</div>

<details open>
  <summary class="cursor-pointer font-medium">Advanced Options</summary>
  <div class="grid grid-cols-1 items-center gap-3 sm:grid-cols-2">
    {#if advanceOptions}
      {#each advanceOptions as params}
        {#if params.field === FieldType.range}
          <InputRange {params} bind:value={pipelineValues[params.id]}></InputRange>
        {:else if params.field === FieldType.seed}
          <SeedInput bind:value={pipelineValues[params.id]}></SeedInput>
        {:else if params.field === FieldType.textarea}
          <TextArea {params} bind:value={pipelineValues[params.id]}></TextArea>
        {:else if params.field === FieldType.checkbox}
          <Checkbox {params} bind:value={pipelineValues[params.id]}></Checkbox>
        {/if}
      {/each}
    {/if}
  </div>
</details>
