<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  let options: string[] = ['1:1', '16:9', '4:3', '3:2', '3:4', '9:16'];
  export let aspectRatio: number = 1;
  const dispatchEvent = createEventDispatcher();

  function onChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    const value = target.value;
    const [width, height] = value.split(':').map((v) => parseInt(v));
    aspectRatio = width / height;
    dispatchEvent('change', aspectRatio);
  }
</script>

<div class="relative">
  <select
    on:change={onChange}
    title="Aspect Ratio"
    class="border-1 block cursor-pointer rounded-md border-gray-800 border-opacity-50 bg-slate-100 bg-opacity-30 p-1 font-medium text-white"
  >
    {#each options as option, i}
      <option value={option}>{option}</option>
    {/each}
  </select>
</div>
