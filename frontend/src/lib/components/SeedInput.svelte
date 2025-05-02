<script lang="ts">
  import Button from "./Button.svelte";
  import { onMount } from "svelte";
  import type { FieldProps } from "$lib/types";
  let { value = $bindable(), params }: { value: number; params: FieldProps } =
    $props();

  function randomize() {
    value = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
  }
  onMount(() => {
    value = Number(params?.default ?? "");
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center gap-3">
  <label class="text-sm font-medium" for="seed">Seed</label>
  <input
    bind:value
    type="number"
    id="seed"
    name="seed"
    class="col-span-2 rounded-md border border-gray-700 p-2 text-right font-light dark:text-black"
  />
  <Button onclick={randomize}>Rand</Button>
</div>
