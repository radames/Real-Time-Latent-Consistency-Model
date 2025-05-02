<script lang="ts">
  import type { FieldProps } from "$lib/types";
  import { onMount } from "svelte";
  let { value = $bindable(), params }: { value: boolean; params: FieldProps } =
    $props();

  onMount(() => {
    value = params?.default ? true : false;
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center justify-items-start gap-3">
  <label class="text-sm font-medium" for={params.id}>{params?.title}</label>
  <input
    bind:checked={value}
    type="checkbox"
    id={params.id}
    class="cursor-pointer"
  />
</div>
