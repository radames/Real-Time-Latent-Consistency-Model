<script lang="ts">
  let {
    change,
    aspectRatio = $bindable(),
  }: { change: (aspectRatio: number) => void; aspectRatio: number } = $props();

  let options: string[] = ["1:1", "16:9", "4:3", "3:2", "3:4", "9:16"];

  function onChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    const value = target.value;
    const [width, height] = value.split(":").map((v) => parseInt(v));
    aspectRatio = width / height;
    change(aspectRatio);
  }
</script>

<div class="relative">
  <select
    onchange={onChange}
    title="Aspect Ratio"
    class="block cursor-pointer rounded-md border border-gray-800/50 bg-slate-100/30 p-1 font-medium text-white"
  >
    {#each options as option (option)}
      <option value={option}>{option}</option>
    {/each}
  </select>
</div>
