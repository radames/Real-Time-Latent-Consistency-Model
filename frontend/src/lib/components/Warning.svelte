<script lang="ts">
  let { message = $bindable() }: { message: string } = $props();

  let timeout = $state(0);
  $effect(() => {
    if (message !== "") {
      console.log("message", message);
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        message = "";
      }, 5000);
    }
  });
</script>

{#if message}
  <div role="alert" class="fixed right-0 top-0 m-4">
    <button
      type="button"
      class="w-full"
      onclick={() => (message = "")}
      onkeydown={(e) => e.key === "Enter" && (message = "")}
    >
      <div class="rounded bg-red-800 p-4 text-white">
        {message}
      </div>
      <div class="bar transition-all duration-500" style="width: 0;"></div>
    </button>
  </div>
{/if}

<style lang="postcss" scoped>
</style>
