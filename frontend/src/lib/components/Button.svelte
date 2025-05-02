<script lang="ts">
  let props = $props();
</script>

<button {...props}>
  {@render props.children()}
</button>

<style lang="postcss">
  @reference "tailwindcss";
  button {
    @apply cursor-pointer rounded bg-gray-700 font-normal text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-300 dark:disabled:bg-gray-700 dark:disabled:text-black;
  }
</style>
