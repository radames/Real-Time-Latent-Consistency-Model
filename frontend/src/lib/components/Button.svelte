<script lang="ts">
  export let classList: string = '';
</script>

<button class="button {classList}" on:click>
  <slot />
</button>

<style lang="postcss">
  .button {
    @apply rounded bg-gray-700 p-2 font-normal text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-300 dark:disabled:bg-gray-700 dark:disabled:text-black;
  }
</style>
