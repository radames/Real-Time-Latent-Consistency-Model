<script lang="ts">
  import { isLCMRunning, lcmLiveState, lcmLiveActions } from '$lib/lcmLive';
  import { onFrameChangeStore } from '$lib/mediaStream';
  import { PUBLIC_BASE_URL } from '$env/static/public';

  $: streamId = $lcmLiveState.streamId;
</script>

<div class="relative overflow-hidden rounded-lg border border-slate-300">
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if $isLCMRunning}
    <img class="aspect-square w-full rounded-lg" src={PUBLIC_BASE_URL + '/stream/' + streamId} />
  {:else}
    <div class="aspect-square w-full rounded-lg" />
  {/if}
  <div class="absolute left-0 top-0 aspect-square w-1/4">
    <div class="relative z-10 aspect-square w-full object-cover">
      <slot />
    </div>
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 448 448"
      width="100"
      class="absolute top-0 z-0 w-full p-4 opacity-20"
    >
      <path
        fill="currentColor"
        d="M224 256a128 128 0 1 0 0-256 128 128 0 1 0 0 256zm-45.7 48A178.3 178.3 0 0 0 0 482.3 29.7 29.7 0 0 0 29.7 512h388.6a29.7 29.7 0 0 0 29.7-29.7c0-98.5-79.8-178.3-178.3-178.3h-91.4z"
      />
    </svg>
  </div>
</div>
