<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { onFrameChangeStore } from '$lib/mediaStream';
  import { PUBLIC_BASE_URL } from '$env/static/public';

  $: {
    console.log('streamId', $streamId);
  }
  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
</script>

<div class="relative overflow-hidden rounded-lg border border-slate-300">
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning}
    <img class="aspect-square w-full rounded-lg" src={PUBLIC_BASE_URL + '/stream/' + $streamId} />
  {:else}
    <div class="aspect-square w-full rounded-lg" />
  {/if}
  <div class="absolute left-0 top-0 aspect-square w-1/4">
    <slot />
  </div>
</div>
