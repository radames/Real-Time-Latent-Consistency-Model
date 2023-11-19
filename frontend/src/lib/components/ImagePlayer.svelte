<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import Button from '$lib/components/Button.svelte';
  import { snapImage } from '$lib/utils';

  $: {
    console.log('streamId', $streamId);
  }
  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;
  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl);
    }
  }
</script>

<div class="flex flex-col">
  <div class="relative overflow-hidden rounded-lg border border-slate-300">
    <!-- svelte-ignore a11y-missing-attribute -->
    {#if isLCMRunning}
      <img
        bind:this={imageEl}
        class="aspect-square w-full rounded-lg"
        src={'/stream/' + $streamId}
      />
    {:else}
      <div class="aspect-square w-full rounded-lg" />
    {/if}
    <div class="absolute left-0 top-0 aspect-square w-1/4">
      <slot />
    </div>
  </div>
  <Button on:click={takeSnapshot} disabled={!isLCMRunning} classList={'ml-auto'}>Snapshot</Button>
</div>
