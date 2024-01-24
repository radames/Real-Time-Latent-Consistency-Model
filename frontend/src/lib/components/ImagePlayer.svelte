<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import Expand from '$lib/icons/expand.svelte';
  import { snapImage, expandWindow } from '$lib/utils';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;
  let expandedWindow: Window;
  let isExpanded = false;
  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }
  async function toggleFullscreen() {
    if (isLCMRunning && !isExpanded) {
      expandedWindow = expandWindow('/api/stream/' + $streamId);
      expandedWindow.addEventListener('beforeunload', () => {
        isExpanded = false;
      });
      isExpanded = true;
    } else {
      expandedWindow?.close();
      isExpanded = false;
    }
  }
</script>

<div
  class="relative mx-auto aspect-square max-w-lg self-center overflow-hidden rounded-lg border border-slate-300"
>
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning}
    {#if !isExpanded}
      <img
        bind:this={imageEl}
        class="aspect-square w-full rounded-lg"
        src={'/api/stream/' + $streamId}
      />
    {/if}
    <div class="absolute bottom-1 right-1">
      <Button
        on:click={toggleFullscreen}
        title={'Expand Fullscreen'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Expand classList={''} />
      </Button>
      <Button
        on:click={takeSnapshot}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else}
    <img
      class="aspect-square w-full rounded-lg"
      src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    />
  {/if}
</div>
