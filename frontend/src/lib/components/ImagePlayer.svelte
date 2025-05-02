<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from "$lib/lcmLive";
  import { getPipelineValues } from "$lib/store";

  import Button from "$lib/components/Button.svelte";
  import Floppy from "$lib/icons/floppy.svelte";
  import Expand from "$lib/icons/expand.svelte";
  import { snapImage, expandWindow } from "$lib/utils";

  let isLCMRunning = $derived(
    $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED &&
      $lcmLiveStatus !== LCMLiveStatus.ERROR,
  );

  let imageEl: HTMLImageElement | undefined = $state();
  let expandedWindow: Window | undefined = $state();
  let isExpanded = $state(false);

  async function takeSnapshot() {
    if (isLCMRunning && imageEl) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt as string,
        negative_prompt: getPipelineValues()?.negative_prompt as string,
        seed: getPipelineValues()?.seed as number,
        guidance_scale: getPipelineValues()?.guidance_scale as number,
      });
    }
  }
  async function toggleFullscreen() {
    if (isLCMRunning && !isExpanded) {
      expandedWindow = expandWindow("/api/stream/" + $streamId);
      expandedWindow.addEventListener("beforeunload", () => {
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
  {#if $lcmLiveStatus === LCMLiveStatus.CONNECTING}
    <!-- Show connecting spinner -->
    <div class="flex h-full w-full items-center justify-center">
      <div
        class="h-16 w-16 animate-spin rounded-full border-b-2 border-white"
      ></div>
      <p class="ml-2 text-white">Connecting...</p>
    </div>
  {:else if isLCMRunning}
    {#if !isExpanded}
      <!-- Handle image error by adding onerror event -->
      <!-- svelte-ignore a11y_missing_attribute -->
      <img
        bind:this={imageEl}
        class="aspect-square w-full rounded-lg"
        src={"/api/stream/" + $streamId}
        onerror={(e) => {
          console.error("Image stream error:", e);
          // If stream fails to load, set status to error
          if ($lcmLiveStatus !== LCMLiveStatus.ERROR) {
            lcmLiveStatus.set(LCMLiveStatus.ERROR);
          }
        }}
      />
    {/if}
    <div class="absolute bottom-1 right-1">
      <Button
        onclick={toggleFullscreen}
        title="Expand Fullscreen"
        class="ml-auto rounded-lg p-1 text-sm text-white opacity-50 shadow-lg"
      >
        <Expand />
      </Button>
      <Button
        onclick={takeSnapshot}
        disabled={!isLCMRunning}
        title="Take Snapshot"
        class="ml-auto rounded-lg p-1 text-sm text-white opacity-50 shadow-lg"
      >
        <Floppy />
      </Button>
    </div>
  {:else if $lcmLiveStatus === LCMLiveStatus.ERROR}
    <!-- Show error state with red border -->
    <div
      class="flex h-full w-full items-center justify-center rounded-lg border-2 border-red-500 bg-gray-900"
    >
      <p class="p-4 text-center text-white">Connection error</p>
    </div>
  {:else}
    <!-- svelte-ignore a11y_missing_attribute -->
    <img
      class="aspect-square w-full rounded-lg"
      src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    />
  {/if}
</div>
