<script lang="ts">
  import { mediaDevices, mediaStreamActions } from "$lib/mediaStream";
  import Screen from "$lib/icons/screen.svelte";
  import AspectRatioSelect from "./AspectRatioSelect.svelte";
  import { onMount } from "svelte";

  let deviceId: string = "";
  let aspectRatio: number = 1;

  onMount(() => {
    deviceId = $mediaDevices[0].deviceId;
  });
</script>

<div
  class="flex items-center justify-center text-xs backdrop-blur-sm backdrop-grayscale"
>
  <AspectRatioSelect
    bind:aspectRatio
    change={(value) => mediaStreamActions.switchCamera(deviceId, value)}
  />
  <button
    title="Share your screen"
    class="my-1 flex cursor-pointer gap-1 rounded-md border border-gray-500/50 bg-slate-100/30 p-1 font-medium text-white"
    onclick={() => mediaStreamActions.startScreenCapture()}
  >
    <span>Share</span>

    <Screen />
  </button>
  {#if $mediaDevices}
    <select
      bind:value={deviceId}
      onchange={() => mediaStreamActions.switchCamera(deviceId, aspectRatio)}
      id="devices-list"
      class="block cursor-pointer rounded-md border border-gray-800/50 bg-slate-100/30 p-1 font-medium text-white"
    >
      {#each $mediaDevices as device (device.deviceId)}
        <option value={device.deviceId}>{device.label}</option>
      {/each}
    </select>
  {/if}
</div>
