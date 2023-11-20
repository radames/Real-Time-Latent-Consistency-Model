<script lang="ts">
  import { mediaDevices, mediaStreamActions } from '$lib/mediaStream';
  import { onMount } from 'svelte';

  let deviceId: string = '';
  $: {
    console.log($mediaDevices);
  }
  $: {
    console.log(deviceId);
  }
  onMount(() => {
    deviceId = $mediaDevices[0].deviceId;
  });
</script>

<div class="text-xs">
  {#if $mediaDevices}
    <select
      bind:value={deviceId}
      on:change={() => mediaStreamActions.switchCamera(deviceId)}
      id="devices-list"
      class="border-1 cursor-pointer rounded-md border-gray-500 border-opacity-50 bg-slate-100 bg-opacity-30 p-1 font-medium text-white"
    >
      {#each $mediaDevices as device, i}
        <option value={device.deviceId}>{device.label}</option>
      {/each}
    </select>
  {/if}
</div>
