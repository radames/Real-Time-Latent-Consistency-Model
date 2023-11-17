<script lang="ts">
  import { onMount } from 'svelte';
  import { PUBLIC_BASE_URL } from '$env/static/public';
  import type { FieldProps, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import {
    mediaStreamActions,
    mediaStreamStatus,
    onFrameChangeStore,
    MediaStreamStatusEnum
  } from '$lib/mediaStream';
  import { pipelineValues } from '$lib/store';

  let pipelineParams: FieldProps[];
  let pipelineInfo: PipelineInfo;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;

  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    const settings = await fetch(`${PUBLIC_BASE_URL}/settings`).then((r) => r.json());
    pipelineParams = Object.values(settings.input_params.properties);
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pipelineParams = pipelineParams.filter((e) => e?.disabled !== true);
    console.log('PARAMS', pipelineParams);
    console.log('SETTINGS', pipelineInfo);
  }
  console.log('isImageMode', isImageMode);

  $: {
    console.log('lcmLiveState', $lcmLiveStatus);
  }
  $: {
    console.log('mediaStreamState', $mediaStreamStatus);
  }
  // send Webcam stream to LCM if image mode
  $: {
    if (
      isImageMode &&
      $lcmLiveStatus === LCMLiveStatus.CONNECTED &&
      $mediaStreamStatus === MediaStreamStatusEnum.CONNECTED
    ) {
      lcmLiveActions.send($pipelineValues);
      lcmLiveActions.send($onFrameChangeStore.blob);
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  // $: {
  //   console.log('onFrameChangeStore', $onFrameChangeStore);
  // }

  // // send Webcam stream to LCM
  // $: {
  //   if ($lcmLiveState.status === LCMLiveStatus.CONNECTED) {
  //     lcmLiveActions.send($pipelineValues);
  //   }
  // }
  async function toggleLcmLive() {
    if (!isLCMRunning) {
      if (isImageMode) {
        await mediaStreamActions.enumerateDevices();
        await mediaStreamActions.start();
      }
      await lcmLiveActions.start();
    } else {
      if (isImageMode) {
        mediaStreamActions.stop();
      }
      lcmLiveActions.stop();
    }
  }
</script>

<div class="fixed right-2 top-2 max-w-xs rounded-lg p-4 text-center text-sm font-bold" id="error" />
<main class="container mx-auto flex max-w-4xl flex-col gap-3 px-4 py-4">
  <article class="flex- mx-auto max-w-xl text-center">
    <h1 class="text-3xl font-bold">Real-Time Latent Consistency Model</h1>
    <p class="py-2 text-sm">
      This demo showcases
      <a
        href="https://huggingface.co/blog/lcm_lora"
        target="_blank"
        class="text-blue-500 underline hover:no-underline">LCM LoRA</a
      >
      Image to Image pipeline using
      <a
        href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/lcm#performing-inference-with-lcm"
        target="_blank"
        class="text-blue-500 underline hover:no-underline">Diffusers</a
      > with a MJPEG stream server.
    </p>
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold">0</span> user(s) sharing the same GPU,
        affecting real-time performance. Maximum queue size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}
  </article>
  {#if pipelineParams}
    <header>
      <h2 class="font-medium">Prompt</h2>
      <p class="text-sm text-gray-500">
        Change the prompt to generate different images, accepts <a
          href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Compel</a
        > syntax.
      </p>
    </header>
    <PipelineOptions {pipelineParams}></PipelineOptions>
    <div class="flex gap-3">
      <Button on:click={toggleLcmLive}>
        {#if isLCMRunning}
          Stop
        {:else}
          Start
        {/if}
      </Button>
      <Button disabled={isLCMRunning} classList={'ml-auto'}>Snapshot</Button>
    </div>

    <ImagePlayer>
      {#if isImageMode}
        <VideoInput></VideoInput>
      {/if}
    </ImagePlayer>
  {:else}
    <!-- loading -->
    <div class="flex items-center justify-center gap-3 py-48 text-2xl">
      <Spinner classList={'animate-spin opacity-50'}></Spinner>
      <p>Loading...</p>
    </div>
  {/if}
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
</style>
