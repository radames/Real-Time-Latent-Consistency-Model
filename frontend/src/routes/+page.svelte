<script lang="ts">
  import { onMount } from 'svelte';
  import { PUBLIC_BASE_URL } from '$env/static/public';
  import type { FieldProps, PipelineInfo } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import { isLCMRunning, lcmLiveState, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import {
    mediaStreamState,
    mediaStreamActions,
    isMediaStreaming,
    onFrameChangeStore
  } from '$lib/mediaStream';

  let pipelineParams: FieldProps[];
  let pipelineInfo: PipelineInfo;
  let pipelineValues = {};

  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    const settings = await fetch(`${PUBLIC_BASE_URL}/settings`).then((r) => r.json());
    pipelineParams = Object.values(settings.input_params.properties);
    pipelineInfo = settings.info.properties;
    pipelineParams = pipelineParams.filter((e) => e?.disabled !== true);
    console.log('PARAMS', pipelineParams);
    console.log('SETTINGS', pipelineInfo);
  }

  // $: {
  //   console.log('isLCMRunning', $isLCMRunning);
  // }
  // $: {
  //   console.log('lcmLiveState', $lcmLiveState);
  // }
  // $: {
  //   console.log('mediaStreamState', $mediaStreamState);
  // }
  // $: if ($lcmLiveState.status === LCMLiveStatus.CONNECTED) {
  //   lcmLiveActions.send(pipelineValues);
  // }
  onFrameChangeStore.subscribe(async (frame) => {
    if ($lcmLiveState.status === LCMLiveStatus.CONNECTED) {
      lcmLiveActions.send(pipelineValues);
      lcmLiveActions.send(frame.blob);
    }
  });
  let startBt: Button;
  let stopBt: Button;
  let snapShotBt: Button;

  async function toggleLcmLive() {
    if (!$isLCMRunning) {
      await mediaStreamActions.enumerateDevices();
      await mediaStreamActions.start();
      lcmLiveActions.start();
    } else {
      mediaStreamActions.stop();
      lcmLiveActions.stop();
    }
  }
  async function startLcmLive() {
    try {
      $isLCMRunning = true;
      // const res = await lcmLive.start();
      $isLCMRunning = false;
      // if (res.status === "timeout")
      // toggleMessage("success")
    } catch (err) {
      console.log(err);
      // toggleMessage("error")
      $isLCMRunning = false;
    }
  }
  async function stopLcmLive() {
    // await lcmLive.stop();
    $isLCMRunning = false;
  }
</script>

<div class="fixed right-2 top-2 max-w-xs rounded-lg p-4 text-center text-sm font-bold" id="error" />
<main class="container mx-auto flex max-w-4xl flex-col gap-3 px-4 py-4">
  <article class="flex- mx-auto max-w-xl text-center">
    <h1 class="text-3xl font-bold">Real-Time Latent Consistency Model</h1>
    <p class="text-sm">
      This demo showcases
      <a
        href="https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7"
        target="_blank"
        class="text-blue-500 underline hover:no-underline">LCM</a
      >
      Image to Image pipeline using
      <a
        href="https://github.com/huggingface/diffusers/tree/main/examples/community#latent-consistency-pipeline"
        target="_blank"
        class="text-blue-500 underline hover:no-underline">Diffusers</a
      > with a MJPEG stream server.
    </p>
    <p class="text-sm">
      There are <span id="queue_size" class="font-bold">0</span> user(s) sharing the same GPU,
      affecting real-time performance. Maximum queue size is 4.
      <a
        href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
        target="_blank"
        class="text-blue-500 underline hover:no-underline">Duplicate</a
      > and run it on your own GPU.
    </p>
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
    <PipelineOptions {pipelineParams} bind:pipelineValues></PipelineOptions>
    <div class="flex gap-3">
      <Button on:click={toggleLcmLive}>
        {#if $isLCMRunning}
          Stop
        {:else}
          Start
        {/if}
      </Button>
      <Button disabled={$isLCMRunning} classList={'ml-auto'}>Snapshot</Button>
    </div>

    <ImagePlayer>
      <VideoInput></VideoInput>
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
