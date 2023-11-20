<script lang="ts">
  import { onMount } from 'svelte';
  import type { FieldProps, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues } from '$lib/store';

  let pipelineParams: FieldProps[];
  let pipelineInfo: PipelineInfo;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    const settings = await fetch('/settings').then((r) => r.json());
    pipelineParams = Object.values(settings.input_params.properties);
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pipelineParams = pipelineParams.filter((e) => e?.disabled !== true);
    if (maxQueueSize > 0) {
      getQueueSize();
      setInterval(() => {
        getQueueSize();
      }, 2000);
    }
  }
  async function getQueueSize() {
    const data = await fetch('/queue_size').then((r) => r.json());
    currentQueueSize = data.queue_size;
  }

  function getSreamdata() {
    if (isImageMode) {
      return [getPipelineValues(), $onFrameChangeStore?.blob];
    } else {
      return [$deboucedPipelineValues];
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;

  let disabled = false;
  async function toggleLcmLive() {
    if (!isLCMRunning) {
      if (isImageMode) {
        await mediaStreamActions.enumerateDevices();
        await mediaStreamActions.start();
      }
      disabled = true;
      await lcmLiveActions.start(getSreamdata);
      disabled = false;
    } else {
      if (isImageMode) {
        mediaStreamActions.stop();
      }
      lcmLiveActions.stop();
    }
  }
</script>

<main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
  <article class="text-center">
    <h1 class="text-3xl font-bold">Real-Time Latent Consistency Model</h1>
    {#if pipelineInfo?.title?.default}
      <h3 class="text-xl font-bold">{pipelineInfo?.title?.default}</h3>
    {/if}
    <p class="text-sm">
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
    <p class="text-sm text-gray-500">
      Change the prompt to generate different images, accepts <a
        href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
        target="_blank"
        class="text-blue-500 underline hover:no-underline">Compel</a
      > syntax.
    </p>
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
        user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}
  </article>
  {#if pipelineParams}
    <article class="my-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
      {#if isImageMode}
        <div class="sm:col-start-1">
          <VideoInput></VideoInput>
        </div>
      {/if}
      <div class={isImageMode ? 'sm:col-start-2' : 'col-span-2'}>
        <ImagePlayer />
      </div>
      <div class="sm:col-span-2">
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg my-1 p-2'}>
          {#if isLCMRunning}
            Stop
          {:else}
            Start
          {/if}
        </Button>
        <PipelineOptions {pipelineParams}></PipelineOptions>
      </div>
    </article>
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
