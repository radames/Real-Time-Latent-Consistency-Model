<script lang="ts">
  import { onMount } from 'svelte';
  import { PUBLIC_BASE_URL } from '$env/static/public';
  import type { FieldProps } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';

  let pipelineParams: FieldProps[];
  let pipelineValues = {};

  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    const settings = await fetch(`${PUBLIC_BASE_URL}/settings`).then((r) => r.json());
    pipelineParams = Object.values(settings.properties);
    pipelineParams = pipelineParams.filter((e) => e?.disabled !== true);
  }

  $: {
    console.log('PARENT', pipelineValues);
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
  <h2 class="font-medium">Prompt</h2>
  <p class="text-sm text-gray-500">
    Change the prompt to generate different images, accepts <a
      href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
      target="_blank"
      class="text-blue-500 underline hover:no-underline">Compel</a
    > syntax.
  </p>
  <PipelineOptions {pipelineParams} bind:pipelineValues></PipelineOptions>
  <div class="flex gap-3">
    <Button>Start</Button>
    <Button>Stop</Button>
    <Button>Snapshot</Button>
  </div>

  <ImagePlayer>
    <VideoInput></VideoInput>
  </ImagePlayer>
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
</style>
