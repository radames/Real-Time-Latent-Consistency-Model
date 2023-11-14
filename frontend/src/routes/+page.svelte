<script lang="ts">
  import { onMount } from 'svelte';
  import { PUBLIC_BASE_URL } from '$env/static/public';

  onMount(() => {
    getSettings();
  });
  async function getSettings() {
    const settings = await fetch(`${PUBLIC_BASE_URL}/settings`).then((r) => r.json());
    console.log(settings);
  }
</script>

<div class="fixed right-2 top-2 max-w-xs rounded-lg p-4 text-center text-sm font-bold" id="error" />
<main class="container mx-auto flex max-w-4xl flex-col gap-4 px-4 py-4">
  <article class="mx-auto max-w-xl text-center">
    <h1 class="text-3xl font-bold">Real-Time Latent Consistency Model</h1>
    <h2 class="mb-4 text-2xl font-bold">Image to Image</h2>
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
  <div>
    <h2 class="font-medium">Prompt</h2>
    <p class="text-sm text-gray-500">
      Change the prompt to generate different images, accepts <a
        href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
        target="_blank"
        class="text-blue-500 underline hover:no-underline">Compel</a
      > syntax.
    </p>
    <div class="text-normal flex items-center rounded-md border border-gray-700 px-1 py-1">
      <textarea
        type="text"
        id="prompt"
        class="mx-1 w-full px-3 py-2 font-light outline-none dark:text-black"
        title="Prompt, this is an example, feel free to modify"
        placeholder="Add your prompt here..."
        >Portrait of The Terminator with , glare pose, detailed, intricate, full of colour,
        cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details,
        unreal engine 5, cinematic, masterpiece</textarea
      >
    </div>
  </div>
  <div class="">
    <details>
      <summary class="cursor-pointer font-medium">Advanced Options</summary>
      <div class="grid max-w-md grid-cols-3 items-center gap-3 py-3">
        <label class="text-sm font-medium" for="guidance-scale">Guidance Scale </label>
        <input
          type="range"
          id="guidance-scale"
          name="guidance-scale"
          min="1"
          max="30"
          step="0.001"
          value="8.0"
          oninput="this.nextElementSibling.value = Number(this.value).toFixed(2)"
        />
        <output
          class="w-[50px] rounded-md border border-gray-700 px-1 py-1 text-center text-xs font-light"
        >
          8.0</output
        >
        <label class="text-sm font-medium" for="strength">Strength</label>
        <input
          type="range"
          id="strength"
          name="strength"
          min="0.20"
          max="1"
          step="0.001"
          value="0.50"
          oninput="this.nextElementSibling.value = Number(this.value).toFixed(2)"
        />
        <output
          class="w-[50px] rounded-md border border-gray-700 px-1 py-1 text-center text-xs font-light"
        >
          0.5</output
        >
        <label class="text-sm font-medium" for="seed">Seed</label>
        <input
          type="number"
          id="seed"
          name="seed"
          value="299792458"
          class="rounded-md border border-gray-700 p-2 text-right font-light dark:text-black"
        />
        <button
          onclick="document.querySelector('#seed').value = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)"
          class="button"
        >
          Rand
        </button>
      </div>
    </details>
  </div>
  <div class="flex gap-3">
    <button id="start" class="button"> Start </button>
    <button id="stop" class="button"> Stop </button>
    <button id="snap" disabled class="button ml-auto"> Snapshot </button>
  </div>
  <div class="relative overflow-hidden rounded-lg border border-slate-300">
    <img
      id="player"
      class="aspect-square w-full rounded-lg"
      src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    />
    <div class="absolute left-0 top-0 aspect-square w-1/4">
      <video
        id="webcam"
        class="relative z-10 aspect-square w-full object-cover"
        playsinline
        autoplay
        muted
        loop
      />
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 448 448"
        width="100"
        class="absolute top-0 z-0 w-full p-4 opacity-20"
      >
        <path
          fill="currentColor"
          d="M224 256a128 128 0 1 0 0-256 128 128 0 1 0 0 256zm-45.7 48A178.3 178.3 0 0 0 0 482.3 29.7 29.7 0 0 0 29.7 512h388.6a29.7 29.7 0 0 0 29.7-29.7c0-98.5-79.8-178.3-178.3-178.3h-91.4z"
        />
      </svg>
    </div>
  </div>
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
  .button {
    @apply rounded bg-gray-700 p-2 font-normal text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-300 dark:disabled:bg-gray-700 dark:disabled:text-black;
  }
</style>
