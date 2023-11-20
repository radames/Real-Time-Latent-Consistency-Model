<script lang="ts">
  import 'rvfc-polyfill';

  import { onDestroy, onMount } from 'svelte';
  import {
    mediaStreamStatus,
    MediaStreamStatusEnum,
    onFrameChangeStore,
    mediaStream,
    mediaDevices
  } from '$lib/mediaStream';
  import MediaListSwitcher from './MediaListSwitcher.svelte';

  let videoEl: HTMLVideoElement;
  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let videoFrameCallbackId: number;
  const WIDTH = 768;
  const HEIGHT = 768;
  // ajust the throttle time to your needs
  const THROTTLE_TIME = 1000 / 15;
  let selectedDevice: string = '';

  onMount(() => {
    ctx = canvasEl.getContext('2d') as CanvasRenderingContext2D;
    canvasEl.width = WIDTH;
    canvasEl.height = HEIGHT;
  });
  $: {
    console.log(selectedDevice);
  }
  onDestroy(() => {
    if (videoFrameCallbackId) videoEl.cancelVideoFrameCallback(videoFrameCallbackId);
  });

  $: if (videoEl) {
    videoEl.srcObject = $mediaStream;
  }
  let lastMillis = 0;
  async function onFrameChange(now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) {
    if (now - lastMillis < THROTTLE_TIME) {
      videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
      return;
    }
    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;
    const blob = await grapCropBlobImg(
      videoEl,
      videoWidth / 2 - WIDTH / 2,
      videoHeight / 2 - HEIGHT / 2,
      WIDTH,
      HEIGHT
    );

    onFrameChangeStore.set({ blob });
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }

  $: if ($mediaStreamStatus == MediaStreamStatusEnum.CONNECTED) {
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }
  async function grapCropBlobImg(
    video: HTMLVideoElement,
    x: number,
    y: number,
    width: number,
    height: number
  ) {
    const canvas = new OffscreenCanvas(width, height);

    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    ctx.drawImage(video, x, y, width, height, 0, 0, width, height);
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 1 });
    return blob;
  }
</script>

<div class="relative mx-auto max-w-lg overflow-hidden rounded-lg border border-slate-300">
  <div class="relative z-10 aspect-square w-full object-cover">
    {#if $mediaDevices.length > 0}
      <div class="absolute bottom-0 right-0 z-10">
        <MediaListSwitcher />
      </div>
    {/if}
    <video
      class="pointer-events-none aspect-square w-full object-cover"
      bind:this={videoEl}
      playsinline
      autoplay
      muted
      loop
    ></video>
    <canvas bind:this={canvasEl} class="absolute left-0 top-0 aspect-square w-full object-cover"
    ></canvas>
  </div>
  <div class="absolute left-0 top-0 flex aspect-square w-full items-center justify-center">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 448" class="w-40 p-5 opacity-20">
      <path
        fill="currentColor"
        d="M224 256a128 128 0 1 0 0-256 128 128 0 1 0 0 256zm-45.7 48A178.3 178.3 0 0 0 0 482.3 29.7 29.7 0 0 0 29.7 512h388.6a29.7 29.7 0 0 0 29.7-29.7c0-98.5-79.8-178.3-178.3-178.3h-91.4z"
      />
    </svg>
  </div>
</div>
