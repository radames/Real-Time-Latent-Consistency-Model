<script lang="ts">
  import "rvfc-polyfill";

  import { onDestroy, onMount } from "svelte";
  import {
    mediaStreamStatus,
    MediaStreamStatusEnum,
    onFrameChangeStore,
    mediaStream,
    mediaDevices,
  } from "$lib/mediaStream";
  import MediaListSwitcher from "./MediaListSwitcher.svelte";
  import Button from "$lib/components/Button.svelte";
  import Expand from "$lib/icons/expand.svelte";

  let { width = 512, height = 512 }: { width: number; height: number } =
    $props();

  const size = { width, height };

  let videoEl: HTMLVideoElement | undefined = $state();
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let ctx: CanvasRenderingContext2D | undefined = $state();
  let videoFrameCallbackId: number | undefined = $state();

  // ajust the throttle time to your needs
  const THROTTLE = 1000 / 120;
  let selectedDevice: string = $state("");
  let videoIsReady = $state(false);

  onMount(() => {
    if (canvasEl) {
      ctx = canvasEl.getContext("2d") as CanvasRenderingContext2D;
      canvasEl.width = size.width;
      canvasEl.height = size.height;
    }
  });
  $effect(() => {
    console.log(selectedDevice);
  });

  onDestroy(() => {
    if (videoFrameCallbackId && videoEl) {
      videoEl.cancelVideoFrameCallback(videoFrameCallbackId);
    }
  });

  $effect(() => {
    if (videoEl && $mediaStream) {
      videoEl.srcObject = $mediaStream;
    }
  });

  let lastMillis = 0;
  async function onFrameChange(now: DOMHighResTimeStamp) {
    if (!videoEl || !ctx || !canvasEl) return;

    if (now - lastMillis < THROTTLE) {
      videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
      return;
    }

    lastMillis = now;
    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;
    // scale down video to fit canvas,  size.width, size.height
    const scale = Math.min(size.width / videoWidth, size.height / videoHeight);
    const width0 = videoWidth * scale;
    const height0 = videoHeight * scale;
    const x0 = (size.width - width0) / 2;
    const y0 = (size.height - height0) / 2;
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.drawImage(videoEl, x0, y0, width0, height0);

    const blob = await new Promise<Blob>((resolve) => {
      canvasEl?.toBlob(
        (blob) => {
          resolve(blob as Blob);
        },
        "image/jpeg",
        1,
      );
    });
    onFrameChangeStore.set({ blob });
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }

  $effect(() => {
    if (
      $mediaStreamStatus == MediaStreamStatusEnum.CONNECTED &&
      videoIsReady &&
      videoEl
    ) {
      videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
    }
  });

  function toggleFullscreen() {
    if (videoIsReady && videoEl) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        videoEl.requestFullscreen();
      }
    }
  }
</script>

<div
  class="relative mx-auto max-w-lg overflow-hidden rounded-lg border border-slate-300"
>
  <div
    class="relative z-10 flex aspect-square w-full items-center justify-center object-cover"
  >
    {#if $mediaDevices.length > 0}
      <div class="absolute bottom-0 right-0 z-10 flex bg-slate-400/40">
        <MediaListSwitcher />
        <Button
          onclick={toggleFullscreen}
          title="Expand Fullscreen"
          class="ml-auto rounded-lg p-1 text-sm text-white opacity-50 shadow-lg"
        >
          <Expand />
        </Button>
      </div>
    {/if}
    <video
      class="pointer-events-none aspect-square w-full justify-center object-contain"
      bind:this={videoEl}
      onloadeddata={() => {
        videoIsReady = true;
      }}
      playsinline
      autoplay
      muted
      loop
    ></video>
    <canvas
      bind:this={canvasEl}
      class="absolute left-0 top-0 aspect-square w-full object-cover"
    ></canvas>
  </div>
  <div
    class="absolute left-0 top-0 flex aspect-square w-full items-center justify-center"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 448 448"
      class="w-40 p-5 opacity-20"
    >
      <path
        fill="currentColor"
        d="M224 256a128 128 0 1 0 0-256 128 128 0 1 0 0 256zm-45.7 48A178.3 178.3 0 0 0 0 482.3 29.7 29.7 0 0 0 29.7 512h388.6a29.7 29.7 0 0 0 29.7-29.7c0-98.5-79.8-178.3-178.3-178.3h-91.4z"
      />
    </svg>
  </div>
</div>
