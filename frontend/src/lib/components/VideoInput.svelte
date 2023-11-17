<script lang="ts">
  import 'rvfc-polyfill';
  import { onMount, onDestroy } from 'svelte';
  import {
    mediaStreamState,
    mediaStreamActions,
    isMediaStreaming,
    MediaStreamStatus,
    onFrameChangeStore
  } from '$lib/mediaStream';

  $: mediaStream = $mediaStreamState.mediaStream;

  let videoEl: HTMLVideoElement;
  let videoFrameCallbackId: number;
  const WIDTH = 512;
  const HEIGHT = 512;

  onDestroy(() => {
    if (videoFrameCallbackId) videoEl.cancelVideoFrameCallback(videoFrameCallbackId);
  });

  function srcObject(node: HTMLVideoElement, stream: MediaStream) {
    node.srcObject = stream;
    return {
      update(newStream: MediaStream) {
        if (node.srcObject != newStream) {
          node.srcObject = newStream;
        }
      }
    };
  }
  async function onFrameChange(now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) {
    const blob = await grapBlobImg();
    onFrameChangeStore.set({ now, metadata, blob });
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }

  $: if ($isMediaStreaming == MediaStreamStatus.CONNECTED) {
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }
  async function grapBlobImg() {
    const canvas = new OffscreenCanvas(WIDTH, HEIGHT);
    const videoW = videoEl.videoWidth;
    const videoH = videoEl.videoHeight;
    const aspectRatio = WIDTH / HEIGHT;

    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    ctx.drawImage(
      videoEl,
      videoW / 2 - (videoH * aspectRatio) / 2,
      0,
      videoH * aspectRatio,
      videoH,
      0,
      0,
      WIDTH,
      HEIGHT
    );
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 1 });
    return blob;
  }
</script>

<div class="relative z-10 aspect-square w-full object-cover">
  <video
    class="aspect-square w-full object-cover"
    bind:this={videoEl}
    playsinline
    autoplay
    muted
    loop
    use:srcObject={mediaStream}
  ></video>
</div>
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
