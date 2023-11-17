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

<video
  class="aspect-square w-full object-cover"
  bind:this={videoEl}
  playsinline
  autoplay
  muted
  loop
  use:srcObject={mediaStream}
></video>
