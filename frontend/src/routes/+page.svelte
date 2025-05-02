<script lang="ts">
  import { onMount } from "svelte";
  import type { Fields, PipelineInfo } from "$lib/types";
  import { PipelineMode } from "$lib/types";
  import ImagePlayer from "$lib/components/ImagePlayer.svelte";
  import VideoInput from "$lib/components/VideoInput.svelte";
  import Button from "$lib/components/Button.svelte";
  import PipelineOptions from "$lib/components/PipelineOptions.svelte";
  import Spinner from "$lib/icons/spinner.svelte";
  import Warning from "$lib/components/Warning.svelte";
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from "$lib/lcmLive";
  import { mediaStreamActions, onFrameChangeStore } from "$lib/mediaStream";
  import { getPipelineValues, deboucedPipelineValues } from "$lib/store";

  let pipelineParams: Fields | undefined = $state();
  let pipelineInfo: PipelineInfo | undefined = $state();
  let pageContent: string | undefined = $state();
  let isImageMode: boolean = $state(false);
  let maxQueueSize: number = $state(0);
  let currentQueueSize: number = $state(0);
  let disabled: boolean = $state(false);
  let queueCheckerRunning: boolean = $state(false);
  let warningMessage: string = $state("");

  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    const settings = await fetch("/api/settings").then((r) => r.json());
    pipelineParams = settings.input_params.properties;
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo?.input_mode?.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pageContent = settings.page_content;
    toggleQueueChecker(true);
    console.log(pipelineParams);
  }
  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }
  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    const data = await fetch("/api/queue").then((r) => r.json());
    currentQueueSize = data.queue_size;
    setTimeout(getQueueSize, 10000);
  }
  function getSreamdata():
    | [Record<string, unknown>]
    | [Record<string, unknown>, Blob] {
    if (isImageMode) {
      return [getPipelineValues(), $onFrameChangeStore?.blob];
    } else {
      return [$deboucedPipelineValues];
    }
  }

  const isLCMRunning = $derived(
    $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED &&
      $lcmLiveStatus !== LCMLiveStatus.ERROR,
  );
  const isConnecting = $derived($lcmLiveStatus === LCMLiveStatus.CONNECTING);

  $effect(() => {
    // Set warning messages based on lcmLiveStatus
    if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
      warningMessage = "Session timed out. Please try again.";
    } else if ($lcmLiveStatus === LCMLiveStatus.ERROR) {
      warningMessage = "Connection error occurred. Please try again.";
    }
  });
  async function toggleLcmLive() {
    try {
      if (!isLCMRunning) {
        if (isConnecting) {
          return; // Don't allow multiple connection attempts
        }

        // Clear any previous warning messages
        warningMessage = "";
        disabled = true;

        try {
          if (isImageMode) {
            await mediaStreamActions.enumerateDevices();
            await mediaStreamActions.start();
          }

          await lcmLiveActions.start(getSreamdata);
          toggleQueueChecker(false);
        } finally {
          // Always re-enable the button even if there was an error
          disabled = false;
        }
      } else {
        // Handle stopping - disable button during this process too
        disabled = true;

        try {
          if (isImageMode) {
            mediaStreamActions.stop();
          }
          await lcmLiveActions.stop();
          toggleQueueChecker(true);
        } finally {
          disabled = false;
        }
      }
    } catch (e) {
      console.error("Error in toggleLcmLive:", e);
      warningMessage =
        e instanceof Error ? e.message : "An unknown error occurred";
      disabled = false;
      toggleQueueChecker(true);
    }
  }

  // Reconnect function for automatic reconnection
  async function reconnect() {
    try {
      disabled = true;
      warningMessage = "Reconnecting...";

      if (isImageMode) {
        await mediaStreamActions.stop();
        await mediaStreamActions.enumerateDevices();
        await mediaStreamActions.start();
      }

      await lcmLiveActions.reconnect(getSreamdata);
      warningMessage = "";
      toggleQueueChecker(false);
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : "Reconnection failed";
      toggleQueueChecker(true);
    } finally {
      disabled = false;
    }
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
  <Warning bind:message={warningMessage}></Warning>
  <article class="text-center">
    {#if pageContent}
      <!-- eslint-disable-next-line svelte/no-at-html-tags -->
      {@html pageContent}
    {/if}
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold"
          >{currentQueueSize}</span
        >
        user(s) sharing the same GPU, affecting real-time performance. Maximum queue
        size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}

    {#if $lcmLiveStatus === LCMLiveStatus.ERROR}
      <p class="mt-2 text-sm">
        <button
          class="text-blue-500 underline hover:no-underline"
          onclick={reconnect}
          {disabled}
        >
          Try reconnecting
        </button>
      </p>
    {/if}
  </article>
  {#if pipelineParams}
    <article class="my-3 grid grid-cols-1 gap-3 sm:grid-cols-4">
      {#if isImageMode}
        <div class="col-span-2 sm:col-start-1">
          <VideoInput
            width={Number(pipelineParams.width.default)}
            height={Number(pipelineParams.height.default)}
          ></VideoInput>
        </div>
      {/if}
      <div class={isImageMode ? "col-span-2 sm:col-start-3" : "col-span-4"}>
        <ImagePlayer />
      </div>
      <div class="sm:col-span-4 sm:row-start-2">
        <Button onclick={toggleLcmLive} {disabled} class="my-1 p-2 text-lg">
          {#if isConnecting}
            Connecting...
          {:else if isLCMRunning}
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
      <Spinner class="animate-spin opacity-50"></Spinner>
      <p>Loading...</p>
    </div>
  {/if}
</main>

<style lang="postcss">
  @reference "tailwindcss";
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
</style>
