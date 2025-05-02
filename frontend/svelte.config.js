import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      pages: "public",
      assets: "public",
      fallback: undefined,
      precompress: false,
      strict: true,
    }),
  },
};

export default config;
