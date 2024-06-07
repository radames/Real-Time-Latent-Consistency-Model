import * as piexif from 'piexifjs';

interface IImageInfo {
  prompt?: string;
  negative_prompt?: string;
  seed?: number;
  guidance_scale?: number;
}

export function snapImage(imageEl: HTMLImageElement, info: IImageInfo) {
  try {
    const zeroth: { [key: string]: any } = {};
    const exif: { [key: string]: any } = {};
    const gps: { [key: string]: any } = {};
    zeroth[piexif.ImageIFD.Make] = 'LCM Image-to-Image ControNet';
    zeroth[piexif.ImageIFD.ImageDescription] =
      `prompt: ${info?.prompt} | negative_prompt: ${info?.negative_prompt} | seed: ${info?.seed} | guidance_scale: ${info?.guidance_scale}`;
    zeroth[piexif.ImageIFD.Software] =
      'https://github.com/radames/Real-Time-Latent-Consistency-Model';
    exif[piexif.ExifIFD.DateTimeOriginal] = new Date().toISOString();

    const exifObj = { '0th': zeroth, Exif: exif, GPS: gps };
    const exifBytes = piexif.dump(exifObj);

    const canvas = document.createElement('canvas');
    canvas.width = imageEl.naturalWidth;
    canvas.height = imageEl.naturalHeight;
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    ctx.drawImage(imageEl, 0, 0);
    const dataURL = canvas.toDataURL('image/jpeg');
    const withExif = piexif.insert(exifBytes, dataURL);

    const a = document.createElement('a');
    a.href = withExif;
    a.download = `lcm_txt_2_img${Date.now()}.png`;
    a.click();
  } catch (err) {
    console.log(err);
  }
}

export function expandWindow(streamURL: string): Window {
  const html = `
    <html>
        <head>
            <title>Real-Time Latent Consistency Model</title>
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background-color: black;
                }
            </style>
        </head>
        <body>
            <script>
                let isFullscreen = false;
                window.onkeydown = function(event) {
                    switch (event.code) {
                        case "Escape":
                            window.close();
                            break;
                        case "Enter":
                            if (isFullscreen) {
                                document.exitFullscreen();
                                isFullscreen = false;
                            } else {
                                document.documentElement.requestFullscreen();
                                isFullscreen = true;
                            }
                            break;
                    }
                }
            </script>

            <img src="${streamURL}" style="width: 100%; height: 100%; object-fit: contain;" />
        </body>
    </html>
    `;
  const newWindow = window.open(
    '',
    '_blank',
    'width=1024,height=1024,scrollbars=0,resizable=1,toolbar=0,menubar=0,location=0,directories=0,status=0'
  ) as Window;
  newWindow.document.write(html);
  return newWindow;
}
