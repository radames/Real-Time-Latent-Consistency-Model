declare module "piexifjs" {
  export const ImageIFD: {
    Make: number;
    ImageDescription: number;
    Software: number;
  };
  export const ExifIFD: {
    DateTimeOriginal: number;
  };
  export function dump(exifObj: Record<string, unknown>): string;
  export function insert(exifBytes: string, dataURL: string): string;
}
