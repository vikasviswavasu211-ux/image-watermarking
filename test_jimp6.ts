import { Jimp } from "jimp";

async function test() {
    try {
        const img = new Jimp({ width: 10, height: 10 });
        // @ts-ignore
        const color = img.getPixelColor(0, 0);
        console.log("getPixelColor exists:", color);
        // @ts-ignore
        img.setPixelColor(0xFFFFFFFF, 1, 1);
        console.log("setPixelColor exists");
    } catch (e) {
        console.error("Error:", e);
    }
}
test();
