import { Jimp } from "jimp";

async function test() {
    try {
        const img = new Jimp({ width: 10, height: 10 });
        console.log("width:", img.width, "height:", img.height);
    } catch (e) {
        console.error("Error:", e);
    }
}
test();
