import { Jimp } from "jimp";

async function test() {
    try {
        const img = new Jimp({ width: 100, height: 100, color: 0xFFFFFFFF });
        const b64 = await img.getBase64("image/png");
        console.log("Success getBase64");
    } catch (e) {
        console.error(e);
    }
}
test();
